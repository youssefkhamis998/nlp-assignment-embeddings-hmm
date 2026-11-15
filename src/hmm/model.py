"""First-order (bigram) HMM POS tagger estimated from tagged sentences.

Model
-----
    P(tags, words) = P(t_1) * prod_i P(t_i | t_{i-1}) * P(</s> | t_T) * prod_i P(w_i | t_i)

Smoothing
---------
* Initial / transition / end probabilities: add-k (Lidstone) smoothing, so every
  tag bigram has non-zero probability.
* Emissions: words seen more than ``rare_threshold`` times form the known
  vocabulary.  All rarer training tokens are collapsed into a single ``<UNK>``
  emission per tag, so P(<UNK> | t) learns how "open" each tag is (NN, JJ, NNP
  get a lot of mass; DT, IN, CC almost none).  Add-k smoothing is then applied
  over the known vocabulary + <UNK>.

Unknown words
-------------
For a word that is not in the known vocabulary:

    P(w | t) = P(<UNK> | t) * P(signature(w) | <UNK>, t)

where ``signature`` maps a word to a morphological/orthographic class
(capitalisation, digits, hyphens, common suffixes).  Signature distributions are
estimated from the rare training tokens at three granularities (full signature,
shape only, plain UNK) and the most specific level that was observed in training
is used.  Optionally, an unseen word whose lowercase form is known (e.g. a
capitalised sentence-initial word) reuses the lowercase word's emissions.

``unknown_log_emission`` is the single extension point used by the
embedding-assisted taggers in Part 3.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from .viterbi import viterbi

UNK = "<UNK>"

# Ordered longest first so the most specific suffix wins.
SUFFIXES = (
    "ness", "ment", "tion", "sion", "ship", "ance", "ence", "able", "ible", "ical",
    "ious", "ward", "wise",
    "ful", "ous", "ive", "ize", "ise", "ity", "ing", "est", "ers", "ism", "ist",
    "ed", "er", "ly", "al", "ic", "es", "an", "en",
    "s", "y",
)
_NUMBER_RE = re.compile(r"^[+-]?[\d.,/:\-]*\d[\d.,/:\-]*%?$")


def word_shape(word: str, position: int) -> str:
    """Coarse orthographic class of a word."""
    if _NUMBER_RE.match(word):
        return "UNK-NUM"
    parts = ["UNK"]
    if word.isupper() and any(c.isalpha() for c in word):
        parts.append("ALLC")
    elif word[:1].isupper():
        parts.append("INITC" if position == 0 else "CAP")
    elif any(c.isupper() for c in word):
        parts.append("MIXC")
    if any(c.isdigit() for c in word):
        parts.append("DIG")
    if "-" in word:
        parts.append("DASH")
    return "-".join(parts)


def signature_levels(word: str, position: int) -> tuple[str, str, str]:
    """(full signature, shape-only signature, plain UNK) from specific to coarse."""
    shape = word_shape(word, position)
    if shape == "UNK-NUM":
        return shape, shape, UNK
    lower = word.lower()
    suffix = next((s for s in SUFFIXES if lower.endswith(s) and len(lower) > len(s) + 1), "")
    full = f"{shape}-{suffix}" if suffix else shape
    return full, shape, UNK


class HMMTagger:
    def __init__(
        self,
        rare_threshold: int = 1,
        trans_k: float = 0.1,
        emit_k: float = 0.01,
        sig_k: float = 0.1,
        lowercase_backoff: bool = True,
        use_signatures: bool = True,
    ):
        self.rare_threshold = rare_threshold
        self.trans_k = trans_k
        self.emit_k = emit_k
        self.sig_k = sig_k
        self.lowercase_backoff = lowercase_backoff
        self.use_signatures = use_signatures

    # ------------------------------------------------------------------ fit
    def fit(self, sentences: list[list[tuple[str, str]]]) -> "HMMTagger":
        self.word_counts = Counter(w for s in sentences for w, _ in s)
        self.tags = sorted({t for s in sentences for _, t in s})
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        N = len(self.tags)

        known = sorted(w for w, c in self.word_counts.items() if c > self.rare_threshold)
        self.word2idx = {w: i for i, w in enumerate(known)}
        unk_col = len(known)
        V = len(known) + 1  # + <UNK>

        init = np.zeros(N)
        trans = np.zeros((N, N))
        end = np.zeros(N)
        emit = np.zeros((N, V))
        # signature counts per level: level -> sig -> (N,) counts
        sig_counts: list[dict[str, np.ndarray]] = [defaultdict(lambda: np.zeros(N)) for _ in range(3)]

        for sent in sentences:
            prev = None
            for pos, (w, t) in enumerate(sent):
                j = self.tag2idx[t]
                if prev is None:
                    init[j] += 1
                else:
                    trans[prev, j] += 1
                col = self.word2idx.get(w)
                if col is None:
                    emit[j, unk_col] += 1
                    for lvl, sig in enumerate(signature_levels(w, pos)):
                        sig_counts[lvl][sig][j] += 1
                else:
                    emit[j, col] += 1
                prev = j
            if prev is not None:
                end[prev] += 1

        k = self.trans_k
        self.log_pi = np.log((init + k) / (init.sum() + k * N))
        # transitions out of tag i also include the end-of-sentence event
        out_totals = trans.sum(axis=1) + end
        self.log_A = np.log((trans + k) / (out_totals[:, None] + k * (N + 1)))
        self.log_end = np.log((end + k) / (out_totals + k * (N + 1)))

        ke = self.emit_k
        self.log_B = np.log((emit + ke) / (emit.sum(axis=1, keepdims=True) + ke * V))
        self.log_p_unk = self.log_B[:, unk_col]

        # P(sig | <UNK>, tag) per level, add-k smoothed within each level
        rare_per_tag = emit[:, unk_col]
        self.log_sig: list[dict[str, np.ndarray]] = []
        for lvl_counts in sig_counts:
            S = len(lvl_counts)
            denom = rare_per_tag + self.sig_k * S
            self.log_sig.append({
                sig: np.log((c + self.sig_k) / denom) for sig, c in lvl_counts.items()
            })
        return self

    # ------------------------------------------------------------ emissions
    def is_known(self, word: str) -> bool:
        """Word has its own emission column (seen > rare_threshold times)."""
        return word in self.word2idx

    def is_oov(self, word: str) -> bool:
        """Word never appeared in the training data (used for evaluation)."""
        return word not in self.word_counts

    def signature_log_emission(self, word: str, position: int) -> np.ndarray:
        """log P(<UNK>|t) + log P(signature(w) | <UNK>, t) using the most specific seen level."""
        if not self.use_signatures:
            return self.log_p_unk
        for lvl, sig in enumerate(signature_levels(word, position)):
            table = self.log_sig[lvl]
            if sig in table:
                return self.log_p_unk + table[sig]
        return self.log_p_unk  # unreachable in practice: UNK level always exists

    def unknown_log_emission(self, words: list[str], position: int) -> np.ndarray:
        """Emission scores (N,) for a word without its own emission column.

        Override this in subclasses to plug in other unknown-word strategies.
        """
        word = words[position]
        if self.lowercase_backoff:
            col = self.word2idx.get(word.lower())
            if col is not None:
                return self.log_B[:, col]
        return self.signature_log_emission(word, position)

    def emission_matrix(self, words: list[str]) -> np.ndarray:
        T, N = len(words), len(self.tags)
        out = np.empty((T, N))
        for i, w in enumerate(words):
            col = self.word2idx.get(w)
            out[i] = self.log_B[:, col] if col is not None else self.unknown_log_emission(words, i)
        return out

    # ------------------------------------------------------------- decoding
    def predict(self, words: list[str]) -> list[str]:
        path, _ = viterbi(self.log_pi, self.log_A, self.emission_matrix(words), self.log_end)
        return [self.tags[j] for j in path]

    def predict_many(self, sentences: list[list[str]]) -> list[list[str]]:
        return [self.predict(ws) for ws in sentences]
