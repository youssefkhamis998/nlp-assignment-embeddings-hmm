"""Text normalization and vocabulary construction from training sentences."""

from __future__ import annotations

import re
from collections import Counter

UNK_TOKEN = "<UNK>"
NUM_TOKEN = "<NUM>"

_NUMBER_RE = re.compile(r"^[+-]?[\d.,:/\-]*\d[\d.,:/\-]*%?$")
_PUNCT_RE = re.compile(r"^[^\w\s]+$")


def normalize_token(
    token: str,
    lowercase: bool = True,
    normalize_numbers: bool = True,
    drop_punctuation: bool = True,
) -> str | None:
    """Return normalized token or None if it should be dropped."""
    t = token.strip()
    if not t:
        return None
    if drop_punctuation and _PUNCT_RE.match(t):
        return None
    if normalize_numbers and _NUMBER_RE.match(t):
        return NUM_TOKEN
    if lowercase:
        t = t.lower()
    return t


class Vocabulary:
    """Training-set vocabulary mapping normalized tokens <-> integer IDs."""

    def __init__(
        self,
        min_count: int = 2,
        lowercase: bool = True,
        normalize_numbers: bool = True,
        drop_punctuation: bool = True,
    ):
        self.min_count = min_count
        self.lowercase = lowercase
        self.normalize_numbers = normalize_numbers
        self.drop_punctuation = drop_punctuation

        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.counts: list[int] = []
        self.raw_token_count: int = 0
        self.kept_token_count: int = 0

    def normalize(self, token: str) -> str | None:
        return normalize_token(
            token,
            lowercase=self.lowercase,
            normalize_numbers=self.normalize_numbers,
            drop_punctuation=self.drop_punctuation,
        )

    def normalize_sentence(self, tokens: list[str]) -> list[str]:
        out: list[str] = []
        for tok in tokens:
            norm = self.normalize(tok)
            if norm is not None:
                out.append(norm)
        return out

    def fit(self, sentences: list[list[str]]) -> "Vocabulary":
        """Build vocabulary strictly from training sentences."""
        counter: Counter[str] = Counter()
        raw_total = 0
        for sent in sentences:
            raw_total += len(sent)
            for tok in sent:
                norm = self.normalize(tok)
                if norm is not None:
                    counter[norm] += 1

        # Sort by descending frequency, then alphabetically for determinism
        kept = sorted(
            [(w, c) for w, c in counter.items() if c >= self.min_count],
            key=lambda x: (-x[1], x[0]),
        )

        self.idx2word = [w for w, _ in kept]
        self.counts = [c for _, c in kept]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.raw_token_count = raw_total
        self.kept_token_count = sum(self.counts)
        return self

    def encode_sentence(self, tokens: list[str], drop_oov: bool = True) -> list[int]:
        """Convert raw sentence tokens into vocabulary indices."""
        ids: list[int] = []
        for tok in tokens:
            norm = self.normalize(tok)
            if norm is None:
                continue
            idx = self.word2idx.get(norm)
            if idx is not None:
                ids.append(idx)
            elif not drop_oov:
                ids.append(-1)
        return ids

    def __len__(self) -> int:
        return len(self.idx2word)

    def __contains__(self, word: str) -> bool:
        norm = self.normalize(word)
        return norm is not None and norm in self.word2idx