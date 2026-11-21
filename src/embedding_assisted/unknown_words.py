
from __future__ import annotations

import numpy as np

from ..hmm.model import HMMTagger


def _logsumexp2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


class EmbeddingAssistedHMM(HMMTagger):
    def __init__(
        self,
        embedding_model=None,
        embed_weight: float = 0.5,
        top_k: int = 10,
        ctx_window: int = 3,
        **hmm_kwargs,
    ):
        super().__init__(**hmm_kwargs)
        self.embedding_model = embedding_model
        self.embed_weight = embed_weight
        self.top_k = top_k
        self.ctx_window = ctx_window

    def _embedding_emission(self, words: list[str], position: int) -> np.ndarray | None:
        word = words[position]
        em = self.embedding_model

        # --- Try to find neighbours via word similarity first
        neighbours = em.most_similar(word, top_k=self.top_k)

        # --- Fallback: query by context if the word is not in embed vocab
        if not neighbours:
            left = max(0, position - self.ctx_window)
            right = min(len(words), position + self.ctx_window + 1)
            ctx = [w for i, w in enumerate(words[left:right], start=left) if i != position]
            neighbours = em.most_similar_from_context(ctx, top_k=self.top_k)

        if not neighbours:
            return None

        # Keep only neighbours that exist as *known* words in the HMM
        # (i.e. have their own emission column).
        weighted: list[tuple[np.ndarray, float]] = []
        for nbr_word, sim in neighbours:
            if sim <= 0.0:
                continue
            col = self.word2idx.get(nbr_word)
            if col is None:
                col = self.word2idx.get(nbr_word.lower())
            if col is not None:
                weighted.append((self.log_B[:, col], sim))

        if not weighted:
            return None

        total_weight = sum(s for _, s in weighted)
        avg_emit = sum(s / total_weight * np.exp(logb) for logb, s in weighted)
        avg_emit = np.maximum(avg_emit, 1e-300)  # guard underflow
        return np.log(avg_emit)

    def unknown_log_emission(self, words: list[str], position: int) -> np.ndarray:
        baseline = super().unknown_log_emission(words, position)

        if self.embedding_model is None or self.embed_weight <= 0.0:
            return baseline

        emb_logp = self._embedding_emission(words, position)
        if emb_logp is None:
            return baseline

        alpha = self.embed_weight
        # log( (1-α)*exp(baseline) + α*exp(emb_logp) )
        return _logsumexp2(
            np.log(1.0 - alpha) + baseline,
            np.log(alpha) + emb_logp,
        )
