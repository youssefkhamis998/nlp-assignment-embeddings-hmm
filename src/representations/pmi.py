"""Sparse Pointwise Mutual Information (PMI / PPMI) word representations."""

from __future__ import annotations

import math
import sys
from collections import defaultdict

import numpy as np

from .vocab import Vocabulary


class PMIRepresentation:
    """Word-context co-occurrence matrix weighted by PMI or Positive PMI (PPMI)."""

    def __init__(
        self,
        window_size: int = 4,
        positive: bool = True,
        min_count: int = 2,
        lowercase: bool = True,
        normalize_numbers: bool = True,
        drop_punctuation: bool = True,
    ):
        self.window_size = window_size
        self.positive = positive
        self.vocab = Vocabulary(
            min_count=min_count,
            lowercase=lowercase,
            normalize_numbers=normalize_numbers,
            drop_punctuation=drop_punctuation,
        )

        # Sparse row storage: word_id -> {context_id: pmi_weight}
        self.vectors: list[dict[int, float]] = []
        # Inverted index: context_id -> [(word_id, pmi_weight), ...]
        self._inverted: dict[int, list[tuple[int, float]]] = {}
        self.norms: np.ndarray = np.empty(0, dtype=np.float64)
        self.nnz: int = 0
        self.total_pairs: float = 0.0

    def fit(self, sentences: list[list[str]]) -> "PMIRepresentation":
        self.vocab.fit(sentences)
        V = len(self.vocab)

        cooc: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        row_sums = np.zeros(V, dtype=np.float64)
        col_sums = np.zeros(V, dtype=np.float64)
        total = 0.0

        W = self.window_size
        for sent in sentences:
            ids = self.vocab.encode_sentence(sent, drop_oov=True)
            L = len(ids)
            for i, w in enumerate(ids):
                left = max(0, i - W)
                right = min(L, i + W + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    c = ids[j]
                    cooc[w][c] += 1.0
                    row_sums[w] += 1.0
                    col_sums[c] += 1.0
                    total += 1.0

        self.total_pairs = total
        self.vectors = [{} for _ in range(V)]
        inverted: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self.norms = np.zeros(V, dtype=np.float64)
        nnz = 0

        if total > 0:
            for w, ctx_dict in cooc.items():
                rw = row_sums[w]
                if rw <= 0:
                    continue
                row_vec: dict[int, float] = {}
                sq_sum = 0.0
                for c, count in ctx_dict.items():
                    cc = col_sums[c]
                    if cc <= 0:
                        continue
                    pmi = math.log2((count * total) / (rw * cc))
                    if self.positive and pmi <= 0.0:
                        continue
                    row_vec[c] = pmi
                    sq_sum += pmi * pmi
                    inverted[c].append((w, pmi))
                    nnz += 1
                self.vectors[w] = row_vec
                self.norms[w] = math.sqrt(sq_sum)

        self._inverted = dict(inverted)
        self.nnz = nnz
        return self

    def _resolve_id(self, word: str) -> int | None:
        norm = self.vocab.normalize(word)
        if norm is None:
            return None
        return self.vocab.word2idx.get(norm)

    def get_vector(self, word: str) -> dict[int, float] | None:
        idx = self._resolve_id(word)
        return self.vectors[idx] if idx is not None else None

    def similarity(self, w1: str, w2: str) -> float:
        i1 = self._resolve_id(w1)
        i2 = self._resolve_id(w2)
        if i1 is None or i2 is None:
            return 0.0
        n1, n2 = self.norms[i1], self.norms[i2]
        if n1 == 0.0 or n2 == 0.0:
            return 0.0

        v1, v2 = self.vectors[i1], self.vectors[i2]
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        dot = sum(val * v2.get(k, 0.0) for k, val in v1.items())
        return float(dot / (n1 * n2))

    def _query_sparse_vector(
        self,
        q_vec: dict[int, float],
        top_k: int = 10,
        exclude_id: int | None = None,
    ) -> list[tuple[str, float]]:
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0.0:
            return []

        dots: dict[int, float] = defaultdict(float)
        for ctx_id, q_val in q_vec.items():
            for cand_id, cand_val in self._inverted.get(ctx_id, ()):
                if cand_id != exclude_id:
                    dots[cand_id] += q_val * cand_val

        scored: list[tuple[str, float]] = []
        for cand_id, dot in dots.items():
            c_norm = self.norms[cand_id]
            if c_norm > 0.0:
                scored.append((self.vocab.idx2word[cand_id], dot / (q_norm * c_norm)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def most_similar(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        idx = self._resolve_id(word)
        if idx is None or self.norms[idx] == 0.0:
            return []
        return self._query_sparse_vector(self.vectors[idx], top_k=top_k, exclude_id=idx)

    def most_similar_from_context(
        self, context_words: list[str], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find words that occur in the same local contexts (useful for OOV words in Part 3)."""
        q_vec: dict[int, float] = defaultdict(float)
        for cw in context_words:
            cid = self._resolve_id(cw)
            if cid is not None:
                q_vec[cid] += 1.0
        return self._query_sparse_vector(q_vec, top_k=top_k)

    @property
    def shape(self) -> tuple[int, int]:
        V = len(self.vocab)
        return (V, V)

    @property
    def sparsity(self) -> float:
        V = len(self.vocab)
        total_cells = V * V
        return 1.0 - (self.nnz / total_cells) if total_cells > 0 else 1.0

    def storage_bytes(self) -> int:
        """Approximate memory footprint in bytes of the sparse representation."""
        total = sys.getsizeof(self.vectors) + self.norms.nbytes
        for row in self.vectors:
            total += sys.getsizeof(row)
            # int key + float value overhead in Python dict
            total += len(row) * (sys.getsizeof(0) + sys.getsizeof(0.0))
        return total