"""Sparse sentence-level TF-IDF word representations."""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict

import numpy as np

from .vocab import Vocabulary


class TFIDFRepresentation:
    """Word-document TF-IDF representation treating each sentence as a document."""

    def __init__(
        self,
        sublinear_tf: bool = False,
        min_count: int = 2,
        lowercase: bool = True,
        normalize_numbers: bool = True,
        drop_punctuation: bool = True,
    ):
        self.sublinear_tf = sublinear_tf
        self.vocab = Vocabulary(
            min_count=min_count,
            lowercase=lowercase,
            normalize_numbers=normalize_numbers,
            drop_punctuation=drop_punctuation,
        )

        # Sparse row storage: word_id -> {sent_id: tfidf_weight}
        self.vectors: list[dict[int, float]] = []
        # Inverted index: sent_id -> [(word_id, tfidf_weight), ...]
        self._inverted: dict[int, list[tuple[int, float]]] = {}
        self.idf: np.ndarray = np.empty(0, dtype=np.float64)
        self.norms: np.ndarray = np.empty(0, dtype=np.float64)
        self.num_docs: int = 0
        self.nnz: int = 0

    def fit(self, sentences: list[list[str]]) -> "TFIDFRepresentation":
        self.vocab.fit(sentences)
        V = len(self.vocab)
        D = len(sentences)
        self.num_docs = D

        # Term frequency per (word_id, sent_id) and document frequency per word_id
        word_doc_tf: list[dict[int, int]] = [{} for _ in range(V)]
        df = np.zeros(V, dtype=np.float64)

        for doc_id, sent in enumerate(sentences):
            ids = self.vocab.encode_sentence(sent, drop_oov=True)
            if not ids:
                continue
            counts = Counter(ids)
            for w_id, cnt in counts.items():
                word_doc_tf[w_id][doc_id] = cnt
                df[w_id] += 1.0

        # Standard IDF: log(D / df(w)); guard against df == 0 or df == D
        self.idf = np.zeros(V, dtype=np.float64)
        for w_id in range(V):
            if 0 < df[w_id] < D:
                self.idf[w_id] = math.log(D / df[w_id])
            elif df[w_id] == D:
                self.idf[w_id] = math.log((D + 1.0) / (df[w_id] + 1.0)) + 1.0

        self.vectors = [{} for _ in range(V)]
        inverted: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self.norms = np.zeros(V, dtype=np.float64)
        nnz = 0

        for w_id in range(V):
            idf_w = float(self.idf[w_id])
            if idf_w == 0.0:
                continue
            row_vec: dict[int, float] = {}
            sq_sum = 0.0
            for doc_id, raw_tf in word_doc_tf[w_id].items():
                tf_val = (1.0 + math.log(raw_tf)) if self.sublinear_tf else float(raw_tf)
                weight = tf_val * idf_w
                row_vec[doc_id] = weight
                sq_sum += weight * weight
                inverted[doc_id].append((w_id, weight))
                nnz += 1
            self.vectors[w_id] = row_vec
            self.norms[w_id] = math.sqrt(sq_sum)

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
        for doc_id, q_val in q_vec.items():
            for cand_id, cand_val in self._inverted.get(doc_id, ()):
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
        """Aggregate sentence-distribution vectors of surrounding words (for Part 3)."""
        q_vec: dict[int, float] = defaultdict(float)
        for cw in context_words:
            cid = self._resolve_id(cw)
            if cid is not None and self.norms[cid] > 0.0:
                inv_n = 1.0 / self.norms[cid]
                for doc_id, val in self.vectors[cid].items():
                    q_vec[doc_id] += val * inv_n
        return self._query_sparse_vector(q_vec, top_k=top_k)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.vocab), self.num_docs)

    @property
    def sparsity(self) -> float:
        V, D = self.shape
        total_cells = V * D
        return 1.0 - (self.nnz / total_cells) if total_cells > 0 else 1.0

    def storage_bytes(self) -> int:
        """Approximate memory footprint in bytes of the sparse representation."""
        total = sys.getsizeof(self.vectors) + self.norms.nbytes + self.idf.nbytes
        for row in self.vectors:
            total += sys.getsizeof(row)
            total += len(row) * (sys.getsizeof(0) + sys.getsizeof(0.0))
        return total