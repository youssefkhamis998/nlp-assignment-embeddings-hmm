"""Skip-Gram with Negative Sampling (Word2Vec) implemented from scratch in NumPy."""

from __future__ import annotations

import numpy as np

from .vocab import Vocabulary


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_clipped = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x)) = -log(1 + exp(-x))."""
    return -np.logaddexp(0.0, -x)


class Word2VecSGNS:
    """Skip-Gram with Negative Sampling trained via manual analytical gradients."""

    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 4,
        neg_samples: int = 5,
        lr: float = 0.025,
        epochs: int = 10,
        batch_size: int = 256,
        subsample_t: float | None = 1e-3,
        combine_embeddings: bool = True,
        min_count: int = 2,
        lowercase: bool = True,
        normalize_numbers: bool = True,
        drop_punctuation: bool = True,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.subsample_t = subsample_t
        self.combine_embeddings = combine_embeddings
        self.seed = seed

        self.vocab = Vocabulary(
            min_count=min_count,
            lowercase=lowercase,
            normalize_numbers=normalize_numbers,
            drop_punctuation=drop_punctuation,
        )

        self.W_in: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.W_out: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.norm_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float64)

        self._noise_table: np.ndarray = np.empty(0, dtype=np.int32)
        self._keep_probs: np.ndarray = np.empty(0, dtype=np.float64)
        self.loss_history: list[float] = []

    def _build_noise_table(self, table_size: int = 1_000_000) -> np.ndarray:
        """Build unigram^(3/4) sampling table for fast O(1) negative sampling."""
        counts = np.asarray(self.vocab.counts, dtype=np.float64)
        powered = np.power(counts, 0.75)
        probs = powered / powered.sum()
        alloc = np.maximum(1, np.round(probs * table_size).astype(np.int64))
        table = np.repeat(np.arange(len(self.vocab), dtype=np.int32), alloc)
        return table

    def _build_keep_probs(self) -> np.ndarray:
        """Mikolov et al. (2013) subsampling probabilities for frequent words."""
        V = len(self.vocab)
        if self.subsample_t is None or self.subsample_t <= 0:
            return np.ones(V, dtype=np.float64)
        freqs = np.asarray(self.vocab.counts, dtype=np.float64) / max(1, self.vocab.kept_token_count)
        t = self.subsample_t
        keep = (np.sqrt(freqs / t) + 1.0) * (t / np.maximum(freqs, 1e-12))
        return np.minimum(1.0, keep)

    def _extract_pairs(
        self,
        sentences: list[list[str]],
        rng: np.random.Generator,
        apply_subsampling: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate (center_id, context_id) skip-gram pairs within sentence boundaries."""
        centers: list[int] = []
        contexts: list[int] = []
        W = self.window_size

        for sent in sentences:
            ids = self.vocab.encode_sentence(sent, drop_oov=True)
            if apply_subsampling and len(self._keep_probs) > 0 and len(ids) > 2:
                rand_u = rng.random(len(ids))
                ids = [w for w, u in zip(ids, rand_u) if u < self._keep_probs[w]]

            L = len(ids)
            if L < 2:
                continue

            # Dynamic window radius in [1, W] (gives higher effective weight to closer words)
            dyn_windows = rng.integers(1, W + 1, size=L) if apply_subsampling else np.full(L, W)
            for i in range(L):
                c = ids[i]
                rad = int(dyn_windows[i])
                left = max(0, i - rad)
                right = min(L, i + rad + 1)
                for j in range(left, right):
                    if i != j:
                        centers.append(c)
                        contexts.append(ids[j])

        return np.asarray(centers, dtype=np.int32), np.asarray(contexts, dtype=np.int32)

    def _forward_and_gradients(
        self,
        centers: np.ndarray,
        contexts: np.ndarray,
        negatives: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Manual SGNS forward pass, loss calculation, and analytical backward pass.

        Args:
            centers:   (B,) int32 center word indices
            contexts:  (B,) int32 positive context word indices
            negatives: (B, K) int32 negative word indices

        Returns:
            (mean_batch_loss, grad_Vc (B, d), grad_Uo (B, d), grad_Uneg (B, K, d))
        """
        # 1. Forward pass: embedding lookups and dot-product logits
        V_c = self.W_in[centers]            # (B, d)
        U_o = self.W_out[contexts]          # (B, d)
        U_neg = self.W_out[negatives]       # (B, K, d)

        s_pos = np.sum(V_c * U_o, axis=1)                        # (B,)
        s_neg = np.einsum("bkd,bd->bk", U_neg, V_c)              # (B, K)

        # 2. Negative sampling loss: -log(sigmoid(s_pos)) - sum_k log(sigmoid(-s_neg_k))
        loss_per_pair = -_log_sigmoid(s_pos) - np.sum(_log_sigmoid(-s_neg), axis=1)
        mean_loss = float(np.mean(loss_per_pair))

        # 3. Backward pass: analytical gradients of sum of batch losses
        # d/ds [-log(sigmoid(s))] = sigmoid(s) - 1
        # d/ds [-log(sigmoid(-s))] = sigmoid(s)
        g_pos = _sigmoid(s_pos) - 1.0       # (B,)
        g_neg = _sigmoid(s_neg)             # (B, K)

        grad_Uo = g_pos[:, None] * V_c                                           # (B, d)
        grad_Uneg = g_neg[:, :, None] * V_c[:, None, :]                          # (B, K, d)
        grad_Vc = g_pos[:, None] * U_o + np.einsum("bk,bkd->bd", g_neg, U_neg)   # (B, d)

        return mean_loss, grad_Vc, grad_Uo, grad_Uneg

    def _finalize_embeddings(self) -> None:
        if self.combine_embeddings:
            self.embeddings = self.W_in + self.W_out
        else:
            self.embeddings = self.W_in.copy()
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.norm_embeddings = self.embeddings / norms

    def fit(self, sentences: list[list[str]], verbose: bool = False) -> "Word2VecSGNS":
        rng = np.random.default_rng(self.seed)
        self.vocab.fit(sentences)
        V = len(self.vocab)
        d = self.embedding_dim

        # Small uniform initialization for W_in, zeros for W_out (standard SGNS init)
        self.W_in = rng.uniform(-0.5 / d, 0.5 / d, size=(V, d)).astype(np.float64)
        self.W_out = np.zeros((V, d), dtype=np.float64)

        self._noise_table = self._build_noise_table()
        self._keep_probs = self._build_keep_probs()
        self.loss_history = []

        table_len = len(self._noise_table)
        K = self.neg_samples
        B = self.batch_size

        # Estimate total steps for linear learning-rate schedule
        base_c, _ = self._extract_pairs(sentences, rng, apply_subsampling=True)
        approx_steps = max(1, (len(base_c) // B) * self.epochs)
        step = 0
        min_lr = self.lr * 0.0001

        for epoch in range(1, self.epochs + 1):
            centers, contexts = self._extract_pairs(sentences, rng, apply_subsampling=True)
            N_pairs = len(centers)
            if N_pairs == 0:
                break

            perm = rng.permutation(N_pairs)
            centers = centers[perm]
            contexts = contexts[perm]

            total_loss = 0.0
            total_seen = 0

            for start in range(0, N_pairs, B):
                end = min(start + B, N_pairs)
                b_c = centers[start:end]
                b_o = contexts[start:end]
                cur_b = len(b_c)

                neg_idx = rng.integers(0, table_len, size=(cur_b, K))
                b_neg = self._noise_table[neg_idx]

                batch_loss, grad_Vc, grad_Uo, grad_Uneg = self._forward_and_gradients(
                    b_c, b_o, b_neg
                )

                # Linear learning rate decay
                progress = min(1.0, step / approx_steps)
                cur_lr = max(min_lr, self.lr * (1.0 - progress))
                step += 1

                # Manual SGD parameter updates (np.add.at safely accumulates duplicate indices)
                np.add.at(self.W_in, b_c, -cur_lr * grad_Vc)
                np.add.at(self.W_out, b_o, -cur_lr * grad_Uo)
                np.add.at(self.W_out, b_neg.reshape(-1), -cur_lr * grad_Uneg.reshape(-1, d))

                total_loss += batch_loss * cur_b
                total_seen += cur_b

            epoch_loss = total_loss / max(1, total_seen)
            self.loss_history.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch:02d}/{self.epochs:02d} | loss = {epoch_loss:.4f}")

        self._finalize_embeddings()
        return self

    def validation_loss(self, val_sentences: list[list[str]], seed: int = 123) -> float:
        """Compute average SGNS loss on held-out validation sentences (no weight updates)."""
        rng = np.random.default_rng(seed)
        centers, contexts = self._extract_pairs(val_sentences, rng, apply_subsampling=False)
        N_pairs = len(centers)
        if N_pairs == 0 or len(self._noise_table) == 0:
            return float("inf")

        table_len = len(self._noise_table)
        K = self.neg_samples
        B = self.batch_size
        total_loss = 0.0

        for start in range(0, N_pairs, B):
            end = min(start + B, N_pairs)
            b_c = centers[start:end]
            b_o = contexts[start:end]
            cur_b = len(b_c)
            b_neg = self._noise_table[rng.integers(0, table_len, size=(cur_b, K))]
            batch_loss, _, _, _ = self._forward_and_gradients(b_c, b_o, b_neg)
            total_loss += batch_loss * cur_b

        return total_loss / N_pairs

    def check_gradients(self, eps: float = 1e-6) -> float:
        """Verify analytical gradients against finite differences on a toy batch."""
        rng = np.random.default_rng(0)
        V, d, B, K = 6, 4, 3, 2
        orig_in, orig_out = self.W_in, self.W_out
        self.W_in = rng.normal(0, 0.5, size=(V, d))
        self.W_out = rng.normal(0, 0.5, size=(V, d))

        centers = np.array([0, 1, 2], dtype=np.int32)
        contexts = np.array([3, 4, 5], dtype=np.int32)
        negatives = np.array([[1, 5], [0, 3], [4, 1]], dtype=np.int32)

        _, grad_Vc, grad_Uo, grad_Uneg = self._forward_and_gradients(centers, contexts, negatives)
        ana_in = np.zeros_like(self.W_in)
        ana_out = np.zeros_like(self.W_out)
        np.add.at(ana_in, centers, grad_Vc)
        np.add.at(ana_out, contexts, grad_Uo)
        np.add.at(ana_out, negatives.reshape(-1), grad_Uneg.reshape(-1, d))

        max_rel_err = 0.0
        for name, param, ana in [("W_in", self.W_in, ana_in), ("W_out", self.W_out, ana_out)]:
            for idx in np.ndindex(param.shape):
                old = param[idx]
                param[idx] = old + eps
                lp, _, _, _ = self._forward_and_gradients(centers, contexts, negatives)
                param[idx] = old - eps
                lm, _, _, _ = self._forward_and_gradients(centers, contexts, negatives)
                param[idx] = old
                # _forward_and_gradients returns mean over B; sum loss = B * mean_loss
                num_g = B * (lp - lm) / (2 * eps)
                denom = max(1e-8, abs(num_g) + abs(ana[idx]))
                max_rel_err = max(max_rel_err, abs(num_g - ana[idx]) / denom)

        self.W_in, self.W_out = orig_in, orig_out
        return max_rel_err

    def _resolve_id(self, word: str) -> int | None:
        norm = self.vocab.normalize(word)
        if norm is None:
            return None
        return self.vocab.word2idx.get(norm)

    def get_vector(self, word: str) -> np.ndarray | None:
        idx = self._resolve_id(word)
        return self.embeddings[idx] if idx is not None else None

    def similarity(self, w1: str, w2: str) -> float:
        i1 = self._resolve_id(w1)
        i2 = self._resolve_id(w2)
        if i1 is None or i2 is None:
            return 0.0
        return float(np.dot(self.norm_embeddings[i1], self.norm_embeddings[i2]))

    def most_similar(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        idx = self._resolve_id(word)
        if idx is None or len(self.norm_embeddings) == 0:
            return []
        sims = self.norm_embeddings @ self.norm_embeddings[idx]
        sims[idx] = -np.inf
        k = min(top_k, len(sims) - 1)
        top_ids = np.argpartition(sims, -k)[-k:]
        top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_ids]

    def most_similar_from_context(
        self, context_words: list[str], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Score words whose center embeddings align with surrounding context embeddings."""
        ctx_ids = [cid for cw in context_words if (cid := self._resolve_id(cw)) is not None]
        if not ctx_ids or len(self.norm_embeddings) == 0:
            return []
        ctx_vec = np.mean(self.W_out[ctx_ids], axis=0)
        norm = np.linalg.norm(ctx_vec)
        if norm < 1e-12:
            return []
        ctx_vec = ctx_vec / norm
        sims = self.norm_embeddings @ ctx_vec
        k = min(top_k, len(sims))
        top_ids = np.argpartition(sims, -k)[-k:]
        top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_ids]

    @property
    def shape(self) -> tuple[int, int]:
        return self.embeddings.shape

    @property
    def sparsity(self) -> float:
        return 0.0

    def storage_bytes(self) -> int:
        """Exact memory footprint in bytes of the final dense embedding matrix."""
        return int(self.embeddings.nbytes)