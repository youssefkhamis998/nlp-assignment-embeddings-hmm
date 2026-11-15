"""Log-space Viterbi decoding implemented with NumPy."""

from __future__ import annotations

import numpy as np


def viterbi(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_emit: np.ndarray,
    log_end: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Most likely state sequence for one observation sequence.

    Args:
        log_pi:   (N,)   log P(t_1 = j)
        log_A:    (N, N) log P(t_i = j | t_{i-1} = i)
        log_emit: (T, N) log P(w_t | tag j) for every position of the sentence
        log_end:  (N,)   optional log P(</s> | t_T = j)

    Working entirely with log-probabilities turns products into sums, so long
    sentences never underflow the way raw probability products would.

    Returns:
        (best_path as list of state indices, its log-probability)
    """
    T, N = log_emit.shape
    if T == 0:
        return [], 0.0

    delta = log_pi + log_emit[0]                      # (N,)
    backptr = np.zeros((T, N), dtype=np.int32)

    for t in range(1, T):
        # scores[i, j] = best score ending in i at t-1, then transition i -> j
        scores = delta[:, None] + log_A               # (N, N)
        backptr[t] = np.argmax(scores, axis=0)
        delta = scores[backptr[t], np.arange(N)] + log_emit[t]

    if log_end is not None:
        delta = delta + log_end

    last = int(np.argmax(delta))
    best_score = float(delta[last])

    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(int(backptr[t, path[-1]]))
    path.reverse()
    return path, best_score
