"""Token-level POS tagging metrics implemented with NumPy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np


def per_tag_prf(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support for every tag seen in gold or pred."""
    tags = sorted(set(gold) | set(pred))
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    out = {}
    for t in tags:
        prec = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        rec = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[t] = {"precision": prec, "recall": rec, "f1": f1, "support": tp[t] + fn[t]}
    return out


def summarize(gold: list[str], pred: list[str]) -> dict[str, float]:
    """Accuracy plus macro- and support-weighted P/R/F1.

    For single-label token classification micro-P = micro-R = micro-F1 = accuracy,
    so macro (every tag equal) and weighted (by gold support) averages are reported.
    Macro averages are taken over tags with gold support > 0.
    """
    n = len(gold)
    if n == 0:
        return {"tokens": 0}
    per_tag = per_tag_prf(gold, pred)
    supported = {t: m for t, m in per_tag.items() if m["support"] > 0}
    sup = np.array([m["support"] for m in supported.values()], dtype=float)
    res = {"tokens": n, "accuracy": float(np.mean(np.array(gold) == np.array(pred)))}
    for key in ("precision", "recall", "f1"):
        vals = np.array([m[key] for m in supported.values()])
        res[f"macro_{key}"] = float(vals.mean())
        res[f"weighted_{key}"] = float((vals * sup).sum() / sup.sum())
    return res


def evaluate(
    gold_sents: list[list[str]],
    pred_sents: list[list[str]],
    word_sents: list[list[str]],
    is_oov: Callable[[str], bool],
) -> dict[str, dict[str, float]]:
    """Metrics on all tokens, in-vocabulary tokens and out-of-vocabulary tokens."""
    gold = [t for s in gold_sents for t in s]
    pred = [t for s in pred_sents for t in s]
    oov = [is_oov(w) for s in word_sents for w in s]
    subsets = {"all": [True] * len(gold), "known": [not o for o in oov], "unknown": oov}
    return {
        name: summarize([g for g, m in zip(gold, mask) if m], [p for p, m in zip(pred, mask) if m])
        for name, mask in subsets.items()
    }


def format_results(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Pretty table: rows = system/subset, columns = metrics."""
    cols = ["tokens", "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1"]
    short = ["tokens", "acc", "macroP", "macroR", "macroF1", "wP", "wR", "wF1"]
    lines = [f"{'system':<28}{'subset':<9}" + "".join(f"{c:>9}" for c in short)]
    lines.append("-" * len(lines[0]))
    for system, by_subset in results.items():
        for subset, m in by_subset.items():
            row = f"{system:<28}{subset:<9}{m['tokens']:>9}"
            row += "".join(f"{m[c]:>9.4f}" for c in cols[1:])
            lines.append(row)
    return "\n".join(lines)


def confusion_matrix(gold: list[str], pred: list[str], tags: list[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(tags)}
    cm = np.zeros((len(tags), len(tags)), dtype=int)
    for g, p in zip(gold, pred):
        cm[idx[g], idx[p]] += 1
    return cm


def top_confusions(gold: list[str], pred: list[str], k: int = 10) -> list[tuple[str, str, int]]:
    c = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [(g, p, n) for (g, p), n in c.most_common(k)]
