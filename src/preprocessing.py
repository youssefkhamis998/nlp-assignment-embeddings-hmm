"""Dataset loading and splitting for CoNLL-2000 (tokens + POS tags only)."""

from __future__ import annotations

import random
from pathlib import Path

DATASET_NAME = "eriktks/conll2000"
# The original repo ships a loading script, which recent `datasets` versions refuse to run.
# The Hub's auto-converted parquet branch contains the exact same data.
DATASET_REVISION = "refs/convert/parquet"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "hf_cache"

Sentence = list[tuple[str, str]]  # [(token, pos_tag), ...]


def load_conll2000(cache_dir: str | Path = DEFAULT_CACHE) -> dict[str, list[Sentence]]:
    """Return {'train': [...], 'test': [...]} with each sentence as [(token, tag), ...].

    Chunk labels are discarded as required by the assignment.
    """
    from datasets import load_dataset  # only used to fetch raw data

    ds = load_dataset(DATASET_NAME, revision=DATASET_REVISION, cache_dir=str(cache_dir))
    tag_names = ds["train"].features["pos_tags"].feature.names
    out = {}
    for split in ("train", "test"):
        out[split] = [
            [(tok, tag_names[t]) for tok, t in zip(ex["tokens"], ex["pos_tags"])]
            for ex in ds[split]
        ]
    return out


def train_val_split(
    sentences: list[Sentence], val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[Sentence], list[Sentence]]:
    """Reproducible random sentence-level split of the training partition."""
    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(sentences) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [s for i, s in enumerate(sentences) if i not in val_idx]
    val = [s for i, s in enumerate(sentences) if i in val_idx]
    return train, val


def words_of(sentence: Sentence) -> list[str]:
    return [w for w, _ in sentence]


def tags_of(sentence: Sentence) -> list[str]:
    return [t for _, t in sentence]
