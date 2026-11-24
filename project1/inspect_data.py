"""Inspect the released project1 matrices without training a model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from project1.data import ProteinSequenceSplit, StandardDataModule  # noqa: E402


def summarize_split(split_name: str, split: ProteinSequenceSplit) -> None:
    """Print a short summary of a validated data split."""

    print(
        f"{split_name:10s} proteins={len(split.ids):5d} shape={split.inputs.shape} "
        f"length[min/median/max]={split.lengths.min()}/"
        f"{np.median(split.lengths):.1f}/{split.lengths.max()} "
        f"valid_residues={int(split.lengths.sum())}"
    )
    if split.labels is None:
        return
    counts = [int(np.sum(split.labels[split.valid_mask] == value)) for value in range(3)]
    total = sum(counts)
    label_summary = " ".join(
        f"{label}={count} ({count / total:.2%})"
        for label, count in zip("HEC", counts)
    )
    print(" " * 11 + "labels " + label_summary)


def main() -> None:
    data = StandardDataModule(PROJECT_ROOT / "standard_data")
    for split_name in ("train", "validation", "test"):
        summarize_split(split_name, data.load_split(split_name))
    print("Standard matrices passed the shape, encoding and padding checks.")


if __name__ == "__main__":
    main()
