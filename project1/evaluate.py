"""Validate a project1 prediction file and compute metrics when labels exist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from project1.data import StandardDataModule, validate_predictions  # noqa: E402
from project1.metrics import compute_q3_metrics  # noqa: E402


def evaluate(data_dir: Path, prediction_file: Path, split_name: str) -> dict[str, object]:
    """Check identifier alignment, output encoding, and optional metrics."""

    data = StandardDataModule(data_dir)
    split = data.load_split(split_name)
    with np.load(prediction_file, allow_pickle=False) as prediction:
        if set(prediction.files) != {"ids", "q3"}:
            raise ValueError("The prediction file must contain exactly the arrays 'ids' and 'q3'.")
        received_ids = prediction["ids"]
        predicted_q3 = prediction["q3"]

    validate_predictions(split, received_ids, predicted_q3)

    result: dict[str, object] = {
        "split": split_name,
        "proteins": int(len(split.ids)),
        "valid_residues": int(split.valid_mask.sum()),
    }
    if split.labels is None:
        result["status"] = "FORMAT_VALID"
    else:
        result.update(
            compute_q3_metrics(
                split.labels,
                predicted_q3,
                ignore_index=data.label_padding_value,
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a project1 prediction file.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "standard_data",
        help="Directory containing the released standard_data splits.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        required=True,
        help="Data split used to check prediction IDs and array shapes.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Path to the predictions.npz file to validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.data, args.pred, args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
