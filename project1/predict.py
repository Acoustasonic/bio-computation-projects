"""Reload an existing selected checkpoint and write course-format predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project1.data import StandardDataModule, save_predictions
from project1.pipeline import choose_device, infer_model, load_trained_model, write_json


def resolve(value: Path | str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection", type=Path, help="Validation-only comparison selection.json.")
    group.add_argument("--checkpoint", type=Path, help="An already fitted model.pt.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "processed_data")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "predictions.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    selection = None
    if args.selection is not None:
        selection = json.loads(resolve(args.selection).read_text(encoding="utf-8"))
        if selection.get("selection_split") != "validation":
            raise ValueError("Final selection must be based on validation, not test.")
        checkpoint = (resolve(selection["selected_checkpoint"]) if selection.get("selected_checkpoint")
                      else resolve(selection["selected_run_dir"]) / "model.pt")
    else:
        checkpoint = resolve(args.checkpoint)
    output = resolve(args.output)
    metadata_path = output.with_suffix(".metadata.json")
    if output.suffix != ".npz":
        raise ValueError("Prediction output must end in .npz.")
    if (output.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError("Prediction output exists; choose a new path or use --overwrite.")
    model, payload = load_trained_model(checkpoint, args.device)
    model_name = payload["model_config"]["name"]
    if selection is not None and model_name != selection["selected_model_name"]:
        raise ValueError("Checkpoint model does not match the validation-selected candidate.")
    split = StandardDataModule(resolve(args.data)).load_split(args.split)
    # Inference never reads labels, even when the requested split is validation.
    from project1.data import ProteinSequenceSplit
    unlabeled = ProteinSequenceSplit(split.inputs, split.ids, split.lengths)
    q3, _ = infer_model(model, unlabeled, device=choose_device(args.device), batch_size=args.batch_size)
    save_predictions(output, split, q3)
    metadata = {"model_name": model_name, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "device": args.device, "split": args.split, "proteins": len(split.ids),
                "selection_split": selection.get("selection_split") if selection else None,
                "command": sys.argv, "python": platform.python_version(),
                "prediction_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    write_json(metadata_path, metadata)
    print(json.dumps({"prediction": str(output), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
