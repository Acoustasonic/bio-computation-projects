"""Validation-only, mask-consistent comparison of three completed experiments."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import compute_q3_metrics

CLASSES = ("H", "E", "C")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(data: Path, split: str):
    """Read only the requested reference; analysis never accesses test labels."""
    directory = data / split
    ids = np.load(directory / "ids.npy", allow_pickle=False)
    lengths = np.load(directory / "lengths.npy", allow_pickle=False)
    inputs = np.load(directory / "X.npy", allow_pickle=False)
    if inputs.ndim != 2 or inputs.shape[1] != 698 or inputs.shape[0] == 0 or ids.ndim != 1 or ids.dtype.kind != "U":
        raise ValueError(f"{split}: invalid X or Unicode IDs")
    if len(set(ids.tolist())) != len(ids) or len(ids) != len(inputs):
        raise ValueError(f"{split}: duplicate IDs or inconsistent rows")
    if lengths.shape != (len(ids),) or lengths.dtype.kind not in "iu":
        raise ValueError(f"{split}: invalid lengths")
    if not np.all((lengths > 0) & (lengths <= inputs.shape[1])):
        raise ValueError(f"{split}: length out of range")
    mask = np.arange(inputs.shape[1])[None, :] < lengths[:, None]
    if inputs.dtype.kind not in "iu" or not np.all(inputs[~mask] == 0) or not np.all((inputs[mask] >= 1) & (inputs[mask] <= 21)):
        raise ValueError(f"{split}: invalid input encoding or padding")
    return ids, lengths, mask


def load_predictions(path: Path, ids, mask):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"ids", "q3"}:
            raise ValueError(f"{path}: expected exactly ids and q3 arrays")
        prediction_ids, predictions = archive["ids"], archive["q3"]
    if prediction_ids.dtype.kind != "U" or not np.array_equal(ids, prediction_ids):
        raise ValueError(f"{path}: IDs must match the released reference order")
    if predictions.shape != mask.shape or predictions.dtype.kind not in "iu":
        raise ValueError(f"{path}: invalid prediction shape or integer dtype")
    if not np.all(np.isin(predictions[mask], (0, 1, 2))) or not np.all(predictions[~mask] == -100):
        raise ValueError(f"{path}: invalid Q3 encoding or padding mask")
    return predictions


def summarize(labels, predictions, mask):
    count = int(mask.sum())
    if count == 0:
        return {"residue_count": 0, "q3": None, "macro_f1": None, "class_f1_H_E_C": [None] * 3}
    result = compute_q3_metrics(labels[mask], predictions[mask])
    return {"residue_count": count, **result}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boundary_mask(labels, lengths):
    """Both residues of each true label transition, expanded by two residues."""
    result = np.zeros(labels.shape, dtype=bool)
    for row, length in enumerate(lengths):
        transitions = np.flatnonzero(labels[row, 1:length] != labels[row, :length - 1])
        for left in transitions:
            result[row, max(0, left - 2):min(int(length), left + 4)] = True
    return result


def make_plots(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [record["model_name"] for record in records]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for record in records:
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        line, = axes[0].plot(epochs, [row["validation_loss"] for row in history], label=f"{record['model_name']} validation")
        axes[0].plot(epochs, [row["train_loss"] for row in history], "--", color=line.get_color(), label=f"{record['model_name']} train")
        axes[1].plot(epochs, [row["q3"] for row in history], label=record["model_name"])
    for axis, metric in zip(axes, ("Residue cross-entropy", "Validation Q3")):
        axis.set(xlabel="Epoch", ylabel=metric)
        axis.legend(fontsize=8)
    figure.savefig(output / "history.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    positions = np.arange(3)
    for index, record in enumerate(records):
        axis.bar(positions + (index - 1) * .25, record["validation_metrics"]["class_f1_H_E_C"], .25, label=record["model_name"])
    axis.set(xticks=positions, xticklabels=CLASSES, ylabel="Validation F1", ylim=(0, 1))
    axis.legend()
    figure.savefig(output / "per_class_f1.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, record in zip(axes, records):
        matrix = np.asarray(record["confusion_matrix_H_E_C"])
        axis.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            axis.text(col, row, str(value), ha="center", va="center", fontsize=8)
        axis.set(title=record["model_name"], xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="True")
    figure.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("q3", "macro_f1")):
        for index, group in enumerate(("boundary", "interior")):
            values = [record["regions"][group][metric] for record in records]
            axis.bar(np.arange(3) + (index - .5) * .35, [np.nan if v is None else v for v in values], .35, label=group)
        axis.set(xticks=range(3), xticklabels=names, ylabel=metric, ylim=(0, 1))
        axis.legend()
    figure.savefig(output / "boundary_metrics.png", dpi=180)
    plt.close(figure)


def analyze(data: Path, baseline: Path, candidates: list[Path], output: Path):
    if len(candidates) != 2 or len({p.resolve() for p in [baseline, *candidates]}) != 3:
        raise ValueError("Supply one baseline and exactly two distinct candidate run directories")
    ids, lengths, valid = load_reference(data, "validation")
    labels = np.load(data / "validation" / "Y.npy", allow_pickle=False)
    if labels.shape != valid.shape or labels.dtype.kind not in "iu" or not np.all(labels[~valid] == -100) or not np.all(np.isin(labels[valid], (0, 1, 2))):
        raise ValueError("Invalid validation labels or mask")
    boundary = boundary_mask(labels, lengths) & valid
    records, per_protein = [], []
    for role, directory in [("baseline", baseline), *[("candidate", p) for p in candidates]]:
        config = read_json(directory / "config.json")
        name = config["model"]["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{directory}: missing model.name")
        history = read_json(directory / "history.json")
        if not isinstance(history, list) or not history:
            raise ValueError(f"{directory}: empty history")
        saved = read_json(directory / "metrics.json")
        run_manifest = read_json(directory / "run_manifest.json")
        if run_manifest.get("status") != "complete" or run_manifest.get("test_used_for_selection") is not False:
            raise ValueError(f"{directory}: run is incomplete or did not use validation-only selection")
        predictions = load_predictions(directory / "validation_predictions.npz", ids, valid)
        metrics = summarize(labels, predictions, valid)
        if not np.isclose(saved["q3"], metrics["q3"]) or not np.allclose(saved["class_f1_H_E_C"], metrics["class_f1_H_E_C"]):
            raise ValueError(f"{directory}: saved metrics disagree with the validation predictions")
        matrix = np.bincount(labels[valid] * 3 + predictions[valid], minlength=9).reshape(3, 3)
        groups = {}
        for group, low, high in [("<=100", 1, 100), ("101-200", 101, 200), ("201-400", 201, 400), (">400", 401, valid.shape[1])]:
            rows = (lengths >= low) & (lengths <= high)
            groups[group] = {"protein_count": int(rows.sum()), **summarize(labels, predictions, valid & rows[:, None])}
        for row, protein_id in enumerate(ids):
            score = summarize(labels[row], predictions[row], valid[row])
            per_protein.append({"model_name": name, "role": role, "id": str(protein_id), "length": int(lengths[row]), "q3": score["q3"], "macro_f1": score["macro_f1"]})
        records.append({"role": role, "model_name": name, "run_dir": str(directory.resolve()), "validation_metrics": metrics,
                        "best_epoch": saved["best_epoch"], "parameter_count": saved["parameter_count"], "training_seconds": saved["training_seconds"],
                        "confusion_matrix_H_E_C": matrix.tolist(), "regions": {"boundary": summarize(labels, predictions, boundary), "interior": summarize(labels, predictions, valid & ~boundary)},
                        "length_groups": groups, "history": history,
                        "evidence_sha256": {filename: sha256(directory / filename) for filename in ("config.json", "metrics.json", "history.json", "history.csv", "validation_predictions.npz", "model.pt")}})
    if len({record["model_name"] for record in records}) != 3:
        raise ValueError("Baseline and both candidates must have different model.name values")
    if records[0]["model_name"] != "convolutional" or any(r["model_name"] == "convolutional" for r in records[1:]):
        raise ValueError("The course convolutional model must be the baseline, not a final candidate")
    selected = sorted(records[1:], key=lambda r: (-r["validation_metrics"]["q3"], -r["validation_metrics"]["macro_f1"], r["model_name"]))[0]
    for record in records:
        record["selected"] = record is selected
    output.mkdir(parents=True, exist_ok=True)
    comparison = {"split": "validation", "class_order": list(CLASSES), "boundary_definition": "For a transition between i and i+1 (zero based), union [i-2, i+3], clipped to sequence.", "reference_sha256": {f: sha256(data / "validation" / f) for f in ("ids.npy", "lengths.npy", "X.npy", "Y.npy")}, "methods": records}
    write_json(output / "comparison.json", comparison)
    write_json(output / "selection.json", {"selected_run_dir": selected["run_dir"], "selected_model_name": selected["model_name"], "selection_metric": "q3", "validation_metrics": selected["validation_metrics"], "tie_break": "descending macro_f1, then ascending model.name", "selection_split": "validation"})
    write_csv(output / "comparison.csv", [{"role": r["role"], "model_name": r["model_name"], "selected": r["selected"], "run_dir": r["run_dir"], "q3": r["validation_metrics"]["q3"], "macro_f1": r["validation_metrics"]["macro_f1"], **{f"f1_{c}": r["validation_metrics"]["class_f1_H_E_C"][i] for i, c in enumerate(CLASSES)}, "best_epoch": r["best_epoch"], "parameter_count": r["parameter_count"], "training_seconds": r["training_seconds"]} for r in records])
    write_csv(output / "per_protein.csv", per_protein)
    make_plots(output, records)
    return selected
