"""Package completed, verified artifacts; never generate missing experimental results."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from project1.analysis import load_predictions, load_reference, read_json, sha256
from project1.multiseed import SUMMARY_FILES, verify_summary

RUN_FILES = ("config.json", "metrics.json", "history.json", "history.csv", "run_manifest.json", "validation_predictions.npz")
AUDIT_FILES = ("summary.json", "record_audit.csv", "accepted_edges.csv", "components.csv", "verification.json")
COMPARISON_FILES = ("comparison.json", "comparison.csv", "selection.json", "per_protein.csv", "history.png", "per_class_f1.png", "confusion_matrices.png", "boundary_metrics.png")
CLI_FILES = ("prepare_data.py", "train.py", "predict.py", "evaluate.py", "analyze_results.py", "analyze_multiseed.py", "package_submission.py", "inspect_data.py", "starter/run_baseline.py")


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty required artifact: {path}")


def package(args):
    data, selection_path, pred_path, report_path = map(project_path, (args.data, args.selection, args.pred, args.report))
    selection = read_json(selection_path)
    multi_seed = selection.get("multi_seed_selection") is True
    multi_records = None
    if multi_seed:
        runs_root = project_path(args.runs_root)
        multi_records, multi_summary, verified_selection = verify_summary(data, runs_root, selection_path.parent)
        if selection != verified_selection:
            raise ValueError("Multi-seed selection differs from the recomputed fixed protocol")
        comparison_dir = runs_root / "comparison"
    else:
        comparison_dir = selection_path.parent
    comparison = read_json(comparison_dir / "comparison.json")
    if comparison.get("split") != "validation" or selection.get("selection_split") != "validation" or selection.get("selection_metric") != ("mean_q3" if multi_seed else "q3"):
        raise ValueError("Selection must use validation Q3")
    methods = comparison.get("methods", [])
    candidates = [method for method in methods if method.get("role") == "candidate"]
    if len(methods) != 3 or len(candidates) != 2 or sum(m.get("role") == "baseline" for m in methods) != 1 or len({m["model_name"] for m in methods}) != 3:
        raise ValueError("Comparison requires a baseline and two differently named candidates")
    if next(m for m in methods if m.get("role") == "baseline")["model_name"] != "convolutional" or any(m["model_name"] == "convolutional" for m in candidates):
        raise ValueError("The released convolutional model must remain the baseline")
    if multi_seed:
        selected = next(r for r in multi_records if r["seed"] == 42 and r["method"] == selection["selected_method"])
        # Detailed seed-42 figures are descriptive; their single-seed winner
        # cannot override the method chosen from the fixed five-seed means.
        for method in methods:
            corresponding = next(r for r in multi_records if r["seed"] == 42 and r["model_name"] == method["model_name"])
            if project_path(method["run_dir"]) != project_path(corresponding["run_dir"]) or method["validation_metrics"] != corresponding["validation_metrics"]:
                raise ValueError("Seed-42 detailed comparison differs from multi-seed evidence")
            if any(corresponding["evidence_sha256"].get(f) != digest for f, digest in method["evidence_sha256"].items()):
                raise ValueError("Seed-42 detailed comparison has stale run evidence")
    else:
        selected = sorted(candidates, key=lambda m: (-m["validation_metrics"]["q3"], -m["validation_metrics"]["macro_f1"], m["model_name"]))[0]
    selected_run = project_path(selection["selected_run_dir"])
    if selected_run != project_path(selected["run_dir"]) or selection["selected_model_name"] != selected["model_name"] or selection["validation_metrics"] != selected["validation_metrics"]:
        raise ValueError("Selection differs from validation comparison ranking")
    ids, lengths, mask = load_reference(data, "test")
    if mask.shape != (1102, 698):
        raise ValueError("Test predictions must have released shape 1102 x 698")
    released_ids, released_lengths, released_mask = load_reference(ROOT / "standard_data", "test")
    if not np.array_equal(ids, released_ids) or not np.array_equal(lengths, released_lengths) or not np.array_equal(mask, released_mask):
        raise ValueError("Processed test IDs/lengths differ from the released test order")
    load_predictions(pred_path, ids, mask)
    prediction_metadata = pred_path.with_suffix(".metadata.json")
    if multi_seed and not prediction_metadata.is_file():
        raise ValueError("Multi-seed submissions require prediction metadata from the selected seed-42 checkpoint")
    if prediction_metadata.exists():
        metadata = read_json(prediction_metadata)
        if multi_seed and (metadata.get("selection_split") != "validation" or metadata.get("split") != "test"):
            raise ValueError("Multi-seed test prediction metadata must record validation-only selection")
        if metadata.get("checkpoint_sha256") != sha256(selected_run / "model.pt"):
            raise ValueError("Prediction metadata does not identify the selected checkpoint")
        if metadata.get("model_name") != selection["selected_model_name"]:
            raise ValueError("Prediction metadata model name differs from selection")
        if metadata.get("prediction_sha256") != sha256(pred_path):
            raise ValueError("Prediction file differs from its saved inference metadata")
    validation_ids, _, validation_mask = load_reference(data, "validation")
    if set(comparison["reference_sha256"]) != {"ids.npy", "lengths.npy", "X.npy", "Y.npy"}:
        raise ValueError("Incomplete seed-42 validation reference evidence")
    for filename, expected in comparison["reference_sha256"].items():
        if filename not in ("ids.npy", "lengths.npy", "X.npy", "Y.npy") or sha256(data / "validation" / filename) != expected:
            raise ValueError("Validation reference changed after comparison")
    require_file(report_path)
    with report_path.open("rb") as handle:
        header = handle.read(8)
        handle.seek(max(0, report_path.stat().st_size - 2048))
        tail = handle.read()
    if report_path.stat().st_size < 1024 or not header.startswith(b"%PDF-") or b"%%EOF" not in tail:
        raise ValueError("report must be a completed PDF, not an empty or placeholder file")
    audit = project_path(args.audit_dir)
    verification = read_json(audit / "verification.json")
    if verification.get("all_passed") is not True:
        raise ValueError("Preprocessing verification is missing or has failed")
    entries = {}

    def add(source, destination):
        require_file(source)
        if destination in entries:
            raise ValueError(f"Duplicate archive path: {destination}")
        entries[destination] = source

    for filename in ("README.md", "REFERENCES.md", "ENVIRONMENT.md", "DATA_DICTIONARY.md", "READING_GUIDE.md", "requirements.txt", "requirements-torch-cpu.txt", *CLI_FILES):
        add(ROOT / filename, filename)
    add(report_path, "report.pdf")
    add(pred_path, "predictions.npz")
    if prediction_metadata.exists():
        add(prediction_metadata, "outputs/prediction_metadata.json")
    add(selected_run / "model.pt", "model/model.pt")
    add(selected_run / "config.json", "model/config.json")
    for folder, pattern in (("src", "*.py"), ("configs", "*.json")):
        files = sorted((ROOT / folder).rglob(pattern))
        if not files:
            raise ValueError(f"No source/config files found in {folder}")
        for path in files:
            if "__pycache__" not in path.parts and not path.is_symlink():
                add(path, path.relative_to(ROOT).as_posix())
    for filename in AUDIT_FILES:
        add(audit / filename, f"outputs/data_audit/{filename}")
    for filename in COMPARISON_FILES:
        if filename != "selection.json" or multi_seed:
            add(comparison_dir / filename, f"outputs/comparison/{filename}")
    if multi_seed:
        add(runs_root / "multiseed" / "run_plan.json", "outputs/multiseed/run_plan.json")
        for filename in SUMMARY_FILES:
            if filename != "selection.json":
                add(selection_path.parent / filename, f"outputs/multiseed_summary/{filename}")
        for filename in ("run_multiseed.py",):
            add(ROOT / filename, filename)
    archive_run_dirs = {}
    for index, method in enumerate(multi_records if multi_seed else methods):
        run = project_path(method["run_dir"])
        config = read_json(run / "config.json")
        if config["model"]["name"] != method["model_name"]:
            raise ValueError(f"Model name changed: {run}")
        run_manifest = read_json(run / "run_manifest.json")
        if run_manifest.get("status") != "complete" or run_manifest.get("test_used_for_selection") is not False:
            raise ValueError(f"Run is incomplete or used test data for selection: {run}")
        if run_manifest.get("model_name") != method["model_name"] or run_manifest.get("model_sha256") != sha256(run / "model.pt"):
            raise ValueError(f"Run manifest does not match the model evidence: {run}")
        load_predictions(run / "validation_predictions.npz", validation_ids, validation_mask)
        expected_evidence = {"config.json", "metrics.json", "history.json", "history.csv", "validation_predictions.npz", "model.pt"}
        if multi_seed:
            expected_evidence.add("run_manifest.json")
        if set(method.get("evidence_sha256", {})) != expected_evidence:
            raise ValueError(f"Incomplete comparison evidence for {run}")
        for filename, digest in method["evidence_sha256"].items():
            if filename not in (*RUN_FILES, "model.pt") or sha256(run / filename) != digest:
                raise ValueError(f"Run evidence changed since analysis: {run / filename}")
        if multi_seed:
            archive_dir = (f"outputs/runs/{method['method']}" if method["seed"] == 42
                           else f"outputs/multiseed/seed_{method['seed']}/{method['method']}")
        else:
            archive_dir = f"outputs/runs/{index}_{method['role']}"
        archive_run_dirs[str(run)] = archive_dir
        for filename in RUN_FILES:
            add(run / filename, f"{archive_dir}/{filename}")
    packaged_selection = {**selection, "selected_run_dir": archive_run_dirs[str(selected_run)], "selected_checkpoint": "model/model.pt"}
    archived_selection_path = "outputs/multiseed_summary/selection.json" if multi_seed else "outputs/comparison/selection.json"
    entries[archived_selection_path] = (json.dumps(packaged_selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    manifest = {"format_version": 1, "student_id": args.student_id, "name": args.name,
                "selected_model_name": selected["model_name"], "selection_metric": selection["selection_metric"], "validation_metrics": selected["validation_metrics"],
                "multi_seed_selection": multi_seed, "representative_seed": 42 if multi_seed else None,
                "files": {name: {"sha256": hashlib.sha256(path).hexdigest() if isinstance(path, bytes) else sha256(path), "bytes": len(path) if isinstance(path, bytes) else path.stat().st_size} for name, path in sorted(entries.items())}}
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.student_id.isdigit() or not args.name.strip() or any(c in args.name for c in '/\\:*?"<>|') or args.name in (".", ".."):
        raise ValueError("Invalid student ID or name for archive filename")
    archive = output / f"project1_{args.student_id}_{args.name}.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite an existing submission: {archive}")
    temporary = archive.with_suffix(".zip.partial")
    if temporary.exists():
        raise FileExistsError(f"Remove or inspect the existing incomplete archive first: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as handle:
            for name, path in sorted(entries.items()):
                if isinstance(path, bytes):
                    handle.writestr(name, path)
                else:
                    handle.write(path, name)
            handle.writestr("submission_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        with zipfile.ZipFile(temporary) as handle:
            if set(handle.namelist()) != set(entries) | {"submission_manifest.json"} or handle.testzip() is not None:
                raise ValueError("Archive member or CRC verification failed")
            for name, evidence in manifest["files"].items():
                content = handle.read(name)
                if len(content) != evidence["bytes"] or hashlib.sha256(content).hexdigest() != evidence["sha256"]:
                    raise ValueError(f"Archive readback hash mismatch: {name}")
        temporary.rename(archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    receipt = archive.with_suffix(".sha256")
    receipt.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"Verified submission: {archive}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="processed_data")
    parser.add_argument("--selection", default="outputs/comparison/selection.json")
    parser.add_argument("--pred", default="predictions.npz")
    parser.add_argument("--report", default="report.pdf")
    parser.add_argument("--audit-dir", default="outputs/data_audit")
    parser.add_argument("--runs-root", default="outputs", help="Root holding runs/, multiseed/, and seed-42 comparison/.")
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default="submissions")
    package(parser.parse_args())


if __name__ == "__main__":
    main()
