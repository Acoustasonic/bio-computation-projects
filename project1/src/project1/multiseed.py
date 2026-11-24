"""Fixed five-seed validation comparison, without selecting a lucky seed."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .analysis import load_predictions, load_reference, read_json, sha256, summarize, write_csv, write_json

SEEDS = (42, 43, 44, 45, 46)
METHODS = ("baseline", "dilated_cnn", "bilstm")
MODEL_NAMES = {"baseline": "convolutional", "dilated_cnn": "dilated_cnn", "bilstm": "bilstm"}
METRICS = ("q3", "macro_f1", "f1_H", "f1_E", "f1_C")
EVIDENCE_FILES = ("config.json", "metrics.json", "history.json", "history.csv", "validation_predictions.npz", "model.pt", "run_manifest.json")
SUMMARY_FILES = ("protocol.json", "per_seed.json", "per_seed.csv", "summary.json", "summary.csv", "paired_differences.csv", "comparison.png", "selection.json")
CORE_FILES = ("train.py", "src/project1/config.py", "src/project1/data.py", "src/project1/factories.py", "src/project1/metrics.py", "src/project1/models.py", "src/project1/pipeline.py")


def protocol():
    return {"seeds": list(SEEDS), "methods": list(METHODS), "split": "validation",
            "class_order": ["H", "E", "C"], "standard_deviation_ddof": 1,
            "error_bar": "sample standard deviation across five seeds; not a confidence interval",
            "selection_metric": "mean_q3", "tie_break": "descending mean macro_f1, then ascending model_name",
            "representative_seed": 42, "representative_rule": "fixed seed 42; no best-seed selection; no ensemble",
            "paired_difference": "candidate minus baseline with the same seed",
            "data_sha256_semantics": "Hashes of the current supplied train/validation reference arrays at analysis time; validation predictions are rescored against these arrays. These are not historical pre-training hashes recorded by each run.",
            "historical_seed42_data_provenance": "Previously audited data and reuse of the recorded common data path, supported by manifest split counts and validation ID hashes; no per-run historical training-array hash is available.",
            "run_plan_evidence_scope": "Fixed method/seed configurations and unchanged core training-source hashes; the plan does not contain pre-training data-array hashes."}


def run_path(outputs: Path, method: str, seed: int) -> Path:
    return outputs / "runs" / method if seed == 42 else outputs / "multiseed" / f"seed_{seed}" / method


def flat_metrics(metrics):
    return {"q3": metrics["q3"], "macro_f1": metrics["macro_f1"],
            **{f"f1_{label}": metrics["class_f1_H_E_C"][i] for i, label in enumerate(("H", "E", "C"))}}


def statistic(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("Each statistic requires exactly five finite seed values")
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)), "n": 5}


def normalized_config(config):
    comparable = copy.deepcopy(config)
    del comparable["training"]["seed"]
    del comparable["runtime"]["output_dir"]
    # Absolute host paths may change when artifacts are downloaded. The hashes
    # recorded below identify today's analysis reference, not past training data.
    comparable["data"]["data_dir"] = "<data identified by data_sha256>"
    return comparable


def config_digest(config):
    value = json.dumps(normalized_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collect(data: Path, outputs: Path):
    """Validate all 15 completed runs and recompute each score on one reference."""
    plan_path = outputs / "multiseed" / "run_plan.json"
    plan = read_json(plan_path)
    root = Path(__file__).resolve().parents[2]
    source = {name: sha256(root / name) for name in CORE_FILES}
    recorded_source = {name.replace("\\", "/"): digest for name, digest in plan["training_source_sha256"].items()}
    if recorded_source != source:
        raise ValueError("Training source hashes differ from the fixed run plan")
    if plan.get("seeds") != list(SEEDS) or plan.get("methods") != list(METHODS) or plan.get("representative_seed") != 42 or plan.get("sample_sd_ddof") != 1:
        raise ValueError("Run plan differs from the fixed five-seed protocol")
    if plan.get("selection") != "candidate mean validation Q3; ties: mean macro-F1 then model name":
        raise ValueError("Run plan has a different model-selection rule")
    planned = {(item["seed"], item["method"]): item for item in plan["runs"]}
    if len(plan["runs"]) != 15 or set(planned) != {(s, m) for s in SEEDS for m in METHODS}:
        raise ValueError("Run plan must contain exactly the 15 fixed method/seed combinations")
    ids, _, valid = load_reference(data, "validation")
    train_ids, _, train_valid = load_reference(data, "train")
    labels = np.load(data / "validation" / "Y.npy", allow_pickle=False)
    train_labels = np.load(data / "train" / "Y.npy", allow_pickle=False)
    for split, truth, mask in (("validation", labels, valid), ("train", train_labels, train_valid)):
        if truth.shape != mask.shape or truth.dtype.kind not in "iu" or not np.all(truth[~mask] == -100) or not np.all(np.isin(truth[mask], (0, 1, 2))):
            raise ValueError(f"Invalid {split} labels or mask")
    reference = {f"{split}/{filename}": sha256(data / split / filename)
                 for split in ("train", "validation") for filename in ("X.npy", "Y.npy", "ids.npy", "lengths.npy")}
    records, config_identity = [], {}
    recorded_data_dirs = set()
    for seed in SEEDS:
        for method in METHODS:
            directory = run_path(outputs, method, seed).resolve()
            for filename in EVIDENCE_FILES:
                if not (directory / filename).is_file() or (directory / filename).stat().st_size == 0:
                    raise ValueError(f"Missing completed-run evidence: {directory / filename}")
            config = read_json(directory / "config.json")
            intended = planned[seed, method]
            if normalized_config(config) != normalized_config(intended["config"]) or intended["config"]["training"]["seed"] != seed:
                raise ValueError(f"Effective run configuration differs from the plan: {directory}")
            if config["data"]["data_dir"] != intended["config"]["data"]["data_dir"]:
                raise ValueError(f"Recorded data source differs from the plan: {directory}")
            if str(config["runtime"]["output_dir"]).replace("\\", "/") != str(intended["run_dir"]).replace("\\", "/"):
                raise ValueError(f"Recorded output directory differs from the plan: {directory}")
            if config["runtime"]["device"] != plan["device"]:
                raise ValueError(f"Run device differs from the plan: {directory}")
            saved = read_json(directory / "metrics.json")
            manifest = read_json(directory / "run_manifest.json")
            history = read_json(directory / "history.json")
            if config["model"]["name"] != MODEL_NAMES[method] or config["training"]["seed"] != seed:
                raise ValueError(f"Wrong model or seed: {directory}")
            if config["training"].get("selection_metric") != "q3":
                raise ValueError(f"Checkpoint selection must use Q3: {directory}")
            comparable = config_digest(config)
            if method in config_identity and config_identity[method] != comparable:
                raise ValueError(f"Effective configurations differ beyond seed/output/data paths for {method}")
            config_identity[method] = comparable
            recorded_data_dirs.add(str(config["data"]["data_dir"]))
            if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is not False or manifest.get("model_name") != MODEL_NAMES[method]:
                raise ValueError(f"Incomplete or non-validation-only run: {directory}")
            evidence = {filename: sha256(directory / filename) for filename in EVIDENCE_FILES}
            if manifest.get("model_sha256") != evidence["model.pt"]:
                raise ValueError(f"Checkpoint differs from completed manifest: {directory}")
            if manifest.get("validation_ids_sha256") != hashlib.sha256(ids.tobytes()).hexdigest():
                raise ValueError(f"Validation IDs differ from training manifest: {directory}")
            if manifest.get("train_proteins") != len(train_ids) or manifest.get("validation_proteins") != len(ids) or manifest.get("validation_residues") != int(valid.sum()):
                raise ValueError(f"Split counts differ from run manifest: {directory}")
            if not isinstance(history, list) or not history or saved["best_epoch"] not in [row["epoch"] for row in history]:
                raise ValueError(f"Invalid epoch history: {directory}")
            predictions = load_predictions(directory / "validation_predictions.npz", ids, valid)
            metrics = summarize(labels, predictions, valid)
            for metric, value in flat_metrics(metrics).items():
                if not np.isclose(value, flat_metrics(saved)[metric], rtol=0, atol=1e-12):
                    raise ValueError(f"Saved {metric} disagrees with predictions: {directory}")
            best_row = next(row for row in history if row["epoch"] == saved["best_epoch"])
            if not np.isclose(best_row["q3"], metrics["q3"], rtol=0, atol=1e-12) or not np.isclose(max(row["q3"] for row in history), metrics["q3"], rtol=0, atol=1e-12):
                raise ValueError(f"Saved checkpoint is inconsistent with best validation epoch: {directory}")
            records.append({"seed": seed, "method": method, "model_name": MODEL_NAMES[method],
                            "role": "baseline" if method == "baseline" else "candidate", "run_dir": str(directory),
                            "validation_metrics": metrics, "best_epoch": saved["best_epoch"], "parameter_count": saved["parameter_count"],
                            "training_seconds": saved["training_seconds"], "comparable_config_sha256": comparable,
                            "recorded_data_dir": str(config["data"]["data_dir"]), "data_sha256": reference,
                            "run_plan_sha256": sha256(plan_path), "training_source_sha256": source,
                            "evidence_sha256": evidence})
    # All runs must declare one common data source. Relocating that common path
    # is allowed; silently mixing differently named training sources is not.
    if len(recorded_data_dirs) != 1:
        raise ValueError("The 15 runs do not declare a common data directory")
    return records, reference


def aggregate(records, reference):
    if len(records) != 15 or {(r["seed"], r["method"]) for r in records} != {(s, m) for s in SEEDS for m in METHODS}:
        raise ValueError("Exactly the fixed 15 method/seed combinations are required")
    by_key = {(r["seed"], r["method"]): r for r in records}
    summaries, differences = [], []
    for method in METHODS:
        values = [flat_metrics(by_key[seed, method]["validation_metrics"]) for seed in SEEDS]
        delta = [{key: values[index][key] - flat_metrics(by_key[seed, "baseline"]["validation_metrics"])[key] for key in METRICS}
                 for index, seed in enumerate(SEEDS)]
        for seed, row in zip(SEEDS, delta):
            differences.append({"seed": seed, "method": method, "model_name": MODEL_NAMES[method], **row})
        summaries.append({"method": method, "model_name": MODEL_NAMES[method], "seed_count": 5,
                          "metrics": {key: statistic([row[key] for row in values]) for key in METRICS},
                          "paired_baseline_differences": {key: statistic([row[key] for row in delta]) for key in METRICS}})
    selected = sorted(summaries[1:], key=lambda r: (-r["metrics"]["q3"]["mean"], -r["metrics"]["macro_f1"]["mean"], r["model_name"]))[0]
    representative = by_key[42, selected["method"]]
    summary = {"format_version": 1, "protocol": protocol(), "reference_sha256": reference, "methods": summaries,
               "selected_method": selected["method"], "selected_model_name": selected["model_name"]}
    if "run_plan_sha256" in records[0]:
        summary["run_plan_sha256"] = records[0]["run_plan_sha256"]
        summary["training_source_sha256"] = records[0]["training_source_sha256"]
    selection = {"selected_run_dir": representative["run_dir"], "selected_model_name": selected["model_name"],
                 "selected_method": selected["method"], "selection_metric": "mean_q3", "selection_score": selected["metrics"]["q3"]["mean"],
                 "selection_split": "validation", "multi_seed_selection": True, "representative_seed": 42,
                 "validation_metrics": representative["validation_metrics"], "method_summary": selected,
                 "selected_checkpoint_sha256": representative["evidence_sha256"]["model.pt"], "tie_break": protocol()["tie_break"]}
    return summary, selection, differences


def tables(records, summary, differences):
    return {
        "per_seed.csv": [{"seed": r["seed"], "method": r["method"], "model_name": r["model_name"], "run_dir": r["run_dir"], **flat_metrics(r["validation_metrics"]), "best_epoch": r["best_epoch"], "parameter_count": r["parameter_count"], "training_seconds": r["training_seconds"], "model_sha256": r["evidence_sha256"]["model.pt"], "config_sha256": r["evidence_sha256"]["config.json"]} for r in records],
        "summary.csv": [{"method": r["method"], "model_name": r["model_name"], "seed_count": 5, **{f"{key}_{stat}": r["metrics"][key][stat] for key in METRICS for stat in ("mean", "sample_sd")}, **{f"delta_baseline_{key}_{stat}": r["paired_baseline_differences"][key][stat] for key in METRICS for stat in ("mean", "sample_sd")}} for r in summary["methods"]],
        "paired_differences.csv": differences,
    }


def verify_summary(data: Path, outputs: Path, directory: Path):
    """Recompute all evidence and statistics; do not trust cached summary numbers."""
    records, reference = collect(data, outputs)
    summary, selection, differences = aggregate(records, reference)
    expected = {"protocol.json": protocol(), "per_seed.json": {"runs": records}, "summary.json": summary, "selection.json": selection}
    for filename, value in expected.items():
        if read_json(directory / filename) != value:
            raise ValueError(f"Multi-seed summary is stale or inconsistent: {filename}")
    for filename, rows in tables(records, summary, differences).items():
        with (directory / filename).open(encoding="utf-8", newline="") as handle:
            observed = list(csv.DictReader(handle))
        if observed != [{key: str(value) for key, value in row.items()} for row in rows]:
            raise ValueError(f"Multi-seed CSV differs from recomputed values: {filename}")
    return records, summary, selection


def analyze(data: Path, outputs: Path, directory: Path):
    records, reference = collect(data, outputs)
    summary, selection, differences = aggregate(records, reference)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", protocol())
    write_json(directory / "per_seed.json", {"runs": records})
    write_json(directory / "summary.json", summary)
    write_json(directory / "selection.json", selection)
    for filename, rows in tables(records, summary, differences).items():
        write_csv(directory / filename, rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("q3", "macro_f1")):
        means = [r["metrics"][metric]["mean"] for r in summary["methods"]]
        sd = [r["metrics"][metric]["sample_sd"] for r in summary["methods"]]
        axis.bar(range(3), means, yerr=sd, capsize=5, color=("#777777", "#2878B5", "#D97930"))
        axis.set(xticks=range(3), xticklabels=[r["model_name"] for r in summary["methods"]], ylabel=f"Validation {metric}", ylim=(0, 1))
    figure.suptitle("Five seeds: mean ± sample SD (ddof=1), not a confidence interval")
    figure.savefig(directory / "comparison.png", dpi=200)
    plt.close(figure)
    return selection
