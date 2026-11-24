"""Complete a paired five-seed experiment without replacing existing runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from project1.config import load_config

SEEDS = (42, 43, 44, 45, 46)
METHODS = ("baseline", "dilated_cnn", "bilstm")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_run(run: Path, expected: dict, data: Path, device: str) -> bool:
    """Reuse only a complete run whose model, configuration and data agree."""
    import numpy as np

    if not run.exists():
        return False
    required = ("model.pt", "metrics.json", "config.json", "history.json", "history.csv",
                "run_manifest.json", "validation_predictions.npz")
    if not all((run / name).is_file() for name in required):
        raise ValueError(f"Incomplete existing run; inspect it before retrying: {run}")
    config = read_json(run / "config.json")
    config["data"]["data_dir"] = str(data)
    config["runtime"] = {"device": device, "output_dir": str(run)}
    if config != expected:
        raise ValueError(f"Existing run configuration differs from the protocol: {run}")
    manifest = read_json(run / "run_manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is not False
            or manifest.get("model_sha256") != digest(run / "model.pt")
            or manifest.get("validation_ids_sha256") != hashlib.sha256(
                np.load(data / "validation/ids.npy", allow_pickle=False).tobytes()).hexdigest()
            or manifest.get("environment", {}).get("device") != device):
        raise ValueError(f"Existing run has incomplete or inconsistent evidence: {run}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="processed_data")
    parser.add_argument("--runs-root", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true", help="Print the fixed plan; do not train.")
    args = parser.parse_args()
    data, outputs = resolve(args.data), resolve(args.runs_root)
    core_files = [ROOT / "train.py", *(ROOT / "src/project1" / name for name in
                  ("config.py", "data.py", "factories.py", "metrics.py", "models.py", "pipeline.py"))]
    plan = {"seeds": list(SEEDS), "methods": list(METHODS), "device": args.device,
            "selection": "candidate mean validation Q3; ties: mean macro-F1 then model name",
            "representative_seed": 42, "sample_sd_ddof": 1,
            "training_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in core_files},
            "runs": []}
    for seed in SEEDS:
        for method in METHODS:
            run = (outputs / "runs" / method if seed == 42
                   else outputs / "multiseed" / f"seed_{seed}" / method)
            config = asdict(load_config(ROOT / "configs" / f"{method}.json", ROOT))
            config["training"]["seed"] = seed
            config["data"]["data_dir"] = str(data)
            config["runtime"] = {"device": args.device, "output_dir": str(run)}
            config = json.loads(json.dumps(config, default=str))
            config_path = outputs / "multiseed/configs" / f"{method}_seed_{seed}.json"
            plan["runs"].append({"seed": seed, "method": method, "run_dir": str(run),
                                 "config_path": str(config_path), "config": config})
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    # This writes the protocol before the first new training run. No test labels are read.
    plan_path = outputs / "multiseed/run_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Existing multiseed plan differs; use a separate output root")
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in plan["runs"]:
        run, config_path = Path(item["run_dir"]), Path(item["config_path"])
        if completed_run(run, item["config"], data, args.device):
            print(f"REUSE seed={item['seed']} method={item['method']}", flush=True)
            continue
        if item["seed"] == 42:
            raise ValueError("Seed 42 must already exist; this protocol reuses the original runs")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(item["config"], indent=2) + "\n", encoding="utf-8")
        print(f"START seed={item['seed']} method={item['method']}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "train.py"), "--config", str(config_path),
                        "--data-dir", str(data), "--output-dir", str(run), "--device", args.device],
                       cwd=ROOT, check=True)
        if not completed_run(run, item["config"], data, args.device):
            raise RuntimeError(f"Training did not finish: {run}")
    print("ALL_15_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
