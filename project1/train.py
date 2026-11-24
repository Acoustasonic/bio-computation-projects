"""Train one explicitly selected configuration; nothing runs when imported."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project1.config import load_config
from project1.pipeline import BaselinePipeline


def resolve(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.json")
    parser.add_argument("--data-dir", type=Path, help="Override the validated matrix directory.")
    parser.add_argument("--output-dir", type=Path, help="Use a distinct directory for each run.")
    parser.add_argument("--device", help="Explicit device override: cpu, cuda[:index] or mps.")
    parser.add_argument("--overwrite", action="store_true", help="Replace only named artifacts in this run directory.")
    args = parser.parse_args()
    config = load_config(resolve(args.config), PROJECT_ROOT)
    if args.data_dir is not None:
        config = replace(config, data=replace(config.data, data_dir=resolve(args.data_dir)))
    config = replace(config, runtime=replace(
        config.runtime,
        device=args.device or config.runtime.device,
        output_dir=resolve(args.output_dir) if args.output_dir is not None else config.runtime.output_dir,
    ))
    metrics = BaselinePipeline(config).run(overwrite=args.overwrite)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
