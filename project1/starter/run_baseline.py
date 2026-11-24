"""Command-line entry point for the project1 convolutional baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from project1.config import load_config  # noqa: E402
from project1.pipeline import BaselinePipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the course baseline configuration."""

    parser = argparse.ArgumentParser(description="Run the project1 convolutional baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.json",
        help="Path to a baseline JSON configuration.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Override the device in the configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the directory used for predictions and checkpoints.",
    )
    parser.add_argument(
        "--predict-test",
        action="store_true",
        help="Also write baseline_test.npz after selecting the validation checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    runtime = replace(
        config.runtime,
        device=args.device or config.runtime.device,
        output_dir=(
            args.output_dir.resolve()
            if args.output_dir is not None
            else config.runtime.output_dir
        ),
    )
    pipeline = BaselinePipeline(replace(config, runtime=runtime))
    metrics = pipeline.run(predict_test=args.predict_test)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
