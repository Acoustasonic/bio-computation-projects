"""Verify all fixed five-seed runs and select a method by mean validation Q3."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from project1.multiseed import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "processed_data")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/multiseed_summary")
    args = parser.parse_args()
    selected = analyze(args.data, args.runs_root, args.output_dir)
    print(f"Selected {selected['selected_model_name']} by five-seed mean Q3; checkpoint seed=42")


if __name__ == "__main__":
    main()
