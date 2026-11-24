"""Compare the completed baseline and two candidates on validation only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from project1.analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    args = parser.parse_args()
    selected = analyze(args.data, args.baseline, args.candidate, args.output_dir)
    print(f"Selected {selected['model_name']} using validation Q3")


if __name__ == "__main__":
    main()
