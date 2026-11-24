"""Prepare the source-derived matrices and an auditable comparison (explicit CLI)."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from project1.preprocessing import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild source data with the released relationship rule, validate, then publish.")
    parser.add_argument("--output-dir", default="processed_data", help="Destination relative to the project root (default: processed_data)")
    parser.add_argument("--audit-dir", default="outputs/data_audit", help="Audit destination relative to the project root")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing output and audit directories after validation")
    args = parser.parse_args()
    summary = prepare_data(ROOT, args.output_dir, args.audit_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
