"""Rebuild course matrices from source records and the released relationship rule.

This module has no import-time data processing. Call ``prepare_data`` explicitly.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("train", "validation", "test")
PRIORITY = {"train": 0, "validation": 1, "test": 2}
MAX_LENGTH = 698
AA_CODES = {aa: index + 1 for index, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_METADATA = {"PAD": 0, "UNK": 21, "amino_acids": AA_CODES}
LABEL_METADATA = {"H": 0, "E": 1, "C": 2, "ignore": -100}
SOURCE_TO_COURSE = {0: 2, 1: 0, 2: 1}
ASCII_WHITESPACE = b" \t\n\r\v\f"


@dataclass
class Record:
    source_id: str
    split: str
    source_row: int
    sequence: str
    labels: tuple[int, ...] | None
    original_length: int
    raw_character_counts: dict[str, int]


def parse_sequence(value: str) -> tuple[str, int, dict[str, int]]:
    """Require exactly one bytes object in a literal Python list."""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ValueError("sequence is not a valid literal bytes-list") from error
    if type(parsed) is not list or len(parsed) != 1 or type(parsed[0]) is not bytes:
        raise ValueError("sequence must have the exact form [b'...']")
    raw = parsed[0]
    cleaned = raw.translate(None, ASCII_WHITESPACE).upper()
    try:
        sequence = cleaned.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("sequence contains non-ASCII bytes") from error
    if not sequence:
        raise ValueError("empty sequence")
    return sequence, len(raw), dict(sorted(Counter(chr(x) for x in raw).items()))


def parse_ss3(value: str, length: int) -> tuple[int, ...]:
    """Parse whitespace-separated source Q3, never Python comma-list syntax."""
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("ss3 must be enclosed in brackets")
    tokens = text[1:-1].split()
    if any(re.fullmatch(r"[012]", token) is None for token in tokens):
        raise ValueError("ss3 must contain only whitespace-separated 0, 1, 2")
    if len(tokens) != length:
        raise ValueError(f"ss3 length {len(tokens)} differs from sequence length {length}")
    return tuple(SOURCE_TO_COURSE[int(token)] for token in tokens)


def read_records(source_dir: Path) -> list[Record]:
    records: list[Record] = []
    seen: set[str] = set()
    for split in SPLITS:
        path = source_dir / f"{split}_raw.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"source_record_id", "sequence"} | ({"ss3"} if split != "test" else set())
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path}: missing required CSV columns")
            if split == "test" and "ss3" in reader.fieldnames:
                raise ValueError(f"{path}: unexpected test label column")
            for row_number, row in enumerate(reader, 1):
                try:
                    if None in row or any(row.get(key) is None for key in required):
                        raise ValueError("malformed CSV record")
                    source_id = row["source_record_id"]
                    if not source_id or source_id != source_id.strip() or source_id in seen:
                        raise ValueError(f"empty, whitespace-padded, or duplicate source ID: {source_id!r}")
                    sequence, original_length, raw_counts = parse_sequence(row["sequence"])
                    labels = None if split == "test" else parse_ss3(row["ss3"], len(sequence))
                    records.append(Record(source_id, split, row_number, sequence, labels, original_length, raw_counts))
                    seen.add(source_id)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}: record {row_number}: {error}") from error
    return records


class UnionFind:
    def __init__(self, keys: list[str]):
        self.parent = dict.fromkeys(keys)
        for key in keys:
            self.parent[key] = key
        self.size = dict.fromkeys(keys, 1)

    def find(self, key: str) -> str:
        parent = self.parent
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_components(records: list[Record], hits: Path, audit_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    by_id = {record.source_id: record for record in records}
    union = UnionFind(list(by_id))
    counts = Counter(total_hits=0, self_hits=0, candidate_hits=0, accepted_hits=0)
    fields = ["query", "target", "evalue", "identity", "aligned_pairs", "shorter_coverage", "alignment_columns"]
    with gzip.open(hits, "rt", encoding="utf-8-sig", newline="") as stream, (audit_dir / "accepted_edges.csv").open("w", encoding="utf-8", newline="") as output:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"query", "target", "query_length", "target_length", "query_alignment", "target_alignment", "evalue"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{hits}: missing alignment columns")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for line, row in enumerate(reader, 2):
            try:
                counts["total_hits"] += 1
                query, target = row["query"], row["target"]
                if not query or not target:
                    raise ValueError("alignment source IDs must be nonempty")
                if query == target:
                    counts["self_hits"] += 1
                    continue
                if query not in by_id or target not in by_id:
                    raise ValueError("alignment refers to an unknown source ID")
                qalign, talign = row["query_alignment"], row["target_alignment"]
                if not qalign or len(qalign) != len(talign):
                    raise ValueError("aligned strings must be nonempty and equal length")
                qlength, tlength = int(row["query_length"]), int(row["target_length"])
                if qlength != len(by_id[query].sequence) or tlength != len(by_id[target].sequence):
                    raise ValueError("alignment full lengths disagree with normalized source lengths")
                evalue = float(row["evalue"])
                if not math.isfinite(evalue) or evalue < 0:
                    raise ValueError("invalid evalue")
                counts["candidate_hits"] += 1
                aligned_pairs = sum(a != "-" and b != "-" for a, b in zip(qalign, talign))
                identical = sum(a == b and a != "-" for a, b in zip(qalign, talign))
                identity = identical / len(qalign)
                coverage = aligned_pairs / min(qlength, tlength)
                accepted = evalue <= 1e-3 and ((identity >= 0.25 and coverage >= 0.8) or (identity >= 0.8 and aligned_pairs >= 50))
                if accepted:
                    counts["accepted_hits"] += 1
                    union.union(query, target)
                    writer.writerow(dict(zip(fields, (query, target, evalue, identity, aligned_pairs, coverage, len(qalign)))))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{hits}: line {line}: {error}") from error
    members: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        members[union.find(record.source_id)].append(record)
    component_ids: dict[str, str] = {}
    winning_splits: dict[str, str] = {}
    component_rows = []
    for number, group in enumerate(members.values(), 1):
        component_id = f"component_{number:06d}"
        winner = max((record.split for record in group), key=PRIORITY.__getitem__)
        split_counts = Counter(record.split for record in group)
        for record in group:
            component_ids[record.source_id] = component_id
            winning_splits[record.source_id] = winner
        component_rows.append({"component_id": component_id, "size": len(group), "split_counts": dict(split_counts), "retained_split": winner, "source_ids": [record.source_id for record in group]})
    _write_json(audit_dir / "components.json", component_rows)
    with (audit_dir / "components.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["component_id", "size", "train", "validation", "test", "retained_split", "source_ids"])
        writer.writeheader()
        for row in component_rows:
            writer.writerow({"component_id": row["component_id"], "size": row["size"], **{split: row["split_counts"].get(split, 0) for split in SPLITS}, "retained_split": row["retained_split"], "source_ids": json.dumps(row["source_ids"])})
    counts["components"] = len(members)
    counts["cross_split_components"] = sum(len(row["split_counts"]) > 1 for row in component_rows)
    counts["component_size_histogram"] = dict(sorted(Counter(len(group) for group in members.values()).items()))
    return component_ids, winning_splits, dict(counts)


def _compare_array(actual: np.ndarray, path: Path) -> dict[str, Any]:
    expected = np.load(path, allow_pickle=False)
    same_shape = actual.shape == expected.shape
    result = {"shape": list(actual.shape), "standard_shape": list(expected.shape), "dtype": str(actual.dtype), "standard_dtype": str(expected.dtype), "shape_equal": same_shape, "values_equal": bool(same_shape and np.array_equal(actual, expected))}
    if same_shape:
        result["mismatched_values"] = int(np.count_nonzero(actual != expected))
    return result


def _guard_paths(source: Path, standard: Path, output: Path, audit: Path, overwrite: bool) -> None:
    def within(path: Path, parent: Path) -> bool:
        return path == parent or parent in path.parents
    if within(output, audit) or within(audit, output):
        raise ValueError("output and audit directories must be separate, non-nested paths")
    for target in (output, audit):
        root = source.parent
        if not target.is_relative_to(root):
            raise ValueError("Generated data and audit directories must stay inside the project root")
        for input_dir in (source, standard, root / "src", root / "configs", root / "starter"):
            if within(target, input_dir) or within(input_dir, target):
                raise ValueError(f"output directory overlaps protected input directory: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists; explicitly pass --overwrite")
        if target.exists() and not target.is_dir():
            raise ValueError(f"output is not a directory: {target}")
        if target.exists() and overwrite:
            marker = target / ("metadata/manifest.json" if target == output else "summary.json")
            if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("status") != "validated":
                raise ValueError(f"Refusing to replace a directory not produced by this preprocessing pipeline: {target}")


def _publish(staged: list[tuple[Path, Path]]) -> None:
    """Publish validated directories, restoring existing results on failure."""
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for temporary, target in staged:
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
                os.replace(target, backup)
                backups.append((backup, target))
            os.replace(temporary, target)
            installed.append(target)
    except BaseException:
        for target in reversed(installed):
            shutil.rmtree(target)
        for backup, target in reversed(backups):
            os.replace(backup, target)
        raise
    for backup, _ in backups:
        shutil.rmtree(backup)


def prepare_data(project_root: str | Path, output_dir: str | Path = "processed_data", audit_dir: str | Path = "outputs/data_audit", *, overwrite: bool = False) -> dict[str, Any]:
    """Build, validate against standards, then publish outputs; never infer exclusions.

    Relative destinations resolve against project_root. Returns the published audit
    summary. Existing destinations require explicit overwrite authorization.
    """
    root = Path(project_root).resolve()
    source, standard = root / "source_data", root / "standard_data"
    output = (root / Path(output_dir)).resolve()
    audit = (root / Path(audit_dir)).resolve()
    source, standard = source.resolve(), standard.resolve()
    _guard_paths(source, standard, output, audit, overwrite)
    for destination in (output, audit):
        destination.parent.mkdir(parents=True, exist_ok=True)
    stage_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    stage_audit = Path(tempfile.mkdtemp(prefix=f".{audit.name}.staging-", dir=audit.parent))
    try:
        records = read_records(source)
        components, winners, graph_summary = build_components(records, source / "candidate_similarity_hits.tsv.gz", stage_audit)
        duplicate_groups: dict[str, list[str]] = defaultdict(list)
        for record in records:
            duplicate_groups[record.sequence].append(record.source_id)
        duplicates = [{"group": f"duplicate_{index:06d}", "source_ids": ids} for index, ids in enumerate((ids for ids in duplicate_groups.values() if len(ids) > 1), 1)]
        _write_json(stage_audit / "exact_duplicate_groups.json", duplicates)
        duplicate_ids = {source_id: group["group"] for group in duplicates for source_id in group["source_ids"]}
        retained = {split: [record for record in records if record.split == split and winners[record.source_id] == split] for split in SPLITS}
        # The retained membership above is final before reading any standard arrays.
        metadata_dir = stage_output / "metadata"
        metadata_dir.mkdir()
        for filename, expected in (("amino_acids.json", AA_METADATA), ("labels.json", LABEL_METADATA)):
            observed = json.loads((standard / "metadata" / filename).read_text(encoding="utf-8-sig"))
            if observed != expected:
                raise ValueError(f"standard metadata {filename} differs from the specified encoding")
            _write_json(metadata_dir / filename, expected)
        course_ids: dict[str, str] = {}
        comparisons: dict[str, Any] = {}
        all_course_ids: set[str] = set()
        for split in SPLITS:
            group = retained[split]
            if any(len(record.sequence) > MAX_LENGTH for record in group):
                raise ValueError(f"{split}: a retained sequence exceeds {MAX_LENGTH}; truncation is forbidden")
            ids = np.load(standard / split / "ids.npy", allow_pickle=False)
            if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != len(group):
                raise ValueError(f"{split}: source-rule retained count {len(group)} cannot be paired with standard course IDs (shape {ids.shape})")
            if any(not str(value) for value in ids) or len(set(ids.tolist())) != len(ids):
                raise ValueError(f"{split}: empty or duplicate course IDs")
            all_course_ids.update(ids.tolist())
            lengths = np.asarray([len(record.sequence) for record in group], dtype=np.int64)
            mask = np.arange(MAX_LENGTH)[None, :] < lengths[:, None]
            x = np.zeros((len(group), MAX_LENGTH), dtype=np.uint8)
            y = None if split == "test" else np.full(x.shape, -100, dtype=np.int16)
            for index, record in enumerate(group):
                length = len(record.sequence)
                x[index, :length] = [AA_CODES.get(aa, 21) for aa in record.sequence]
                if y is not None:
                    y[index, :length] = record.labels
                course_ids[record.source_id] = str(ids[index])
            arrays = {"X": x, "ids": ids, "lengths": lengths, "valid_mask": mask}
            if y is not None:
                arrays["Y"] = y
            destination = stage_output / split
            destination.mkdir()
            comparisons[split] = {}
            for name, array in arrays.items():
                np.save(destination / f"{name}.npy", array, allow_pickle=False)
                if name != "valid_mask":
                    comparisons[split][name] = _compare_array(array, standard / split / f"{name}.npy")
            for name in ("X", "Y"):
                if name in comparisons[split] and comparisons[split][name]["dtype"] != comparisons[split][name]["standard_dtype"]:
                    raise ValueError(f"{split}/{name}: standard matrix dtype differs from required dtype")
        _write_json(stage_audit / "verification.json", {"all_passed": all(result["values_equal"] for split in comparisons.values() for result in split.values()), "splits": comparisons})
        if not all(result["values_equal"] for split in comparisons.values() for result in split.values()):
            raise ValueError("reconstructed arrays differ from standard data; results were not published")
        fields = ["split", "source_row", "source_record_id", "course_id", "original_bytes_length", "normalized_length", "raw_character_counts", "normalized_character_counts", "course_label_counts", "source_label_counts", "component_id", "exact_duplicate_group", "retained", "exclusion_reason"]
        with (stage_audit / "record_audit.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                keep = record.split == winners[record.source_id]
                label_counts = Counter(record.labels) if record.labels is not None else None
                writer.writerow({"split": record.split, "source_row": record.source_row, "source_record_id": record.source_id, "course_id": course_ids.get(record.source_id, ""), "original_bytes_length": record.original_length, "normalized_length": len(record.sequence), "raw_character_counts": json.dumps(record.raw_character_counts, ensure_ascii=True, sort_keys=True), "normalized_character_counts": json.dumps(dict(sorted(Counter(record.sequence).items()))), "course_label_counts": json.dumps(dict(label_counts)) if label_counts is not None else "", "source_label_counts": json.dumps({str(key): label_counts[value] for key, value in SOURCE_TO_COURSE.items()}) if label_counts is not None else "", "component_id": components[record.source_id], "exact_duplicate_group": duplicate_ids.get(record.source_id, ""), "retained": keep, "exclusion_reason": "" if keep else f"component_contains_higher_priority_split:{winners[record.source_id]}"})
        summary = {"status": "validated", "max_length": MAX_LENGTH, "rule": {"evalue_max": 0.001, "identity_min": 0.25, "shorter_coverage_min": 0.8, "alternative_identity_min": 0.8, "alternative_aligned_pairs_min": 50, "split_priority": ["test", "validation", "train"]}, "stages": {split: {"source": sum(record.split == split for record in records), "retained": len(retained[split]), "excluded": sum(record.split == split for record in records) - len(retained[split])} for split in SPLITS}, "graph": graph_summary, "exact_duplicate_groups": len(duplicates), "standard_values_equal": True}
        _write_json(stage_audit / "summary.json", summary)
        source_hashes = {}
        for path in sorted(source.iterdir()):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                source_hashes[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        manifest = {**summary, "source_files": source_hashes, "encoding": {"amino_acids": AA_METADATA, "labels": LABEL_METADATA, "source_to_course_labels": SOURCE_TO_COURSE}, "id_mapping": [{"split": record.split, "source_record_id": record.source_id, "course_id": course_ids[record.source_id]} for record in records if record.source_id in course_ids]}
        _write_json(metadata_dir / "manifest.json", manifest)
        # Recheck immediately before publication to catch newly created targets.
        _guard_paths(source, standard, output, audit, overwrite)
        _publish([(stage_output, output), (stage_audit, audit)])
        return summary
    finally:
        for temporary in (stage_output, stage_audit):
            if temporary.exists():
                shutil.rmtree(temporary)


