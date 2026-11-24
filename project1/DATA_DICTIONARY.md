# Project 1 Data Dictionary

## 1. Predefined splits

Use `train` for model fitting and `validation` for candidate comparison and configuration selection. Test sequences may be used for the prescribed cross-split sequence-similarity check and for final prediction, but not to revise processing rules or model configurations. Keep the released split membership and apply the same label-independent processing rules to all three splits. Test labels are not provided.

The source files contain 8,803 train records, 1,102 validation records, and 1,102 test records. After applying the relationship rules stated in the assignment, the standard data contain 8,084, 1,057, and 1,102 records, respectively.

## 2. Source data: `source_data/`

### 2.1 `train_raw.csv` and `validation_raw.csv`

| Field | Meaning |
|---|---|
| `source_record_id` | Stable source-record identifier; also used as `query` and `target` in the similarity file |
| `sequence` | Text representation of the source sequence, stored as a single-element bytes-list wrapper that may contain whitespace |
| `ss3` | Text representation of the integer Q3 sequence; source encoding is `0=C, 1=H, 2=E` |

Accept only the specified single-element bytes-list form when parsing `sequence`. Remove internal ASCII whitespace and convert the sequence to uppercase. Use explicit parsing or a safe parser such as `ast.literal_eval`; do not use `eval` on source text. The `ss3` field is a separate format: integers are separated by whitespace inside brackets and may span multiple lines. Parse it accordingly, then check the brackets, integer range, and number of labels.

### 2.2 `test_raw.csv`

The test source file contains `source_record_id` and `sequence` only. It does not contain `ss3` or another structural label.

### 2.3 `partition_manifest.csv`

This file records the predefined split, source filename, source-record count, and label availability.

### 2.4 `candidate_similarity_hits.tsv.gz`

This gzip-compressed, tab-separated file contains candidate sequence alignments. It can be read directly with pandas or Python's `csv` module. Its fields are:

- `query`, `target`: source-record identifiers;
- `query_length`, `target_length`: full sequence lengths;
- `query_start`, `query_end`, `target_start`, `target_end`: aligned intervals;
- `query_alignment`, `target_alignment`: equal-length aligned strings; `-` denotes a gap;
- `evalue`, `bit_score`: similarity-search statistics.

The file includes self-hits and candidates that do not pass the relationship rule. It is not a final edge list or exclusion list. See the assignment brief for the released rule and the definitions of the three derived statistics.

## 3. Standard data: `standard_data/`

Each split is stored under `standard_data/<split>/`:

- `X.npy`: a `uint8` matrix of shape `number of proteins × 698`; `PAD=0`, the 20 standard amino acids use codes `1–20`, and `UNK=21`;
- `Y.npy`: an `int16` matrix with `H=0`, `E=1`, `C=2`, and `-100` at padded positions; available for train and validation only;
- `ids.npy`: a one-dimensional Unicode array of course sample IDs, aligned row by row with the retained source records; these strings are not the source-record IDs;
- `lengths.npy`: a one-dimensional array containing the original sequence lengths.

After applying the exclusion rule, pair the retained source IDs in file order with the course IDs in `ids.npy` for that split. Keep both IDs in your audit so the correspondence can be checked.

The complete mappings are stored in `standard_data/metadata/`. The source and course Q3 encodings differ, so labels must be remapped explicitly.

The course labels H, E, and C denote helical, beta-strand, and other structural states. C is a valid structural label that includes turns, bends, and states not assigned to H or E. It is not a missing label and does not necessarily indicate intrinsic disorder.

Input and label padding have different roles. `X=0` is the input vocabulary's PAD index. `Y=-100` is not a fourth structural class; it marks positions excluded from the loss and metrics. Valid positions in both matrices are determined by `lengths.npy`.

The standard matrices support baseline experiments, program checks, and comparison with the output of your source-data pipeline. Determine whether a source record is retained from the source fields and released rules, then compare counts, order, and encoded values with the standard data.

## 4. Prediction file

`predictions.npz` contains exactly two arrays:

- `ids`: a one-dimensional Unicode array in the released split order;
- `q3`: an integer matrix of shape `N × 698`, containing `0`, `1`, or `2` at valid residues and `-100` at padded positions.

Q3 and macro-F1 are computed over real residue positions only.
