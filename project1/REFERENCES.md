# References and implementation provenance

## Course materials

- `project1_assignment.pdf`: task, fixed relationship rule, split priority, evaluation and submission contract.
- `DATA_DICTIONARY.md`: raw sequence/ss3 formats and encodings.
- `READING_GUIDE.md`: method-selection background. The two candidates below are simplified single-sequence implementations, not reproductions of the full published systems.
- Supplied `models.py`, `factories.py`, `pipeline.py` and `starter/run_baseline.py`: the course convolutional baseline is retained; orchestration was extended to record comparable artifacts and reload selected models.

## Method background

1. Wang S, Peng J, Ma J, Xu J. Protein secondary structure prediction using deep convolutional neural fields. *Scientific Reports*, 2016, 6:18962. [DOI](https://doi.org/10.1038/srep18962).
   The course reading guide motivates learning sequence context with convolutions. Our candidate uses residual dilated convolutions to change context range. It has no CRF/neural-field output layer, alignment profile or structural template and is not an implementation of the complete paper.
2. Heffernan R, Yang Y, Paliwal K, Zhou Y. Capturing non-local interactions by long short-term memory bidirectional recurrent neural networks for improving prediction of protein secondary structure, backbone angles, contact numbers and solvent accessibility. *Bioinformatics*, 2017, 33:2842–2849. [DOI](https://doi.org/10.1093/bioinformatics/btx218).
   Our candidate uses bidirectional recurrence over the amino-acid sequence. It omits the paper's additional input features and structural-property outputs; only H/E/C classification is implemented.
3. Kabsch W, Sander C. Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features. *Biopolymers*, 1983, 22:2577–2637. [DOI](https://doi.org/10.1002/bip.360221211).
   Background for structural labels. The assignment's released three-state encoding and mapping, rather than a new DSSP calculation, define this project.

These entries follow the supplied reading guide. No published performance is presented as a directly comparable course score, and no third-party paper implementation was copied.

## Libraries and tools

- Python `ast.literal_eval` and `csv`: safe parsing of the specified source formats.
- NumPy: encoding, saved arrays and metric aggregation.
- scikit-learn: per-class F1, using the course H/E/C order.
- PyTorch 2.5.1 (course dependency): Conv1d, Embedding, LSTM, packed sequences, AdamW and tensor checkpoints. [Documentation](https://pytorch.org/docs/stable/index.html). Reload explicitly uses `weights_only=True`; metadata use primitive types.
- Matplotlib: validation figures. Dependency pins remain in the supplied requirements files; actual versions are recorded only when a run executes.
- OpenAI Codex: assisted implementation, interface review and documentation. This code-only stage did not execute preprocessing, model code, training, prediction, evaluation, tests or packaging. No experimental results are claimed.
