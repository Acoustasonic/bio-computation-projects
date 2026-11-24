# Environment

Use Python 3.11. Open a terminal in `project1/` and check that `python --version` reports 3.11 before creating the environment.

```text
python -m venv .venv
```

If another Python version is on your path, use `py -3.11 -m venv .venv` on Windows or `python3.11 -m venv .venv` on Linux or macOS, with Python 3.11 installed.

Activate it in your shell.

- Linux or macOS: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below and in the README.

Install the project dependencies.

```text
python -m pip install -r requirements.txt
```

Install PyTorch for the commands in the README.

- Windows or Linux, CPU: `python -m pip install -r requirements-torch-cpu.txt`
- Apple silicon Mac, CPU: `python -m pip install torch==2.5.1`
- NVIDIA GPU: install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) in the same environment and use `--device cuda` in the baseline command.

Check the installation with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. The README uses the CPU command, which also works when the CUDA check prints `False`.

Record your actual dependency versions and run commands in your submission.

## Verified cloud run (2026-11-24)

The completed run used an isolated Python environment on Ubuntu 24.04 with one
NVIDIA GeForce RTX 4090 (24 GB), GPU 0 only, and 8 CPU computation threads.
The container limit was 10 CPU cores and 60 GiB RAM.

| Component | Actual version |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.5.1 |
| CUDA runtime used by PyTorch | 12.4 |
| NumPy | 1.26.4 |
| SciPy | 1.14.1 |
| scikit-learn | 1.5.2 |
| Matplotlib | 3.9.4 |

The GPU commands were the README training and prediction commands with
`--device cuda`, using these shell settings:

```sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
```

Each `outputs/runs/*/run_manifest.json` contains the actual command, environment,
source commit and model hash. No GPU or Python environment is bundled in the
submission; install dependencies before reloading the selected model.

The paired multi-seed extension uses seeds 42, 43, 44, 45 and 46 for all three
methods on the same processed train/validation split. Seed 42 reuses the original
three runs (commit `c3538979a95d348d54256d8a51c28c45c3ec5585`); the twelve
additional runs use commit `2c65968461749996ef39fb7ca7b343e70527a9bb`.
The training entry, model, configuration defaults, data loader, optimizer,
metrics and trainer core are unchanged between these commits. The run plan
records their file hashes, and each run saves its effective configuration.
Only the seed and output directory differ within a method. The same CUDA
environment and eight computation threads were used throughout.

Report mean and sample standard deviation (`ddof=1`) across the five runs.
Select a candidate method by mean validation Q3, then mean macro-F1 and model
name on ties. The submitted checkpoint uses the selected method's fixed seed
42; it is neither the best seed nor an ensemble. Current reference-array hashes
identify the arrays used for analysis. They are not historical per-run training
hashes; the original seed-42 data evidence is the preceding reconstruction audit
and its recorded validation IDs/counts.
