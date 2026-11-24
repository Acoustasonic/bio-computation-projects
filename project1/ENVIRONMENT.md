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
