"""Explicit training/inference entry points; importing this module starts no job."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from project1.config import BaselineConfig, ModelConfig
from project1.data import ProteinDataset, ProteinSequenceSplit, StandardDataModule, save_predictions
from project1.factories import build_model, build_optimizer
from project1.metrics import compute_q3_metrics


def json_ready(value: Any) -> Any:
    """Keep checkpoint/config metadata limited to portable JSON primitives."""
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_commit(project_root: Path) -> str | None:
    """Record the dedicated deployment repository, never an unrelated parent repo."""
    if not (project_root.parent / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root.parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def choose_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA index is unavailable.")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is unavailable.")
    elif device.type != "cpu":
        raise ValueError("Use cpu, cuda[:index] or mps.")
    return device


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_trained_model(checkpoint: Path, device_name: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load tensor state plus primitive metadata, without accepting arbitrary pickle code."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), dict):
        raise ValueError("Checkpoint must contain model_config and model_state_dict.")
    model_config = dict(payload["model_config"])
    if model_config.get("num_classes") != 3 or model_config.get("vocabulary_size") != 22:
        raise ValueError("Checkpoint does not use the course input and Q3 encoding.")
    if payload.get("label_encoding", {"H": 0, "E": 1, "C": 2}) != {"H": 0, "E": 1, "C": 2}:
        raise ValueError("Checkpoint label mapping differs from the course encoding.")
    if "dilations" in model_config:
        model_config["dilations"] = tuple(model_config["dilations"])
    model = build_model(ModelConfig(**model_config))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(choose_device(device_name)).eval()
    return model, payload


@torch.inference_mode()
def infer_model(model: nn.Module, split: ProteinSequenceSplit, *, device: torch.device,
                batch_size: int, ignore_index: int = -100) -> tuple[np.ndarray, float | None]:
    """Keep the full released width for baseline-compatible convolution boundaries."""
    loader = DataLoader(ProteinDataset(split), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    predictions = []
    total_loss, count = 0.0, 0
    model.eval()
    for batch in loader:
        if split.labels is None:
            inputs, labels = batch, None
        else:
            inputs, labels = batch
        logits = model(inputs.to(device))
        if logits.shape != (*inputs.shape, 3):
            raise ValueError("Model must return logits with shape [batch, 698, 3].")
        predictions.append(logits.argmax(dim=-1).cpu().numpy().astype(np.int16))
        if labels is not None:
            labels = labels.to(device)
            total_loss += float(criterion(logits.reshape(-1, 3), labels.reshape(-1)).item())
            count += int(labels.ne(ignore_index).sum().item())
    if not predictions:
        raise ValueError("Cannot infer an empty split.")
    q3 = np.concatenate(predictions)
    q3[~split.valid_mask] = ignore_index
    return q3, total_loss / count if count else None


class BaselinePipeline:
    """Shared trainer; legacy class name is retained for the released baseline CLI."""

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.device = choose_device(config.runtime.device)
        set_random_seed(config.training.seed)
        torch.set_num_threads(config.training.num_threads)
        self.data = StandardDataModule(config.data.data_dir, config.data.input_padding_value,
                                       config.data.label_padding_value)
        self.model = build_model(config.model).to(self.device)
        self.history: list[dict[str, Any]] = []
        self.best_epoch = 0
        self.training_seconds = 0.0

    def fit(self, train_split: ProteinSequenceSplit,
            validation_split: ProteinSequenceSplit) -> dict[str, torch.Tensor]:
        if train_split.labels is None or validation_split.labels is None:
            raise ValueError("Training and validation require labels.")
        training = self.config.training
        optimizer = build_optimizer(self.model, training)
        criterion = nn.CrossEntropyLoss(ignore_index=self.config.data.label_padding_value)
        loader = DataLoader(ProteinDataset(train_split), batch_size=training.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(training.seed))
        best_score, best_state, stale = -float("inf"), None, 0
        self.history = []
        started = time.perf_counter()
        for epoch in range(1, training.max_epochs + 1):
            self.model.train()
            total_loss, count = 0.0, 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                if logits.shape != (*inputs.shape, 3):
                    raise ValueError("Model logits must have shape [batch, 698, 3].")
                loss = criterion(logits.reshape(-1, 3), labels.reshape(-1))
                loss.backward()
                if training.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), training.gradient_clip_norm)
                optimizer.step()
                valid_count = int(labels.ne(self.config.data.label_padding_value).sum().item())
                total_loss += float(loss.item()) * valid_count
                count += valid_count
            if count == 0:
                raise ValueError("Training data contain no valid residues.")
            predictions, validation_loss = infer_model(
                self.model, validation_split, device=self.device, batch_size=training.batch_size)
            metrics = compute_q3_metrics(validation_split.labels, predictions)
            row = {"epoch": epoch, "train_loss": total_loss / count,
                   "validation_loss": validation_loss, **metrics}
            self.history.append(row)
            print(f"epoch={epoch} train_loss={row['train_loss']:.5f} "
                  f"validation_q3={metrics['q3']:.5f} macro_f1={metrics['macro_f1']:.5f}", flush=True)
            score = float(metrics[training.selection_metric])
            if score > best_score:
                best_score, self.best_epoch, stale = score, epoch, 0
                best_state = {name: value.detach().cpu().clone()
                              for name, value in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= training.patience:
                    break
        self.training_seconds = time.perf_counter() - started
        if best_state is None:
            raise RuntimeError("No checkpoint was produced.")
        self.model.load_state_dict(best_state)
        return best_state

    def predict(self, split: ProteinSequenceSplit) -> np.ndarray:
        unlabeled = ProteinSequenceSplit(split.inputs, split.ids, split.lengths)
        return infer_model(self.model, unlabeled, device=self.device,
                           batch_size=self.config.training.batch_size)[0]

    def run(self, *, predict_test: bool = False, overwrite: bool = False) -> dict[str, Any]:
        """Select checkpoints using validation only; test inference is opt-in."""
        output = self.config.runtime.output_dir.resolve()
        data_dir = self.config.data.data_dir.resolve()
        project_root = Path(__file__).resolve().parents[2]
        protected = [data_dir, project_root / "source_data", project_root / "standard_data"]
        if any(output == path.resolve() or output.is_relative_to(path.resolve()) for path in protected):
            raise ValueError("Run outputs must be outside course/source data directories.")
        artifacts = ["model.pt", "metrics.json", "config.json", "history.json", "history.csv",
                     "validation_predictions.npz", "run_manifest.json", "baseline_model.pt",
                     "baseline_validation.npz", "baseline_metrics.json", "baseline_test.npz"]
        occupied = [name for name in artifacts if (output / name).exists()]
        if occupied and not overwrite:
            raise FileExistsError("Run artifacts already exist. Choose a new output directory or --overwrite.")
        output.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for name in occupied:
                path = output / name
                if not path.is_file():
                    raise ValueError(f"Refusing to overwrite non-file artifact: {path}")
                path.unlink()
        manifest: dict[str, Any] = {"status": "running", "model_name": self.config.model.name,
                                    "test_used_for_selection": False, "command": sys.argv}
        write_json(output / "config.json", asdict(self.config))
        write_json(output / "run_manifest.json", manifest)
        try:
            train_split = self.data.load_split("train")
            validation_split = self.data.load_split("validation")
            state = self.fit(train_split, validation_split)
            predictions, loss = infer_model(self.model, validation_split, device=self.device,
                                           batch_size=self.config.training.batch_size)
            metrics = compute_q3_metrics(validation_split.labels, predictions)
            metrics.update(best_epoch=self.best_epoch, validation_loss=loss,
                           selection_metric=self.config.training.selection_metric,
                           parameter_count=sum(p.numel() for p in self.model.parameters()),
                           training_seconds=self.training_seconds)
            checkpoint = {"format_version": 1, "model_state_dict": state,
                          "model_config": json_ready(asdict(self.config.model)),
                          "config": json_ready(asdict(self.config)), "best_epoch": self.best_epoch,
                          "validation_metrics": metrics, "label_encoding": {"H": 0, "E": 1, "C": 2},
                          "input_padding": 0, "ignore_index": -100}
            torch.save(checkpoint, output / "model.pt")
            save_predictions(output / "validation_predictions.npz", validation_split, predictions)
            write_json(output / "metrics.json", metrics)
            write_json(output / "history.json", self.history)
            with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "q3",
                                                            "macro_f1", "f1_H", "f1_E", "f1_C"])
                writer.writeheader()
                for row in self.history:
                    h, e, c = row["class_f1_H_E_C"]
                    writer.writerow({key: row[key] for key in ["epoch", "train_loss", "validation_loss", "q3", "macro_f1"]}
                                    | {"f1_H": h, "f1_E": e, "f1_C": c})
            # Retain filenames documented by the released starter, without changing its CNN.
            if self.config.model.name == "convolutional":
                torch.save(checkpoint, output / "baseline_model.pt")
                save_predictions(output / "baseline_validation.npz", validation_split, predictions)
                write_json(output / "baseline_metrics.json", metrics)
            if predict_test:
                split = self.data.load_split("test")
                save_predictions(output / "baseline_test.npz", split, self.predict(split))
            manifest.update(status="complete", best_epoch=self.best_epoch,
                            environment={"python": platform.python_version(), "numpy": np.__version__,
                                         "torch": str(torch.__version__), "device": str(self.device),
                                         "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
                                         "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None},
                            source_commit=source_commit(project_root),
                            validation_ids_sha256=hashlib.sha256(validation_split.ids.tobytes()).hexdigest(),
                            train_proteins=len(train_split.ids), validation_proteins=len(validation_split.ids),
                            validation_residues=int(validation_split.valid_mask.sum()),
                            model_sha256=hashlib.sha256((output / "model.pt").read_bytes()).hexdigest())
            write_json(output / "run_manifest.json", manifest)
            return metrics
        except Exception as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(output / "run_manifest.json", manifest)
            raise
