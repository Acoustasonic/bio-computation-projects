"""Configuration objects for the project1 course baseline."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Location and encoding values used by the standard matrices."""

    data_dir: Path
    input_padding_value: int = 0
    label_padding_value: int = -100


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Parameters passed to the selected model factory."""

    name: str
    vocabulary_size: int
    embedding_dim: int
    hidden_dim: int
    kernel_size: int
    dropout: float
    num_classes: int
    num_layers: int = 2
    dilations: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization and early-stopping settings."""

    optimizer: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int
    num_threads: int
    weight_decay: float = 0.01
    gradient_clip_norm: float = 0.0
    selection_metric: str = "q3"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution device and output location."""

    device: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Complete configuration for one baseline run."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = payload.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return section


def _resolve_path(value: object, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Configuration paths must be nonempty strings without NUL characters.")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _integer(section: Mapping[str, Any], key: str, minimum: int | None = None,
             default: int | None = None) -> int:
    value = section.get(key, default)
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"'{key}' must be an integer" +
                         (f" >= {minimum}." if minimum is not None else "."))
    return value


def _number(section: Mapping[str, Any], key: str, minimum: float,
            default: float | None = None, *, positive: bool = False) -> float:
    value = section.get(key, default)
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or (positive and value == minimum)):
        raise ValueError(f"'{key}' must be a finite number {'>' if positive else '>='} {minimum}.")
    return float(value)


def _choice(section: Mapping[str, Any], key: str, choices: set[str],
            default: str | None = None) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"'{key}' must be one of {', '.join(sorted(choices))}.")
    return value


def _known_keys(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} configuration keys: {', '.join(sorted(unknown))}.")


def load_config(config_path: Path, project_root: Path) -> BaselineConfig:
    """Load a JSON configuration and resolve project-relative paths."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("The baseline configuration must contain a JSON object.")

    data = _require_mapping(payload, "data")
    model = _require_mapping(payload, "model")
    training = _require_mapping(payload, "training")
    runtime = _require_mapping(payload, "runtime")

    _known_keys(payload, {"data", "model", "training", "runtime"}, "top-level")
    _known_keys(data, {"data_dir", "input_padding_value", "label_padding_value"}, "data")
    _known_keys(model, {"name", "vocabulary_size", "embedding_dim", "hidden_dim",
                        "kernel_size", "dropout", "num_classes", "num_layers", "dilations"}, "model")
    _known_keys(training, {"optimizer", "learning_rate", "batch_size", "max_epochs", "patience",
                           "seed", "num_threads", "weight_decay", "gradient_clip_norm",
                           "selection_metric"}, "training")
    _known_keys(runtime, {"device", "output_dir"}, "runtime")
    input_padding = _integer(data, "input_padding_value", default=0)
    label_padding = _integer(data, "label_padding_value", default=-100)
    vocabulary_size = _integer(model, "vocabulary_size", 1)
    num_classes = _integer(model, "num_classes", 1)
    if (input_padding, label_padding, vocabulary_size, num_classes) != (0, -100, 22, 3):
        raise ValueError("Encoding requires PAD=0, label ignore=-100, vocabulary_size=22 and num_classes=3.")
    kernel_size = _integer(model, "kernel_size", 1)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    dropout = _number(model, "dropout", 0.0)
    if dropout >= 1.0:
        raise ValueError("dropout must be less than 1.")
    dilations = model.get("dilations", (1, 2, 4, 8))
    if (not isinstance(dilations, (list, tuple)) or not dilations
            or any(type(value) is not int or value < 1 for value in dilations)):
        raise ValueError("dilations must be a nonempty list of positive integers.")
    device = runtime.get("device")
    if not isinstance(device, str) or re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device) is None:
        raise ValueError("device must be cpu, mps, cuda or cuda:<nonnegative index>.")

    return BaselineConfig(
        data=DataConfig(
            data_dir=_resolve_path(data["data_dir"], project_root),
            input_padding_value=input_padding,
            label_padding_value=label_padding,
        ),
        model=ModelConfig(
            name=_choice(model, "name", {"convolutional", "dilated_cnn", "bilstm"}),
            vocabulary_size=vocabulary_size,
            embedding_dim=_integer(model, "embedding_dim", 1),
            hidden_dim=_integer(model, "hidden_dim", 1),
            kernel_size=kernel_size,
            dropout=dropout,
            num_classes=num_classes,
            num_layers=_integer(model, "num_layers", 1, default=2),
            dilations=tuple(dilations),
        ),
        training=TrainingConfig(
            optimizer=_choice(training, "optimizer", {"adamw"}),
            learning_rate=_number(training, "learning_rate", 0.0, positive=True),
            batch_size=_integer(training, "batch_size", 1),
            max_epochs=_integer(training, "max_epochs", 1),
            patience=_integer(training, "patience", 1),
            seed=_integer(training, "seed", 0),
            num_threads=_integer(training, "num_threads", 1),
            weight_decay=_number(training, "weight_decay", 0.0, default=0.01),
            gradient_clip_norm=_number(training, "gradient_clip_norm", 0.0, default=0.0),
            selection_metric=_choice(training, "selection_metric", {"q3", "macro_f1"}, default="q3"),
        ),
        runtime=RuntimeConfig(
            device=device,
            output_dir=_resolve_path(runtime["output_dir"], project_root),
        ),
    )
