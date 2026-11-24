"""Factories for replaceable project1 model and optimizer components."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn
from torch.optim import AdamW, Optimizer

from project1.config import ModelConfig, TrainingConfig
from project1.models import (
    BidirectionalLSTMResidueClassifier,
    ConvolutionalResidueClassifier,
    DilatedResidualResidueClassifier,
)


ModelBuilder = Callable[[ModelConfig], nn.Module]


def _build_convolutional_model(config: ModelConfig) -> nn.Module:
    return ConvolutionalResidueClassifier(
        vocabulary_size=config.vocabulary_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )


def _build_dilated_model(config: ModelConfig) -> nn.Module:
    return DilatedResidualResidueClassifier(
        vocabulary_size=config.vocabulary_size, embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim, kernel_size=config.kernel_size,
        dropout=config.dropout, num_classes=config.num_classes,
        dilations=config.dilations,
    )


def _build_bilstm_model(config: ModelConfig) -> nn.Module:
    return BidirectionalLSTMResidueClassifier(
        vocabulary_size=config.vocabulary_size, embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim, dropout=config.dropout,
        num_classes=config.num_classes, num_layers=config.num_layers,
    )


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "convolutional": _build_convolutional_model,
    "dilated_cnn": _build_dilated_model,
    "bilstm": _build_bilstm_model,
}


def build_model(config: ModelConfig) -> nn.Module:
    """Construct a model selected by its configuration name."""

    try:
        builder = MODEL_REGISTRY[config.name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        message = f"Unknown model '{config.name}'. Available models: {available}."
        raise ValueError(message) from error
    return builder(config)


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Construct the optimizer requested by the training configuration."""

    if config.optimizer == "adamw":
        return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError("Unknown optimizer. The course baseline currently provides: adamw.")
