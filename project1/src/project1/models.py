"""Neural network modules used by the project1 baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConvolutionalResidueClassifier(nn.Module):
    """Predict one Q3 label per sequence position from local context."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if min(vocabulary_size, embedding_dim, hidden_dim, num_classes) <= 0:
            raise ValueError("Model dimensions must be positive integers.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")
        padding = kernel_size // 2
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[batch, sequence_length, classes]``."""

        hidden = self.embedding(inputs).transpose(1, 2)
        encoded = self.encoder(hidden)
        return self.classifier(encoded).transpose(1, 2)


class DilatedResidualResidueClassifier(nn.Module):
    """Masked residual convolutions with receptive field ``1+(k-1)*sum(d)``.

    Each residual block contains one convolution. With kernel 5 and dilations
    (1, 2, 4, 8), the exact nominal receptive field is 61 sequence positions.
    Only token 0 is padding; token 21 remains an ordinary unknown residue.
    """

    def __init__(
        self, vocabulary_size: int, embedding_dim: int, hidden_dim: int,
        kernel_size: int, dropout: float, num_classes: int,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not dilations or any(type(d) is not int or d <= 0 for d in dilations):
            raise ValueError("dilations must contain positive integers.")
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.projection = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=1)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                      padding=dilation * (kernel_size // 2), dilation=dilation)
            for dilation in dilations
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return ``[B, L, 3]`` logits, with zero logits at padding positions."""
        mask = inputs.ne(0).unsqueeze(1)
        hidden = self.embedding(inputs).transpose(1, 2) * mask
        hidden = torch.relu(self.projection(hidden)) * mask
        for convolution in self.convolutions:
            update = self.dropout(torch.relu(convolution(hidden))) * mask
            hidden = torch.relu(hidden + update) * mask
        return (self.classifier(hidden) * mask).transpose(1, 2)


class BidirectionalLSTMResidueClassifier(nn.Module):
    """Encode right-padded sequences using their actual nonzero lengths."""

    def __init__(
        self, vocabulary_size: int, embedding_dim: int, hidden_dim: int,
        dropout: float, num_classes: int, num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Restore the original padded width and zero all PAD logits."""
        mask = inputs.ne(0)
        lengths = mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("BiLSTM inputs must contain at least one residue per sequence.")
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=inputs.shape[1],
        )
        return self.classifier(self.dropout(hidden)) * mask.unsqueeze(-1)
