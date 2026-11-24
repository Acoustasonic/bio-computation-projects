"""Load and validate the standard project1 matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class ProteinSequenceSplit:
    """Arrays belonging to one predefined data split."""

    inputs: np.ndarray
    ids: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray | None = None

    @property
    def valid_mask(self) -> np.ndarray:
        """Return positions that belong to the original, unpadded sequences."""

        positions = np.arange(self.inputs.shape[1])[None, :]
        return positions < self.lengths[:, None]


class StandardDataModule:
    """Load and validate the course-provided fixed-width matrices.

    The input and label matrices deliberately use different padding values:
    ``0`` is the embedding padding index in ``X.npy``, whereas ``-100`` marks
    label positions ignored by the classification loss in ``Y.npy``.
    """

    def __init__(
        self,
        data_dir: Path,
        input_padding_value: int = 0,
        label_padding_value: int = -100,
    ) -> None:
        self.data_dir = data_dir
        self.input_padding_value = input_padding_value
        self.label_padding_value = label_padding_value

    def load_split(self, split_name: str, *, validate: bool = True) -> ProteinSequenceSplit:
        """Load one split without changing its released row order."""

        if split_name not in {"train", "validation", "test"}:
            raise ValueError("Split must be train, validation, or test.")
        split_dir = self.data_dir / split_name
        label_path = split_dir / "Y.npy"
        if split_name == "test" and label_path.exists():
            raise ValueError("Test labels are not part of the released task and must not be loaded.")
        if split_name != "test" and not label_path.is_file():
            raise ValueError(f"{split_name}: labels are required.")
        split = ProteinSequenceSplit(
            inputs=np.load(split_dir / "X.npy", allow_pickle=False),
            ids=np.load(split_dir / "ids.npy", allow_pickle=False),
            lengths=np.load(split_dir / "lengths.npy", allow_pickle=False),
            labels=(
                np.load(label_path, allow_pickle=False)
                if label_path.exists()
                else None
            ),
        )
        if validate:
            self.validate_split(split_name, split)
        return split

    def validate_split(self, split_name: str, split: ProteinSequenceSplit) -> None:
        """Check shape, identifier, encoding, and padding invariants."""

        inputs, ids, lengths = split.inputs, split.ids, split.lengths
        if inputs.ndim != 2 or inputs.shape[1] != 698 or inputs.shape[0] == 0:
            raise ValueError(f"{split_name}: X.npy must have nonempty shape [N, 698].")
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError(f"{split_name}: ids.npy must be a one-dimensional Unicode array.")
        if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError(f"{split_name}: lengths.npy must be a one-dimensional integer array.")
        if len(ids) != inputs.shape[0] or lengths.shape != (inputs.shape[0],):
            raise ValueError(f"{split_name}: X, ids, and lengths row counts do not agree.")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{split_name}: duplicate protein IDs were found.")
        if not np.issubdtype(inputs.dtype, np.integer):
            raise ValueError(f"{split_name}: X.npy must contain integer codes.")
        if not np.all((lengths > 0) & (lengths <= inputs.shape[1])):
            raise ValueError(f"{split_name}: lengths contains an invalid sequence length.")

        valid = split.valid_mask
        if not np.all(inputs[~valid] == self.input_padding_value):
            raise ValueError(f"{split_name}: input padding does not match lengths.npy.")
        if not np.all((inputs[valid] >= 1) & (inputs[valid] <= 21)):
            raise ValueError(f"{split_name}: an invalid amino-acid code was found.")

        if split.labels is None:
            return
        if split.labels.shape != inputs.shape:
            raise ValueError(f"{split_name}: X.npy and Y.npy must have identical shapes.")
        if not np.issubdtype(split.labels.dtype, np.integer):
            raise ValueError(f"{split_name}: Y.npy must contain integer label codes.")
        if not np.all(split.labels[~valid] == self.label_padding_value):
            raise ValueError(f"{split_name}: label padding does not match lengths.npy.")
        if not np.all(np.isin(split.labels[valid], (0, 1, 2))):
            raise ValueError(f"{split_name}: an invalid Q3 label was found.")


def validate_predictions(
    split: ProteinSequenceSplit, ids: np.ndarray, q3: np.ndarray
) -> None:
    """Validate prediction identity, residue codes and right padding together."""

    if ids.ndim != 1 or ids.dtype.kind != "U" or not np.array_equal(ids, split.ids):
        raise ValueError("Prediction IDs must be Unicode and match the released order exactly.")
    if q3.shape != split.inputs.shape or q3.dtype.kind not in "iu":
        raise ValueError("Prediction q3 must be an integer matrix with shape [N, 698].")
    if not np.all(np.isin(q3[split.valid_mask], (0, 1, 2))):
        raise ValueError("Valid prediction positions must contain H=0, E=1 or C=2.")
    if not np.all(q3[~split.valid_mask] == -100):
        raise ValueError("Prediction padding must be -100.")


def save_predictions(path: Path, split: ProteinSequenceSplit, q3: np.ndarray) -> None:
    """Write only the two arrays required by the course prediction format."""

    validate_predictions(split, split.ids, q3)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=split.ids, q3=q3)


class ProteinDataset(Dataset):
    """Expose NumPy matrices to a PyTorch ``DataLoader``."""

    def __init__(self, split: ProteinSequenceSplit) -> None:
        self.inputs = split.inputs
        self.labels = split.labels

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.from_numpy(self.inputs[index].astype(np.int64, copy=False))
        if self.labels is None:
            return inputs
        labels = torch.from_numpy(self.labels[index].astype(np.int64, copy=False))
        return inputs, labels
