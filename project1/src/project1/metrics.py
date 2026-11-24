"""Metrics shared by training and standalone evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def compute_q3_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    ignore_index: int = -100,
) -> dict[str, object]:
    """Compute residue-level Q3 accuracy and F1 scores on labelled positions."""

    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    valid = labels != ignore_index
    if not np.any(valid):
        raise ValueError("No labelled residue positions were provided.")
    truth = labels[valid]
    predicted = predictions[valid]
    if not np.all(np.isin(truth, (0, 1, 2))) or not np.all(np.isin(predicted, (0, 1, 2))):
        raise ValueError("Valid labels and predictions must use H=0, E=1 and C=2.")
    class_f1 = f1_score(
        truth,
        predicted,
        labels=(0, 1, 2),
        average=None,
        zero_division=0,
    )
    return {
        "q3": float(np.mean(truth == predicted)),
        "macro_f1": float(class_f1.mean()),
        "class_f1_H_E_C": class_f1.tolist(),
        "valid_residues": int(valid.sum()),
        "confusion_matrix_H_E_C": np.bincount(
            truth.astype(np.int64) * 3 + predicted.astype(np.int64), minlength=9
        ).reshape(3, 3).tolist(),
    }
