"""
Miscellaneous shared utilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


LABEL_NAMES = ["keep", "turn", "bc"]


def idx2label(idx: int) -> str:
    """Map a class index to its name."""
    return LABEL_NAMES[idx]


def compute_metrics(labels, predictions):
    """Compute a comprehensive metrics dictionary.

    Parameters
    ----------
    labels      : array-like of int, shape (N,)
    predictions : array-like of int, shape (N,)

    Returns
    -------
    dict
        ``accuracy``, ``per_class`` (precision/recall/f1 per class),
        ``macro`` (macro-averaged), ``weighted`` (frequency-weighted).
        Macro-F1 is the primary metric reported in the thesis.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "per_class": {
            "recall":    recall_score(labels, predictions, average=None, zero_division=0),
            "f1":        f1_score(labels, predictions, average=None, zero_division=0),
            "precision": precision_score(labels, predictions, average=None, zero_division=0),
        },
        "macro": {
            "recall":    recall_score(labels, predictions, average="macro", zero_division=0),
            "f1":        f1_score(labels, predictions, average="macro", zero_division=0),
            "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        },
        "weighted": {
            "recall":    recall_score(labels, predictions, average="weighted", zero_division=0),
            "f1":        f1_score(labels, predictions, average="weighted", zero_division=0),
            "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        },
    }


def format_metrics(metrics: dict, header: str = "EVALUATION RESULTS") -> str:
    """Render a metrics dict as a human-readable text block."""
    sep_thick = "=" * 60
    sep_thin = "-" * 60
    lines = [sep_thick, header, sep_thick, "",
             f"Overall Accuracy: {metrics['accuracy']:.4f}", "",
             sep_thin, "Per-Class Metrics:", sep_thin,
             f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}",
             sep_thin]
    for idx, name in enumerate(LABEL_NAMES):
        lines.append(
            f"{name:<10} "
            f"{metrics['per_class']['precision'][idx]:<12.4f} "
            f"{metrics['per_class']['recall'][idx]:<12.4f} "
            f"{metrics['per_class']['f1'][idx]:<12.4f}"
        )
    lines += ["", sep_thin, "Macro-Averaged Metrics:", sep_thin,
              f"Precision: {metrics['macro']['precision']:.4f}",
              f"Recall:    {metrics['macro']['recall']:.4f}",
              f"F1-Score:  {metrics['macro']['f1']:.4f}",
              "", sep_thin, "Weighted-Averaged Metrics:", sep_thin,
              f"Precision: {metrics['weighted']['precision']:.4f}",
              f"Recall:    {metrics['weighted']['recall']:.4f}",
              f"F1-Score:  {metrics['weighted']['f1']:.4f}",
              sep_thick]
    return "\n".join(lines)


def count_parameters(model) -> int:
    """Number of trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
