"""Evaluation helpers for Phase 3 outputs."""

from __future__ import annotations

from sklearn.metrics import classification_report


def summarize_classification(y_true: list, y_pred: list) -> dict:
    """Return a dictionary report for downstream rendering/logging."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)

