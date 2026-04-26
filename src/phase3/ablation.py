"""Ablation utilities for DL vs Hybrid comparisons.

This module intentionally avoids fabricated metrics.
Rows are populated only from available result artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _empty_row(configuration: str) -> dict:
    return {
        "Configuration": configuration,
        "Macro-F1": float("nan"),
        "Accuracy": float("nan"),
        "Precision": float("nan"),
        "Recall": float("nan"),
        "Status": "missing",
        "Source": "",
        "EvidencePath": "",
        "Notes": "No validated artifact found.",
    }


def _to_float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _has_any_metric(row: dict) -> bool:
    metric_cols = ["Macro-F1", "Accuracy", "Precision", "Recall"]
    return pd.notna(pd.Series({k: row.get(k) for k in metric_cols})).any()


def build_ablation_table(
    *,
    phase2_results_path: str = "results/phase2/results.json",
    hybrid_results_path: str = "reports/phase3/hybrid_eval.json",
) -> pd.DataFrame:
    """
    Build an ablation table using only existing artifacts.

    - DL_Only: populated from results/phase2/results.json when present.
    - Hybrid: populated from reports/phase3/hybrid_eval.json when present.
    """
    rows = [
        _empty_row("DL_Only"),
        _empty_row("Hybrid"),
    ]
    by_name = {row["Configuration"]: row for row in rows}

    phase2_file = Path(phase2_results_path)
    if phase2_file.exists():
        payload = json.loads(phase2_file.read_text(encoding="utf-8"))
        row = by_name["DL_Only"]
        row["Macro-F1"] = _to_float_or_nan(payload.get("macro_f1"))
        row["Accuracy"] = _to_float_or_nan(payload.get("accuracy"))
        row["Source"] = "Phase 2 evaluation artifact"
        row["EvidencePath"] = str(phase2_file)
        if _has_any_metric(row):
            row["Status"] = "available"
            row["Notes"] = "Precision/Recall unavailable in current Phase 2 artifact."
        else:
            row["Status"] = "missing"
            row["Notes"] = "Phase 2 artifact exists but metric fields are empty."

    hybrid_file = Path(hybrid_results_path)
    if hybrid_file.exists():
        payload = json.loads(hybrid_file.read_text(encoding="utf-8"))
        row = by_name["Hybrid"]
        row["Macro-F1"] = _to_float_or_nan(payload.get("macro_f1"))
        row["Accuracy"] = _to_float_or_nan(payload.get("accuracy"))
        row["Precision"] = _to_float_or_nan(payload.get("precision"))
        row["Recall"] = _to_float_or_nan(payload.get("recall"))
        row["Source"] = "Phase 3 hybrid evaluation artifact"
        row["EvidencePath"] = str(hybrid_file)
        if _has_any_metric(row):
            row["Status"] = "available"
            row["Notes"] = "Strictly loaded from generated hybrid_eval.json."
        else:
            row["Status"] = "missing"
            row["Notes"] = "Hybrid artifact exists but metric fields are empty."

    return pd.DataFrame(rows)


def write_ablation_table(output_path: str = "reports/phase3/ablation_results.csv") -> pd.DataFrame:
    table = build_ablation_table()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table

