"""Load pre-built Phase 2 CSV splits (no filtering or re-splitting here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("filename", "text", "label", "label_id")
def load_dataset_manifest(path: str | Path) -> dict:
    """Load `dataset_manifest.json` from a processed directory (includes train label counts)."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def inverse_freq_weights(
    label_counts: dict[str, int],
    *,
    normalize: bool = False,
) -> dict[str, float]:
    """
    Per-class weights `1 / count` (inverse frequency).

    If ``normalize=True``, scales weights so they sum to 1 (stable coefficients vs raw inverses).
    """
    raw: dict[str, float] = {str(k): 1.0 / max(1, int(v)) for k, v in label_counts.items()}
    if not normalize:
        return raw
    total = sum(raw.values())
    if total <= 0.0:
        return raw
    return {k: v / total for k, v in raw.items()}


def load_label2id(path: str | Path) -> dict[str, int]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object mapping str->int in {p}")
    return {str(k): int(v) for k, v in data.items()}


def load_split_csv(
    path: str | Path,
    *,
    columns: tuple[str, ...] | None = None,
    required: tuple[str, ...] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """
    Read a single split written by `dl_dataset_builder.save_build_result`.

    Performs no transforms beyond pandas dtype parsing — keep training I/O deterministic.
    Extra columns (e.g. `category` for the binary Yes/No export) are kept unless `columns` is set.
    """
    p = Path(path)
    df = pd.read_csv(p)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing columns {missing}; found {list(df.columns)}")
    has_len = "length" in df.columns
    has_log = "log_length" in df.columns
    if has_len ^ has_log:
        raise ValueError(f"{p}: expected both length and log_length, got length={has_len}, log_length={has_log}")
    if columns is not None:
        miss = [c for c in columns if c not in df.columns]
        if miss:
            raise ValueError(f"{p} missing projected columns {miss}")
        return df[list(columns)].copy()
    return df.copy()


def load_processed_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    *,
    label2id_path: str | Path | None = None,
    columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int] | None]:
    """Load train / val / test and optionally the label map."""
    train = load_split_csv(train_csv, columns=columns)
    val = load_split_csv(val_csv, columns=columns)
    test = load_split_csv(test_csv, columns=columns)
    label2id = load_label2id(label2id_path) if label2id_path is not None else None
    return train, val, test, label2id


def hf_dict_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Small helper for `datasets.Dataset.from_list` without importing torch."""
    return frame.to_dict(orient="records")
