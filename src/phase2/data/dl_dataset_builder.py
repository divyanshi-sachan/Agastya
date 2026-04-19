"""
One-time Phase 2 dataset build from Phase 1 inputs (CUAD `master_clauses.csv`).

Mirrors the audited rules in `notebooks/Phase_1/Part_03_Feature_Engineering.ipynb`:
long table → Yes/No anomaly drop → optional task filter → clean text → optional min token
filter → `length` / `log_length` features → document-level splits → label encoding → CSV +
`label2id.json` + `dataset_manifest.json` (train label counts, class proportions).

Training code should import only `dataset_loader` against the emitted artifacts, not rebuild.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PLACEHOLDER_CLAUSE = frozenset({"[]", ""})

TaskName = Literal["multiclass_real_spans", "binary_yesno"]


def build_long_table(master_csv: Path) -> pd.DataFrame:
    """Unpivot master_clauses.csv to one row per (contract × category)."""
    df = pd.read_csv(master_csv, low_memory=False)
    cols = list(df.columns)
    pairs = [(cols[i], cols[i + 1]) for i in range(1, len(cols), 2)]
    rows: list[dict] = []
    for _, r in df.iterrows():
        file = str(r["Filename"])
        for ctxt_col, ans_col in pairs:
            clause = r[ctxt_col]
            ans = r[ans_col]
            c_str = "" if pd.isna(clause) else str(clause).strip()
            a_str = "" if pd.isna(ans) else str(ans).strip()
            is_ph = c_str in PLACEHOLDER_CLAUSE
            rows.append(
                {
                    "filename": file,
                    "category": ctxt_col,
                    "clause_text": c_str,
                    "answer": a_str,
                    "has_real_clause": not is_ph,
                }
            )
    return pd.DataFrame(rows)


def category_answer_profile(answers: pd.Series) -> dict:
    s = answers.astype(str).str.strip().replace({"nan": ""})
    uniq = set(s.unique()) - {""}
    is_yesno = uniq <= {"Yes", "No"} and len(uniq) > 0
    if is_yesno:
        pos = float((s == "Yes").mean())
        return {"kind": "yes_no", "positive_rate_yes": pos, "n_distinct_answers": len(uniq)}
    return {"kind": "other", "positive_rate_yes": float("nan"), "n_distinct_answers": len(uniq)}


def yes_no_category_names(long_df: pd.DataFrame) -> set[str]:
    rows_cs = []
    for cat, g in long_df.groupby("category", sort=False):
        prof = category_answer_profile(g["answer"])
        rows_cs.append({"category": cat, **prof})
    cat_stats = pd.DataFrame(rows_cs).set_index("category")
    yesno = cat_stats[cat_stats["kind"] == "yes_no"]
    return set(yesno.index)


def drop_yes_without_real_span(long_df: pd.DataFrame, yn_cats: set[str]) -> tuple[pd.DataFrame, int]:
    """Remove CUAD rows where answer is Yes but span is still a placeholder (Part 02/03)."""
    sub_yn = long_df[long_df["category"].isin(yn_cats)].copy()
    sub_yn = sub_yn[sub_yn["answer"].isin(["Yes", "No"])]
    bad = sub_yn[(sub_yn["answer"] == "Yes") & (~sub_yn["has_real_clause"])]
    n_bad = len(bad)
    cleaned = long_df.drop(index=bad.index)
    return cleaned, n_bad


def clean_clause_text(raw: str, char_cap: int = 4000) -> str:
    """Strip, collapse whitespace, cap length (same spirit as Part 03 `clause_to_model_text`)."""
    s = "" if pd.isna(raw) else str(raw).strip()
    if s in PLACEHOLDER_CLAUSE:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:char_cap] if char_cap > 0 else s


def word_counts(text: pd.Series) -> pd.Series:
    """Whitespace token count per row (matches `str.split()` semantics)."""
    return text.fillna("").astype(str).str.split().str.len().astype(int)


def add_length_features(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """`length` = token count, `log_length` = log1p(length) for hybrid / weighting features."""
    length = word_counts(text)
    log_length = np.log1p(length.astype(np.float64))
    return length, log_length


def _train_label_stats(train_df: pd.DataFrame, label_col: str = "label") -> dict:
    """Counts and proportions on the train split only (no leakage from val/test)."""
    vc = train_df[label_col].value_counts()
    label_counts_train = {str(k): int(v) for k, v in vc.items()}
    total = int(vc.sum()) if len(vc) else 0
    if total:
        class_distribution_train = {str(k): float(v) / float(total) for k, v in vc.items()}
    else:
        class_distribution_train = {}
    return {
        "label_counts_train": label_counts_train,
        "class_distribution_train": class_distribution_train,
    }


def _train_manifest_fields(train_df: pd.DataFrame, *, random_state: int, label_col: str = "label") -> dict:
    """Train-only manifest fields: class stats, mean token length (vocab / max-seq hint), split seed."""
    fields = _train_label_stats(train_df, label_col=label_col)
    if len(train_df):
        fields["avg_length"] = float(train_df["length"].astype(np.float64).mean())
    else:
        fields["avg_length"] = 0.0
    fields["seed"] = int(random_state)
    return fields


@dataclass
class BuildResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label2id: dict[str, int]
    manifest: dict


def _document_three_way_split(
    filenames: pd.Series,
    *,
    random_state: int,
    test_size: float,
    val_size_within_trainval: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Boolean masks aligned to `filenames` index: train, val, test."""
    unique = np.unique(filenames.astype(str).values)
    g_temp, g_test = train_test_split(
        unique,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    g_train, g_val = train_test_split(
        g_temp,
        test_size=val_size_within_trainval,
        random_state=random_state,
        shuffle=True,
    )
    s_train, s_val, s_test = set(g_train), set(g_val), set(g_test)
    fn = filenames.astype(str)
    return fn.isin(s_train), fn.isin(s_val), fn.isin(s_test)


def build_multiclass_real_spans(
    long_df: pd.DataFrame,
    *,
    yn_cats: set[str],
    random_state: int = 42,
    char_cap: int = 4000,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    min_words_gt: int | None = None,
) -> BuildResult:
    """
    Rows with real clause text only; label is CUAD category name (Part 05 / §8 export style).

    Document-level 80/10/10 split: first hold out `test_frac`, then hold `val_frac` of remainder.
    `val_frac` is applied to the (1 - test_frac) slice via sklearn's `train_test_split`, i.e.
    val_size = val_frac / (1 - test_frac) of that slice to target ~val_frac of all documents.
    """
    long_clean, n_bad = drop_yes_without_real_span(long_df, yn_cats)
    data = long_clean[long_clean["has_real_clause"]].copy()
    data["text"] = data["clause_text"].map(lambda c: clean_clause_text(c, char_cap))
    data = data[data["text"].str.len() > 0].reset_index(drop=True)
    if min_words_gt is not None:
        wc = word_counts(data["text"])
        data = data.loc[wc > min_words_gt].reset_index(drop=True)
    data["length"], data["log_length"] = add_length_features(data["text"])
    data["label"] = data["category"].astype(str)

    val_size_of_temp = val_frac / max(1e-9, (1.0 - test_frac))
    m_tr, m_va, m_te = _document_three_way_split(
        data["filename"],
        random_state=random_state,
        test_size=test_frac,
        val_size_within_trainval=val_size_of_temp,
    )
    train_df = data.loc[m_tr].reset_index(drop=True)
    val_df = data.loc[m_va].reset_index(drop=True)
    test_df = data.loc[m_te].reset_index(drop=True)

    labels_sorted = sorted(train_df["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}

    def add_ids(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["label_id"] = out["label"].map(label2id)
        missing = out["label_id"].isna().sum()
        if missing:
            raise ValueError(f"{int(missing)} labels in split not present in train label set")
        out["label_id"] = out["label_id"].astype(int)
        return out[["filename", "text", "label", "label_id", "length", "log_length"]]

    manifest = {
        "task": "multiclass_real_spans",
        "n_bad_yes_no_span_dropped": n_bad,
        "n_rows_train": len(train_df),
        "n_rows_val": len(val_df),
        "n_rows_test": len(test_df),
        "n_labels": len(label2id),
        "random_state": random_state,
        "char_cap": char_cap,
        "min_words_gt": min_words_gt,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "split": "document_level_three_way",
        **_train_manifest_fields(train_df, random_state=random_state),
    }
    return BuildResult(
        train=add_ids(train_df),
        val=add_ids(val_df),
        test=add_ids(test_df),
        label2id=label2id,
        manifest=manifest,
    )


def build_binary_yesno(
    long_df: pd.DataFrame,
    *,
    yn_cats: set[str],
    random_state: int = 42,
    char_cap: int = 4000,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    min_words_gt: int | None = None,
) -> BuildResult:
    """Per-row Yes/No on Yes/No categories (Part 03 §2); text may be empty for No + []."""
    long_clean, n_bad = drop_yes_without_real_span(long_df, yn_cats)
    sub = long_clean[long_clean["category"].isin(yn_cats)].copy()
    sub = sub[sub["answer"].isin(["Yes", "No"])].reset_index(drop=True)
    sub["text"] = sub["clause_text"].map(lambda c: clean_clause_text(c, char_cap))
    if min_words_gt is not None:
        wc = word_counts(sub["text"])
        sub = sub.loc[wc > min_words_gt].reset_index(drop=True)
    sub["length"], sub["log_length"] = add_length_features(sub["text"])
    sub["label"] = (sub["answer"] == "Yes").map({True: "Yes", False: "No"}).astype(str)
    sub["category"] = sub["category"].astype(str)

    val_size_of_temp = val_frac / max(1e-9, (1.0 - test_frac))
    m_tr, m_va, m_te = _document_three_way_split(
        sub["filename"],
        random_state=random_state,
        test_size=test_frac,
        val_size_within_trainval=val_size_of_temp,
    )
    train_df = sub.loc[m_tr].reset_index(drop=True)
    val_df = sub.loc[m_va].reset_index(drop=True)
    test_df = sub.loc[m_te].reset_index(drop=True)

    label2id = {"No": 0, "Yes": 1}

    def pack(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["label_id"] = out["label"].map(label2id).astype(int)
        return out[["filename", "category", "text", "label", "label_id", "length", "log_length"]]

    manifest = {
        "task": "binary_yesno",
        "n_bad_yes_no_span_dropped": n_bad,
        "n_rows_train": len(train_df),
        "n_rows_val": len(val_df),
        "n_rows_test": len(test_df),
        "n_labels": len(label2id),
        "random_state": random_state,
        "char_cap": char_cap,
        "min_words_gt": min_words_gt,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "split": "document_level_three_way",
        **_train_manifest_fields(train_df, random_state=random_state),
    }
    return BuildResult(
        train=pack(train_df),
        val=pack(val_df),
        test=pack(test_df),
        label2id=label2id,
        manifest=manifest,
    )


def build_dl_dataset(
    master_csv: Path,
    *,
    task: TaskName = "multiclass_real_spans",
    out_dir: Path | None = None,
    random_state: int = 42,
    char_cap: int = 4000,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    min_words_gt: int | None = None,
) -> BuildResult:
    out_dir = out_dir or Path("data/processed")
    long_df = build_long_table(master_csv)
    yn_cats = yes_no_category_names(long_df)
    if task == "multiclass_real_spans":
        result = build_multiclass_real_spans(
            long_df,
            yn_cats=yn_cats,
            random_state=random_state,
            char_cap=char_cap,
            test_frac=test_frac,
            val_frac=val_frac,
            min_words_gt=min_words_gt,
        )
    elif task == "binary_yesno":
        result = build_binary_yesno(
            long_df,
            yn_cats=yn_cats,
            random_state=random_state,
            char_cap=char_cap,
            test_frac=test_frac,
            val_frac=val_frac,
            min_words_gt=min_words_gt,
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return result


def save_build_result(result: BuildResult, out_dir: Path, *, label_map_name: str = "label2id.json") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.train.to_csv(out_dir / "train.csv", index=False)
    result.val.to_csv(out_dir / "val.csv", index=False)
    result.test.to_csv(out_dir / "test.csv", index=False)
    label_path = out_dir / label_map_name
    label_path.write_text(json.dumps(result.label2id, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(result.manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote:", out_dir / "train.csv", out_dir / "val.csv", out_dir / "test.csv", label_path, manifest_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 2 CSV splits from CUAD master_clauses.csv")
    p.add_argument("--master-csv", type=Path, default=Path("data/CUAD_v1/master_clauses.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--task",
        choices=("multiclass_real_spans", "binary_yesno"),
        default="multiclass_real_spans",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--char-cap", type=int, default=4000)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--label-map-name", type=str, default="label2id.json")
    p.add_argument(
        "--min-words-gt",
        type=int,
        default=None,
        metavar="N",
        help="If set, keep only rows with word count > N (e.g. 5 drops clauses with <=5 words).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.master_csv.is_file():
        print(f"Missing master CSV: {args.master_csv}", file=sys.stderr)
        return 1
    result = build_dl_dataset(
        args.master_csv,
        task=args.task,
        out_dir=args.out_dir,
        random_state=args.seed,
        char_cap=args.char_cap,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        min_words_gt=args.min_words_gt,
    )
    save_build_result(result, args.out_dir, label_map_name=args.label_map_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
