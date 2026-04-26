"""Generate strict Hybrid evaluation artifacts for ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.phase3.bayesian.bootstrap import ensure_seed_model
from src.phase3.hybrid_pipeline import AgastyaHybridPipeline

# Expanded set — aligned with the full _PHASE2_TO_PHASE3 mapping in phase2_adapter.py.
# Covers all CUAD labels that now feed BN evidence nodes (26 labels across 5 clause types).
_PAYMENT_LABELS = {
    "Revenue/Profit Sharing", "Minimum Commitment", "Price Restrictions",
    "Liquidated Damages", "Most Favored Nation",
}
_TERMINATION_LABELS = {
    "Termination For Convenience", "Notice Period To Terminate Renewal",
    "Post-Termination Services", "Renewal Term",
}
_LIABILITY_LABELS = {
    "Cap On Liability", "Uncapped Liability", "Warranty Duration", "Insurance",
}
_CONFIDENTIALITY_LABELS = {
    "Non-Compete", "Non-Disparagement", "No-Solicit Of Customers",
    "No-Solicit Of Employees", "Ip Ownership Assignment", "Joint Ip Ownership",
    "Source Code Escrow", "Non-Transferable License", "Irrevocable Or Perpetual License",
}
_DISPUTE_LABELS = {"Governing Law", "Covenant Not To Sue", "Audit Rights"}

# All risky labels (union of Payment, Termination, Liability, Confidentiality)
# Dispute resolution REDUCES risk (it's a mitigating factor).
_RISKY_LABELS = (
    _PAYMENT_LABELS | _TERMINATION_LABELS | _LIABILITY_LABELS | _CONFIDENTIALITY_LABELS
)


def _derive_contract_risk(labels: set[str]) -> str:
    """Derive 3-class risk aligned with BN clause-type evidence nodes.

    Scoring: each of the 4 risky clause *types* (Payment, Termination, Liability,
    Confidentiality) that has at least one label present contributes +1.
    Dispute Resolution presence subtracts 1 (mitigating factor).
    High = score >= 2, Medium = score == 1, Low = score == 0.
    """
    type_score = 0
    if labels & _PAYMENT_LABELS:
        type_score += 1
    if labels & _TERMINATION_LABELS:
        type_score += 1
    if labels & _LIABILITY_LABELS:
        type_score += 1
    if labels & _CONFIDENTIALITY_LABELS:
        type_score += 1
    if labels & _DISPUTE_LABELS:
        type_score = max(0, type_score - 1)
    if type_score >= 2:
        return "High"
    if type_score == 1:
        return "Medium"
    return "Low"


def build_contract_dataset(test_df: pd.DataFrame) -> pd.DataFrame:
    grouped = test_df.groupby("filename", as_index=False)
    rows = []
    for filename, frame in grouped:
        labels = set(str(x).strip() for x in frame["label"].dropna().tolist())
        full_text = " ".join(str(x) for x in frame["text"].dropna().tolist())
        rows.append(
            {
                "filename": filename,
                "text": full_text,
                "true_risk": _derive_contract_risk(labels),
            }
        )
    return pd.DataFrame(rows)


def generate_hybrid_eval_artifact(
    *,
    test_csv_path: str = "data/processed/test.csv",
    output_json_path: str = "reports/phase3/hybrid_eval.json",
    bn_model_path: str = "results/phase3/bayesian_network.pkl",
    bert_checkpoint_path: str = "results/phase2/models/legal_bert_phase2.pt",
    label_map_path: str = "results/phase2/label2id.json",
    adapter_path: str | None = "results/phase2/models/legal_bert_lora_adapter",
) -> dict:
    test_df = pd.read_csv(test_csv_path)
    contract_df = build_contract_dataset(test_df)

    model_path = ensure_seed_model(bn_model_path)
    pipeline = AgastyaHybridPipeline(
        bn_model_path=model_path,
        bert_checkpoint_path=bert_checkpoint_path,
        label_map_path=label_map_path,
        adapter_path=adapter_path,
    )

    y_true = contract_df["true_risk"].tolist()
    y_pred: list[str] = []
    for text in contract_df["text"].tolist():
        out = pipeline.predict(text)
        risk = out.get("risk_level")
        if risk not in {"Low", "Medium", "High"}:
            risk = "Low"
        y_pred.append(risk)

    labels = ["Low", "Medium", "High"]
    payload = {
        "task": "contract_risk_level_3way",
        "n_contracts": int(len(contract_df)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "ground_truth": "Derived from Phase 2 test labels using deterministic risk-label mapping.",
        "risk_labels": sorted(_RISKY_LABELS),
    }

    out = Path(output_json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

