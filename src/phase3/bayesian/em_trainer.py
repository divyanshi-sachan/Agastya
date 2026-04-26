"""EM utilities for fitting/saving/loading the BN."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from pgmpy.estimators import ExpectationMaximization
from pgmpy.models import BayesianNetwork

from src.phase3.bayesian.cpt_definitions import get_seed_cpts
from src.phase3.bayesian.network import build_network


# Expanded label sets — aligned with phase2_adapter._PHASE2_TO_PHASE3
# and hybrid_eval._RISKY_LABELS. Covers 26 of 41 CUAD labels.
_PAYMENT_SRC = {
    "Revenue/Profit Sharing", "Minimum Commitment", "Price Restrictions",
    "Liquidated Damages", "Most Favored Nation",
}
_TERMINATION_SRC = {
    "Termination For Convenience", "Notice Period To Terminate Renewal",
    "Post-Termination Services", "Renewal Term",
}
_LIABILITY_SRC = {
    "Cap On Liability", "Uncapped Liability", "Warranty Duration", "Insurance",
}
_CONFIDENTIALITY_SRC = {
    "Non-Compete", "Non-Disparagement", "No-Solicit Of Customers",
    "No-Solicit Of Employees", "Ip Ownership Assignment", "Joint Ip Ownership",
    "Source Code Escrow", "Non-Transferable License", "Irrevocable Or Perpetual License",
}
_DISPUTE_SRC = {"Governing Law", "Covenant Not To Sue", "Audit Rights"}

_PRESENCE_COLUMNS = [
    "Has_Payment_Clause",
    "Has_Termination_Clause",
    "Has_Liability_Clause",
    "Has_Confidentiality_Clause",
    "Has_Dispute_Resolution_Clause",
]


def train_with_em(
    model: BayesianNetwork,
    train_df: pd.DataFrame,
    n_iter: int = 100,
) -> BayesianNetwork:
    model.fit(
        data=train_df,
        estimator=ExpectationMaximization,
        n_jobs=1,
        max_iter=n_iter,
    )
    return model


def save_model(model: BayesianNetwork, path: str) -> None:
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str) -> BayesianNetwork:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def prepare_bn_training_data(
    cuad_train_df: pd.DataFrame,
    bert_risk_labels: pd.DataFrame,
    conflict_labels: pd.Series,
    risk_labels: pd.Series,
) -> pd.DataFrame:
    """Assemble a per-contract BN training matrix with proper state names."""
    bn_df = pd.DataFrame()
    bn_df["Has_Payment_Clause"] = cuad_train_df["revenue_royalty_payment"].apply(
        lambda x: "Present" if x == 1 else "Absent"
    )
    bn_df["Has_Termination_Clause"] = cuad_train_df[
        "termination_for_convenience"
    ].apply(lambda x: "Present" if x == 1 else "Absent")
    bn_df["Has_Liability_Clause"] = cuad_train_df["limitation_of_liability"].apply(
        lambda x: "Present" if x == 1 else "Absent"
    )
    bn_df["Has_Confidentiality_Clause"] = cuad_train_df["non_compete"].apply(
        lambda x: "Present" if x == 1 else "Absent"
    )
    bn_df["Has_Dispute_Resolution_Clause"] = cuad_train_df["governing_law"].apply(
        lambda x: "Present" if x == 1 else "Absent"
    )
    bn_df["Payment_Or_Termination_Risky"] = bert_risk_labels["payment_term_risky"].apply(
        lambda x: "Risky" if bool(x) else "Not_Risky"
    )
    bn_df["Liability_Or_Confidentiality_Risky"] = bert_risk_labels[
        "liability_conf_risky"
    ].apply(lambda x: "Risky" if bool(x) else "Not_Risky")
    bn_df["Cross_Clause_Conflict"] = conflict_labels.apply(
        lambda x: "Conflict" if bool(x) else "No_Conflict"
    )
    bn_df["Contract_Risk_Level"] = risk_labels
    return bn_df


def build_bn_training_data_from_phase2_df(phase2_df: pd.DataFrame) -> pd.DataFrame:
    """Build BN-ready per-contract rows from processed Phase 2 data.

    Uses expanded label sets aligned with hybrid_eval._derive_contract_risk.
    Expected schema: filename, label (CUAD class name).
    """
    required_cols = {"filename", "label"}
    missing = required_cols - set(phase2_df.columns)
    if missing:
        raise ValueError(f"Missing columns for BN training data: {sorted(missing)}")

    contracts = sorted(phase2_df["filename"].dropna().unique().tolist())
    rows: list[dict[str, str]] = []
    for filename in contracts:
        labels = set(
            str(x).strip()
            for x in phase2_df.loc[phase2_df["filename"] == filename, "label"].dropna().tolist()
        )

        # Clause-type presence using expanded label sets
        payment_present       = bool(labels & _PAYMENT_SRC)
        termination_present   = bool(labels & _TERMINATION_SRC)
        liability_present     = bool(labels & _LIABILITY_SRC)
        confidentiality_present = bool(labels & _CONFIDENTIALITY_SRC)
        dispute_present       = bool(labels & _DISPUTE_SRC)

        r1_risky = payment_present or termination_present
        r2_risky = liability_present or confidentiality_present
        conflict = (r1_risky and r2_risky) or (
            not dispute_present and (r1_risky or r2_risky)
            and (payment_present or liability_present)
        )

        # Type-score risk (aligned with _derive_contract_risk in hybrid_eval.py)
        type_score = sum([
            payment_present, termination_present,
            liability_present, confidentiality_present,
        ])
        if dispute_present:
            type_score = max(0, type_score - 1)
        if type_score >= 2:
            risk_level = "High"
        elif type_score == 1:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        rows.append(
            {
                "Has_Payment_Clause":              "Present" if payment_present else "Absent",
                "Has_Termination_Clause":          "Present" if termination_present else "Absent",
                "Has_Liability_Clause":            "Present" if liability_present else "Absent",
                "Has_Confidentiality_Clause":      "Present" if confidentiality_present else "Absent",
                "Has_Dispute_Resolution_Clause":   "Present" if dispute_present else "Absent",
                "Payment_Or_Termination_Risky":    "Risky" if r1_risky else "Not_Risky",
                "Liability_Or_Confidentiality_Risky": "Risky" if r2_risky else "Not_Risky",
                "Cross_Clause_Conflict":           "Conflict" if conflict else "No_Conflict",
                "Contract_Risk_Level":             risk_level,
            }
        )
    return pd.DataFrame(rows)


def train_bn_from_phase2_processed(
    train_csv_path: str = "data/processed/train.csv",
    output_model_path: str = "results/phase3/bayesian_network.pkl",
    n_iter: int = 30,
) -> tuple[BayesianNetwork, pd.DataFrame]:
    """Train BN with EM using processed Phase 2 train split and persist to disk."""
    train_df = pd.read_csv(train_csv_path)
    bn_train_df = build_bn_training_data_from_phase2_df(train_df)

    model = build_network()
    model.add_cpds(*get_seed_cpts())
    if not model.check_model():
        raise ValueError("Seed model invalid before EM training.")

    trained = train_with_em(model, bn_train_df, n_iter=n_iter)
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(trained, str(out_path))
    return trained, bn_train_df

