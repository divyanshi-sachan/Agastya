from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.phase3.bayesian.bootstrap import build_seed_model
from src.phase3.bayesian.cpt_definitions import get_seed_cpts
from src.phase3.bayesian.em_trainer import (
    build_bn_training_data_from_phase2_df,
    load_model,
    train_bn_from_phase2_processed,
)
from src.phase3.bayesian.inference import run_inference
from src.phase3.bayesian.network import build_network


def _build_seeded_model():
    model = build_network()
    model.add_cpds(*get_seed_cpts())
    return model


def test_bn_model_is_valid():
    model = _build_seeded_model()
    assert model.check_model() is True


def test_inference_returns_risk_distribution():
    model = _build_seeded_model()
    out = run_inference(
        model,
        {
            "Has_Payment_Clause": "Present",
            "Has_Termination_Clause": "Absent",
            "Has_Liability_Clause": "Present",
            "Has_Confidentiality_Clause": "Present",
            "Has_Dispute_Resolution_Clause": "Absent",
        },
    )
    assert set(out["probabilities"].keys()) == {"Low", "Medium", "High"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-9


def test_bn_training_data_builder_from_phase2_schema():
    source = pd.DataFrame(
        [
            {"filename": "a.pdf", "label": "Revenue/Profit Sharing"},
            {"filename": "a.pdf", "label": "Termination For Convenience"},
            {"filename": "a.pdf", "label": "Governing Law"},
            {"filename": "b.pdf", "label": "Document Name"},
        ]
    )
    bn_df = build_bn_training_data_from_phase2_df(source)
    assert len(bn_df) == 2
    assert set(bn_df.columns) == {
        "Has_Payment_Clause",
        "Has_Termination_Clause",
        "Has_Liability_Clause",
        "Has_Confidentiality_Clause",
        "Has_Dispute_Resolution_Clause",
        "Payment_Or_Termination_Risky",
        "Liability_Or_Confidentiality_Risky",
        "Cross_Clause_Conflict",
        "Contract_Risk_Level",
    }


def test_em_training_changes_cpd_and_persists_model(tmp_path):
    train_csv = tmp_path / "train.csv"
    out_model = tmp_path / "bayesian_network.pkl"
    train_df = pd.DataFrame(
        [
            {"filename": "a.pdf", "label": "Revenue/Profit Sharing"},
            {"filename": "a.pdf", "label": "Termination For Convenience"},
            {"filename": "a.pdf", "label": "Cap On Liability"},
            {"filename": "a.pdf", "label": "Non-Compete"},
            {"filename": "b.pdf", "label": "Document Name"},
            {"filename": "c.pdf", "label": "Governing Law"},
        ]
    )
    train_df.to_csv(train_csv, index=False)

    seed = build_seed_model()
    seed_values = seed.get_cpds("Has_Payment_Clause").values.copy()

    trained, _ = train_bn_from_phase2_processed(
        train_csv_path=str(train_csv),
        output_model_path=str(out_model),
        n_iter=5,
    )
    assert trained.check_model() is True
    assert Path(out_model).exists()

    loaded = load_model(str(out_model))
    assert loaded.check_model() is True
    trained_values = trained.get_cpds("Has_Payment_Clause").values
    assert not (trained_values == seed_values).all()

