from __future__ import annotations

import numpy as np

from src.phase3.interface.confidence_mapper import map_confidence_to_virtual_evidence
from src.phase3.interface.evidence_encoder import CLAUSE_MAP, encode_evidence
from src.phase3.interface.feature_extractor import extract_cross_clause_features


def test_encode_evidence_schema():
    inputs = [{"clause_type": "Payment", "confidence": 0.9}]
    evidence = encode_evidence(inputs)
    assert "hard_evidence" in evidence
    assert "virtual_evidence" in evidence
    assert "aggregated_scores" in evidence
    assert "clause_count" in evidence
    assert "absence_penalty" in evidence
    assert "thresholds" in evidence
    assert set(evidence["hard_evidence"].keys()) == set(CLAUSE_MAP.values())
    assert all(v in {"Present", "Absent"} for v in evidence["hard_evidence"].values())
    assert evidence["hard_evidence"]["Has_Payment_Clause"] == "Present"
    assert evidence["clause_count"]["Has_Payment_Clause"] == 1


def test_aggregate_confidence_empty_returns_uncertain():
    from src.phase3.interface.evidence_encoder import aggregate_confidence
    assert aggregate_confidence([]) == 0.5


def test_absence_to_virtual_capped():
    from src.phase3.interface.evidence_encoder import absence_to_virtual, ABSENCE_VE_PMAX
    cpd = absence_to_virtual(1.0, scale=1.0)
    assert cpd[0] <= ABSENCE_VE_PMAX + 1e-9
    assert abs(sum(cpd) - 1.0) < 1e-9


def test_absence_ve_scale_zero_disables():
    from src.phase3.interface.evidence_encoder import absence_to_virtual
    cpd = absence_to_virtual(1.0, scale=0.0)
    assert cpd == [0.5, 0.5]


def test_virtual_evidence_probabilities_sum_to_one():
    inputs = [{"clause_type": "Payment", "confidence": 0.7}]
    ve = map_confidence_to_virtual_evidence(inputs)
    assert "Has_Payment_Clause" in ve
    assert abs(sum(ve["Has_Payment_Clause"]) - 1.0) < 1e-9


def test_cross_clause_feature_is_bounded():
    emb = {
        "Payment": np.ones(8, dtype=float),
        "Termination": np.zeros(8, dtype=float),
    }
    value = extract_cross_clause_features(emb)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0

