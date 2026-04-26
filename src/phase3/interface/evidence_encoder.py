"""Evidence encoder: BERT predictions -> BN hard + virtual evidence.

Upgrades (Phase 3 final architecture):
- Damped log-odds aggregation with PER-NODE correlation alpha
  (sqrt(1 + alpha * (n - 1)) denominator) and clipping for stability.
- Empty-confidence list -> 0.5 (max uncertainty).
- Per-node calibrated thresholds (loaded from results/phase3/thresholds.json
  if present; fallback defaults otherwise).
- Per-node alpha (loaded from results/phase3/alpha_map.json if present;
  fallback ALPHA_DEFAULTS otherwise).
- Scaled absence penalty (importance weight x (1 - aggregated_score)).
- Absent-biased virtual evidence injection into BN with bounded strength
  (cap at 0.85; ABSENCE_VE_SCALE knob persisted in feedback_config.json).
- ``clause_count`` per BN type (report-only — not injected into BN).
- Returns a dict carrying hard_evidence, virtual_evidence, aggregated_scores,
  clause_count, absence_penalty, thresholds (the encoder's expanded contract).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLAUSE_MAP: dict[str, str] = {
    "Payment": "Has_Payment_Clause",
    "Termination": "Has_Termination_Clause",
    "Liability": "Has_Liability_Clause",
    "Confidentiality": "Has_Confidentiality_Clause",
    "Dispute Resolution": "Has_Dispute_Resolution_Clause",
}

# --- Per-node defaults --------------------------------------------------------

THRESHOLD_DEFAULTS: dict[str, float] = {
    "Has_Payment_Clause":            0.55,
    "Has_Termination_Clause":        0.50,
    "Has_Liability_Clause":          0.60,
    "Has_Confidentiality_Clause":    0.55,
    "Has_Dispute_Resolution_Clause": 0.50,
}

ALPHA_DEFAULTS: dict[str, float] = {
    "Has_Payment_Clause":            0.5,
    "Has_Termination_Clause":        0.5,
    "Has_Liability_Clause":          0.3,
    "Has_Confidentiality_Clause":    0.4,
    "Has_Dispute_Resolution_Clause": 0.4,
}

ABSENCE_WEIGHTS: dict[str, float] = {
    "Has_Payment_Clause":            0.6,
    "Has_Termination_Clause":        0.6,
    "Has_Liability_Clause":          1.0,
    "Has_Confidentiality_Clause":    0.4,
    "Has_Dispute_Resolution_Clause": 0.3,
}

# Default ABSENCE_VE_SCALE (overridden by feedback_config.json if present).
# Caps absent-biased virtual evidence at 0.85 to avoid double-counting hard "Absent".
ABSENCE_VE_SCALE_DEFAULT: float = 0.6
ABSENCE_VE_PMAX: float = 0.85

# --- Persisted-config loaders -------------------------------------------------

_THRESHOLDS_PATH = Path("results/phase3/thresholds.json")
_ALPHA_PATH = Path("results/phase3/alpha_map.json")
_FEEDBACK_PATH = Path("results/phase3/feedback_config.json")


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_thresholds() -> dict[str, float]:
    persisted = _load_json_dict(_THRESHOLDS_PATH)
    return {**THRESHOLD_DEFAULTS, **{k: float(v) for k, v in persisted.items()
                                     if k in THRESHOLD_DEFAULTS}}


def load_alpha_map() -> dict[str, float]:
    persisted = _load_json_dict(_ALPHA_PATH)
    return {**ALPHA_DEFAULTS, **{k: float(v) for k, v in persisted.items()
                                 if k in ALPHA_DEFAULTS}}


def load_absence_ve_scale() -> float:
    persisted = _load_json_dict(_FEEDBACK_PATH)
    val = persisted.get("absence_ve_scale", ABSENCE_VE_SCALE_DEFAULT)
    try:
        return float(val)
    except (TypeError, ValueError):
        return ABSENCE_VE_SCALE_DEFAULT


# --- Aggregation --------------------------------------------------------------

def aggregate_confidence(confs: list[float], alpha: float = 0.4) -> float:
    """Damped log-odds aggregation.

    - Empty list -> 0.5 (max uncertainty).
    - Clip raw confidences to [eps, 1-eps] and log-odds to +/- 4 to avoid
      saturating the aggregated signal.
    - Damping denom = sqrt(1 + alpha * (n - 1)) corrects for correlated
      evidence without over-penalising large evidence sets (sqrt(n) was too
      aggressive for n >> 1).
    """
    if not confs:
        return 0.5
    arr = np.clip(np.asarray(confs, dtype=float), 1e-6, 1.0 - 1e-6)
    log_odds = np.log(arr / (1.0 - arr))
    log_odds = np.clip(log_odds, -4.0, 4.0)
    n = len(arr)
    denom = float(np.sqrt(1.0 + alpha * max(n - 1, 0)))
    damped = float(log_odds.mean()) / max(denom, 1e-9)
    return float(1.0 / (1.0 + np.exp(-damped)))


# --- Absence -> virtual evidence ---------------------------------------------

def absence_to_virtual(
    penalty: float,
    scale: float | None = None,
    pmax: float = ABSENCE_VE_PMAX,
) -> list[float]:
    """Map a scaled absence penalty in [0, 1] to a virtual CPD.

    Returns ``[p_absent, p_present]`` with ``p_absent`` capped at ``pmax``
    (default 0.85) and globally softened by ``scale``. Set scale=0 to disable.

    NOTE: ordering is ``[Absent, Present]``. The state-aware
    ``build_virtual_cpds`` re-orders to match the BN model's state vector.
    """
    if scale is None:
        scale = load_absence_ve_scale()
    if scale <= 0:
        return [0.5, 0.5]
    p_absent = 0.5 + (pmax - 0.5) * float(np.clip(penalty, 0.0, 1.0)) * float(scale)
    p_absent = float(min(p_absent, pmax))
    return [p_absent, 1.0 - p_absent]


# --- Encoder ------------------------------------------------------------------

def _aggregate_per_node(
    confs_by_node: dict[str, list[float]],
    alpha_map: dict[str, float],
) -> dict[str, float]:
    return {
        node: aggregate_confidence(confs_by_node.get(node, []),
                                   alpha=alpha_map.get(node, 0.4))
        for node in CLAUSE_MAP.values()
    }


def encode_evidence(
    bert_outputs: list[dict],
    *,
    thresholds: dict[str, float] | None = None,
    alpha_map: dict[str, float] | None = None,
    absence_ve_scale: float | None = None,
) -> dict:
    """Encode BERT outputs into the expanded encoder contract.

    Backwards-compatibility: callers that only consume ``hard_evidence`` should
    look in the returned dict's ``hard_evidence`` key.

    Returns:
        {
            "hard_evidence":     {node: "Present" | "Absent"},
            "virtual_evidence":  {node: [p_absent, p_present]},
            "aggregated_scores": {node: float},
            "clause_count":      {node: int},
            "absence_penalty":   {node: float},  # scaled, only for absent nodes
            "thresholds":        {node: float},
            "alpha_map":         {node: float},
            "absence_ve_scale":  float,
        }
    """
    th_map = thresholds if thresholds is not None else load_thresholds()
    a_map = alpha_map if alpha_map is not None else load_alpha_map()
    ve_scale = absence_ve_scale if absence_ve_scale is not None else load_absence_ve_scale()

    confs_by_node: dict[str, list[float]] = defaultdict(list)
    for clause in bert_outputs or []:
        node = CLAUSE_MAP.get(clause.get("clause_type"))
        if node is None:
            continue
        try:
            conf = float(clause.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        confs_by_node[node].append(conf)

    aggregated_scores = _aggregate_per_node(confs_by_node, a_map)
    clause_count = {node: len(confs_by_node.get(node, [])) for node in CLAUSE_MAP.values()}

    hard_evidence: dict[str, str] = {}
    for node in CLAUSE_MAP.values():
        threshold = float(th_map.get(node, 0.5))
        hard_evidence[node] = (
            "Present" if aggregated_scores[node] >= threshold else "Absent"
        )

    absence_penalty: dict[str, float] = {}
    virtual_evidence: dict[str, list[float]] = {}
    for node in CLAUSE_MAP.values():
        score = aggregated_scores[node]
        if hard_evidence[node] == "Present":
            virtual_evidence[node] = [1.0 - score, score]
            continue
        weight = float(ABSENCE_WEIGHTS.get(node, 0.5))
        penalty = float(weight * (1.0 - score))
        absence_penalty[node] = penalty
        virtual_evidence[node] = absence_to_virtual(penalty, scale=ve_scale)

    return {
        "hard_evidence": hard_evidence,
        "virtual_evidence": virtual_evidence,
        "aggregated_scores": aggregated_scores,
        "clause_count": clause_count,
        "absence_penalty": absence_penalty,
        "thresholds": dict(th_map),
        "alpha_map": dict(a_map),
        "absence_ve_scale": float(ve_scale),
    }


__all__ = [
    "CLAUSE_MAP",
    "THRESHOLD_DEFAULTS",
    "ALPHA_DEFAULTS",
    "ABSENCE_WEIGHTS",
    "ABSENCE_VE_PMAX",
    "ABSENCE_VE_SCALE_DEFAULT",
    "aggregate_confidence",
    "absence_to_virtual",
    "encode_evidence",
    "load_thresholds",
    "load_alpha_map",
    "load_absence_ve_scale",
]
