"""Inference helpers for Phase 3 BN.

Supports:
- Hard evidence (dict[str, str]).
- Virtual evidence supplied as ``dict[node, [p_absent_or_no, p_present_or_yes]]``,
  converted to a list of state-aware ``TabularCPD`` objects so pgmpy's
  ``BeliefPropagation.query`` accepts it correctly.

The encoder always emits virtual CPDs in ``[p_absent, p_present]`` (or for the
conflict node, ``[p_no_conflict, p_conflict]``). ``build_virtual_cpds`` reads
each node's state ordering from the BN and re-orders if the BN stores states
in the opposite direction (e.g. ``["Present", "Absent"]``).
"""

from __future__ import annotations

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

# Canonical positive/negative labels per node. The encoder emits virtual
# evidence as ``[p_negative, p_positive]`` and we re-order using these to match
# the BN's actual state vector.
_POSITIVE_STATES = {
    "Has_Payment_Clause": "Present",
    "Has_Termination_Clause": "Present",
    "Has_Liability_Clause": "Present",
    "Has_Confidentiality_Clause": "Present",
    "Has_Dispute_Resolution_Clause": "Present",
    "Cross_Clause_Conflict": "Conflict",
}
_NEGATIVE_STATES = {
    "Has_Payment_Clause": "Absent",
    "Has_Termination_Clause": "Absent",
    "Has_Liability_Clause": "Absent",
    "Has_Confidentiality_Clause": "Absent",
    "Has_Dispute_Resolution_Clause": "Absent",
    "Cross_Clause_Conflict": "No_Conflict",
}


def build_virtual_cpds(
    model: BayesianNetwork,
    virtual_evidence: dict[str, list[float]] | None,
) -> list[TabularCPD]:
    """Build a list of ``TabularCPD`` matching the BN's actual state ordering.

    The encoder always supplies values as ``[p_negative, p_positive]`` —
    e.g. ``[p_absent, p_present]`` — but ``pgmpy`` requires the values to
    line up with each CPD's ``state_names`` ordering.
    """
    if not virtual_evidence:
        return []

    cpds: list[TabularCPD] = []
    for node, probs in virtual_evidence.items():
        if probs is None or len(probs) != 2:
            continue
        p_neg, p_pos = float(probs[0]), float(probs[1])
        try:
            existing = model.get_cpds(node)
        except Exception:
            continue
        if existing is None:
            continue
        states = list(existing.state_names[node])
        pos_label = _POSITIVE_STATES.get(node)
        neg_label = _NEGATIVE_STATES.get(node)
        if pos_label is None or neg_label is None:
            continue
        if pos_label not in states or neg_label not in states:
            raise ValueError(
                f"Cannot align virtual evidence for {node}: "
                f"states {states} vs expected {{ {pos_label}, {neg_label} }}"
            )
        # Build values vector aligned with model's ordering.
        ordered = [p_neg if s == neg_label else p_pos for s in states]
        cpd = TabularCPD(
            variable=node,
            variable_card=2,
            values=[[v] for v in ordered],
            state_names={node: states},
        )
        cpds.append(cpd)
    return cpds


def run_inference(
    model: BayesianNetwork,
    evidence: dict[str, str] | None = None,
    virtual_evidence: dict[str, list[float]] | None = None,
    query_var: str = "Contract_Risk_Level",
    *,
    bp_engine: BeliefPropagation | None = None,
) -> dict:
    """Run belief propagation and return a normalized risk payload.

    ``virtual_evidence`` is auto-converted to a list of state-aware
    ``TabularCPD`` objects. Pass ``bp_engine`` to reuse a pre-built engine.
    """
    bp = bp_engine if bp_engine is not None else BeliefPropagation(model)
    hard_evidence = {k: v for k, v in (evidence or {}).items() if v is not None}
    vcpds = build_virtual_cpds(model, virtual_evidence)
    kwargs: dict = {"variables": [query_var], "evidence": hard_evidence}
    if vcpds:
        kwargs["virtual_evidence"] = vcpds
    try:
        result = bp.query(**kwargs)
    except TypeError:
        # Older pgmpy versions don't accept virtual_evidence; fall back gracefully.
        result = bp.query(variables=[query_var], evidence=hard_evidence)
    factor = result[query_var] if isinstance(result, dict) else result
    states = factor.state_names[query_var]
    values = factor.values
    probabilities = {state: float(values[idx]) for idx, state in enumerate(states)}
    risk_level = max(probabilities, key=probabilities.get)
    return {
        "distribution": factor,
        "risk_level": risk_level,
        "probabilities": probabilities,
    }


def query_node_posterior(
    model: BayesianNetwork,
    node: str,
    *,
    bp_engine: BeliefPropagation | None = None,
) -> dict[str, float]:
    """Marginal P(node) under no evidence — used for cached priors."""
    bp = bp_engine if bp_engine is not None else BeliefPropagation(model)
    factor = bp.query(variables=[node], evidence={})
    factor = factor[node] if isinstance(factor, dict) else factor
    return {s: float(factor.values[i]) for i, s in enumerate(factor.state_names[node])}
