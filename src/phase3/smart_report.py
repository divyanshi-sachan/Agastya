"""Smart Report builder for the hybrid pipeline.

Synthesises the BN posterior, evidence, and feedback-loop trace into a
structured report with four sections:
  - risk_score     : level + probabilities + uncertainty (entropy)
  - clause_explanation : per-clause + aggregated_scores + clause_count + absence
  - conflict_reasoning : raw + calibrated conflict + iteration trace + warnings
  - top_risk_factors   : prior-corrected importance, ranked

Future work (deferred):
  - Clause-snippet-grounded reason strings (currently templated).
  - Isotonic regression on conflict signal (currently percentile-only).
"""

from __future__ import annotations

import math
from typing import Iterable

# Friendly clause-type labels for templated reason strings.
_NODE_TO_LABEL = {
    "Has_Payment_Clause": "Payment terms",
    "Has_Termination_Clause": "Termination clause",
    "Has_Liability_Clause": "Liability / indemnification",
    "Has_Confidentiality_Clause": "Confidentiality / IP",
    "Has_Dispute_Resolution_Clause": "Dispute resolution",
}

# Map BN intermediate / leaf nodes to their "risky" state for importance.
_RISKY_STATE = {
    "Has_Payment_Clause": "Present",
    "Has_Termination_Clause": "Present",
    "Has_Liability_Clause": "Present",
    "Has_Confidentiality_Clause": "Present",
    "Has_Dispute_Resolution_Clause": "Absent",  # missing dispute resolution = risk
    "Payment_Or_Termination_Risky": "Risky",
    "Liability_Or_Confidentiality_Risky": "Risky",
    "Cross_Clause_Conflict": "Conflict",
}


def _entropy(probs: Iterable[float]) -> float:
    total = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        total -= p * math.log(p + 1e-12)
    return float(total)


def _format_pct(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"


def _build_risk_score(probabilities: dict[str, float], risk_level: str) -> dict:
    probs_list = [float(probabilities.get(k, 0.0)) for k in ("Low", "Medium", "High")]
    return {
        "level": risk_level,
        "probabilities": {k: float(probabilities.get(k, 0.0)) for k in ("Low", "Medium", "High")},
        "confidence": float(probabilities.get(risk_level, 0.0)),
        "uncertainty": _entropy(probs_list),
    }


def _build_clause_explanation(
    bert_outputs: list[dict],
    encoder_payload: dict,
) -> dict:
    per_clause: list[dict] = []
    for clause in bert_outputs or []:
        per_clause.append(
            {
                "clause_type": clause.get("clause_type"),
                "phase2_label": clause.get("phase2_label"),
                "confidence": float(clause.get("confidence", 0.0)),
                "snippet": str(clause.get("clause_text", ""))[:240],
            }
        )
    return {
        "per_clause": per_clause,
        "aggregated_scores": {
            k: float(v) for k, v in encoder_payload.get("aggregated_scores", {}).items()
        },
        "clause_count": {
            k: int(v) for k, v in encoder_payload.get("clause_count", {}).items()
        },
        "absence_penalty": {
            k: float(v) for k, v in encoder_payload.get("absence_penalty", {}).items()
        },
        "thresholds": {
            k: float(v) for k, v in encoder_payload.get("thresholds", {}).items()
        },
        "hard_evidence": dict(encoder_payload.get("hard_evidence", {})),
    }


def _build_conflict_reasoning(
    raw_conflict: float,
    calibrated_conflict: float | None,
    cross_clause_posterior: dict[str, float] | None,
    iteration_trace: list[dict] | None,
    calibration_warning: str | None,
) -> dict:
    return {
        "raw_conflict_signal": float(raw_conflict),
        "calibrated_conflict_signal": (
            float(calibrated_conflict) if calibrated_conflict is not None else None
        ),
        "cross_clause_posterior": dict(cross_clause_posterior or {}),
        "iteration_trace": list(iteration_trace or []),
        "calibration_warning": calibration_warning,
    }


def _node_importance(
    node: str,
    posterior: float,
    prior: float,
) -> float:
    """``posterior * (1 - prior)`` — surfaces unexpectedly-risky nodes."""
    return float(max(0.0, posterior) * max(0.0, 1.0 - prior))


def _build_top_risk_factors(
    node_posteriors: dict[str, dict[str, float]],
    node_priors: dict[str, dict[str, float]],
    encoder_payload: dict,
    top_k: int = 3,
) -> list[dict]:
    items: list[dict] = []
    aggregated = encoder_payload.get("aggregated_scores", {})
    hard = encoder_payload.get("hard_evidence", {})
    for node, posterior_dict in node_posteriors.items():
        risky_state = _RISKY_STATE.get(node)
        if risky_state is None:
            continue
        post = float(posterior_dict.get(risky_state, 0.0))
        prior = float(node_priors.get(node, {}).get(risky_state, 0.5))
        importance = _node_importance(node, post, prior)
        items.append(
            {
                "node": node,
                "label": _NODE_TO_LABEL.get(node, node),
                "risky_state": risky_state,
                "posterior_risky": post,
                "prior_risky": prior,
                "importance": importance,
                "aggregated_score": float(aggregated.get(node, 0.0)) if node in aggregated else None,
                "hard_evidence": hard.get(node),
                "reason": _reason_template(node, risky_state, hard.get(node)),
            }
        )
    items.sort(key=lambda d: d["importance"], reverse=True)
    return items[:top_k]


def _reason_template(node: str, risky_state: str, hard_evidence: str | None) -> str:
    label = _NODE_TO_LABEL.get(node, node)
    if node == "Has_Dispute_Resolution_Clause":
        if hard_evidence == "Absent":
            return f"{label} appears to be MISSING — escalates contract risk."
        return f"{label} present — mitigates contract risk."
    if node == "Cross_Clause_Conflict":
        return "Semantic mismatch between Payment & Termination clauses signals cross-clause conflict."
    if hard_evidence == "Present":
        return f"{label} detected with high confidence — drives {risky_state.lower()} posterior."
    if hard_evidence == "Absent":
        return f"{label} appears to be MISSING — under-specified contract increases {risky_state.lower()} posterior."
    return f"{label} contributes to {risky_state.lower()} posterior."


def build_smart_report(
    *,
    risk_level: str,
    probabilities: dict[str, float],
    bert_outputs: list[dict],
    encoder_payload: dict,
    raw_conflict: float,
    calibrated_conflict: float | None,
    cross_clause_posterior: dict[str, float] | None,
    iteration_trace: list[dict] | None,
    calibration_warning: str | None,
    node_posteriors: dict[str, dict[str, float]],
    node_priors: dict[str, dict[str, float]],
    top_k: int = 3,
) -> dict:
    """Build the structured Smart Report payload."""
    return {
        "risk_score": _build_risk_score(probabilities, risk_level),
        "clause_explanation": _build_clause_explanation(bert_outputs, encoder_payload),
        "conflict_reasoning": _build_conflict_reasoning(
            raw_conflict=raw_conflict,
            calibrated_conflict=calibrated_conflict,
            cross_clause_posterior=cross_clause_posterior,
            iteration_trace=iteration_trace,
            calibration_warning=calibration_warning,
        ),
        "top_risk_factors": _build_top_risk_factors(
            node_posteriors=node_posteriors,
            node_priors=node_priors,
            encoder_payload=encoder_payload,
            top_k=top_k,
        ),
        "calibration_warning": calibration_warning,
    }


def format_text_report(report: dict) -> str:
    """Render the Smart Report as a human-readable plaintext block."""
    lines: list[str] = []
    rs = report.get("risk_score", {})
    lines.append(f"Risk Level: {rs.get('level', '?')} (confidence {_format_pct(rs.get('confidence', 0.0))})")
    probs = rs.get("probabilities", {})
    if probs:
        prob_str = ", ".join(f"{k}={_format_pct(v)}" for k, v in probs.items())
        lines.append(f"Probabilities: {prob_str}")
    lines.append(f"Uncertainty (entropy): {rs.get('uncertainty', 0.0):.3f}")

    if report.get("calibration_warning"):
        lines.append("")
        lines.append(f"⚠ {report['calibration_warning']}")

    lines.append("")
    lines.append("Top risk factors:")
    for i, factor in enumerate(report.get("top_risk_factors", []), start=1):
        lines.append(
            f"  {i}. [{factor.get('label')}] importance={factor.get('importance', 0.0):.3f} "
            f"(posterior={factor.get('posterior_risky', 0.0):.2f}, "
            f"prior={factor.get('prior_risky', 0.0):.2f})"
        )
        lines.append(f"     {factor.get('reason')}")

    cr = report.get("conflict_reasoning", {})
    lines.append("")
    lines.append(
        "Conflict signal: raw="
        f"{cr.get('raw_conflict_signal', 0.0):.3f}, "
        f"calibrated={cr.get('calibrated_conflict_signal')}"
    )
    if cr.get("iteration_trace"):
        lines.append(f"Feedback iterations: {len(cr['iteration_trace'])}")
    return "\n".join(lines)


__all__ = ["build_smart_report", "format_text_report"]
