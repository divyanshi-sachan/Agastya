"""Virtual evidence mapping from BERT confidence."""

from __future__ import annotations

from .evidence_encoder import CLAUSE_MAP


def map_confidence_to_virtual_evidence(bert_outputs: list[dict]) -> dict[str, list[float]]:
    """
    Returns {node_name: [P(Absent), P(Present)]} for each presence node.
    """
    virtual_evidence: dict[str, list[float]] = {}
    for clause in bert_outputs:
        node = CLAUSE_MAP.get(clause.get("clause_type"))
        if node is None:
            continue
        confidence = float(clause.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
        virtual_evidence[node] = [1.0 - confidence, confidence]
    return virtual_evidence

