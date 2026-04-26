"""Phase 3 interface layer between BERT and Bayesian reasoning."""

from .confidence_mapper import map_confidence_to_virtual_evidence
from .evidence_encoder import CLAUSE_MAP, encode_evidence
from .feature_extractor import extract_cross_clause_features
from .phase2_adapter import Phase2BertAdapter, has_phase2_artifacts

__all__ = [
    "CLAUSE_MAP",
    "encode_evidence",
    "map_confidence_to_virtual_evidence",
    "extract_cross_clause_features",
    "Phase2BertAdapter",
    "has_phase2_artifacts",
]

