"""Cross-clause semantic feature extraction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_cross_clause_features(embeddings: dict[str, np.ndarray]) -> float:
    """
    Compute a conflict signal in [0, 1] based on Payment/Termination mismatch.
    """
    if "Payment" in embeddings and "Termination" in embeddings:
        sim = cosine_similarity(
            embeddings["Payment"].reshape(1, -1),
            embeddings["Termination"].reshape(1, -1),
        )[0][0]
        return float(np.clip(1.0 - sim, 0.0, 1.0))
    return 0.5

