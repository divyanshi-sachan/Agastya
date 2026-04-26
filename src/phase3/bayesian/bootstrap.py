"""Bootstrap a valid seed Bayesian model for local runs."""

from __future__ import annotations

from pathlib import Path

from src.phase3.bayesian.cpt_definitions import get_seed_cpts
from src.phase3.bayesian.em_trainer import save_model
from src.phase3.bayesian.network import build_network


def build_seed_model():
    model = build_network()
    model.add_cpds(*get_seed_cpts())
    if not model.check_model():
        raise ValueError("Seed BN model failed consistency checks.")
    return model


def ensure_seed_model(path: str = "results/phase3/bayesian_network.pkl") -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        save_model(build_seed_model(), str(out))
    return str(out)

