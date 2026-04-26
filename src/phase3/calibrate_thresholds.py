"""Joint calibration of (per-node threshold, per-node alpha, ENTROPY_GATE,
ABSENCE_VE_SCALE) on the validation contracts.

Outputs:
  - results/phase3/thresholds.json
  - results/phase3/alpha_map.json
  - results/phase3/feedback_config.json (ENTROPY_GATE + ABSENCE_VE_SCALE)
  - results/phase3/calibration_topk.json (top-5 configs + stability stats)

Optimisation strategy:
  - Coordinate ascent over BN evidence nodes (per-node (threshold, alpha)
    sweep) with 3 random restarts.
  - For each candidate config, run the full hybrid pipeline on val contracts
    (with feedback loop) and score by macro-F1 of Contract_Risk_Level.
  - After per-node coordinate ascent converges, sweep ENTROPY_GATE in
    {0.6, 0.75, 0.85, 0.95} and ABSENCE_VE_SCALE in {0.0, 0.3, 0.6, 0.9}
    using the best per-node vector.
  - Persist best per-node config + the feedback knobs.

Important: this module is imported lazily — running calibration is opt-in via
``python -m src.phase3.calibrate_thresholds``.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.phase3.bayesian.bootstrap import ensure_seed_model
from src.phase3.hybrid_eval import build_contract_dataset
from src.phase3.interface.evidence_encoder import (
    ALPHA_DEFAULTS,
    THRESHOLD_DEFAULTS,
)

logger = logging.getLogger(__name__)

THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
ALPHA_GRID = [0.2, 0.3, 0.4, 0.5, 0.6]
ENTROPY_GATE_GRID = [0.6, 0.75, 0.85, 0.95]
ABSENCE_VE_SCALE_GRID = [0.0, 0.3, 0.6, 0.9]
RESTARTS = 3
COORD_ASCENT_PASSES = 2

_NODES = list(THRESHOLD_DEFAULTS.keys())
_LABELS = ["Low", "Medium", "High"]


@dataclass
class CalibConfig:
    thresholds: dict[str, float] = field(default_factory=lambda: dict(THRESHOLD_DEFAULTS))
    alpha_map: dict[str, float] = field(default_factory=lambda: dict(ALPHA_DEFAULTS))
    entropy_gate: float = 0.85
    absence_ve_scale: float = 0.6

    def to_dict(self) -> dict:
        return {
            "thresholds": dict(self.thresholds),
            "alpha_map": dict(self.alpha_map),
            "entropy_gate": float(self.entropy_gate),
            "absence_ve_scale": float(self.absence_ve_scale),
        }


def _macro_f1(pipeline, contract_df: pd.DataFrame, config: CalibConfig) -> float:
    """Run pipeline against val contracts and compute macro-F1 vs ground truth."""
    pipeline.set_runtime_overrides(
        thresholds=config.thresholds,
        alpha_map=config.alpha_map,
        entropy_gate=config.entropy_gate,
        absence_ve_scale=config.absence_ve_scale,
    )
    y_true = contract_df["true_risk"].tolist()
    y_pred: list[str] = []
    for text in contract_df["text"].tolist():
        try:
            out = pipeline.predict(text)
            y_pred.append(out["risk_level"])
        except Exception as exc:
            logger.warning("Pipeline failed on a val contract during calibration: %s", exc)
            y_pred.append("Low")
    return float(f1_score(y_true, y_pred, labels=_LABELS, average="macro", zero_division=0))


def _coordinate_ascent_per_node(
    pipeline,
    contract_df: pd.DataFrame,
    config: CalibConfig,
    *,
    score_history: list[tuple[float, dict]],
    rng: random.Random,
) -> CalibConfig:
    """One full pass of coordinate ascent over per-node (threshold, alpha)."""
    best_config = CalibConfig(
        thresholds=dict(config.thresholds),
        alpha_map=dict(config.alpha_map),
        entropy_gate=config.entropy_gate,
        absence_ve_scale=config.absence_ve_scale,
    )
    best_score = _macro_f1(pipeline, contract_df, best_config)
    score_history.append((best_score, best_config.to_dict()))
    nodes = list(_NODES)
    rng.shuffle(nodes)
    for node in nodes:
        local_best = best_score
        local_best_th = best_config.thresholds[node]
        local_best_al = best_config.alpha_map[node]
        for th in THRESHOLD_GRID:
            for al in ALPHA_GRID:
                trial = CalibConfig(
                    thresholds={**best_config.thresholds, node: th},
                    alpha_map={**best_config.alpha_map, node: al},
                    entropy_gate=best_config.entropy_gate,
                    absence_ve_scale=best_config.absence_ve_scale,
                )
                score = _macro_f1(pipeline, contract_df, trial)
                score_history.append((score, trial.to_dict()))
                if score > local_best:
                    local_best = score
                    local_best_th = th
                    local_best_al = al
        best_config.thresholds[node] = local_best_th
        best_config.alpha_map[node] = local_best_al
        best_score = local_best
    return best_config


def _restart_initial(rng: random.Random, base: CalibConfig) -> CalibConfig:
    return CalibConfig(
        thresholds={n: rng.choice(THRESHOLD_GRID) for n in _NODES},
        alpha_map={n: rng.choice(ALPHA_GRID) for n in _NODES},
        entropy_gate=base.entropy_gate,
        absence_ve_scale=base.absence_ve_scale,
    )


def _topk_summary(score_history: list[tuple[float, dict]], k: int = 5) -> dict:
    """Top-k by score with stability stats (Pitfall #1)."""
    deduped: dict[str, tuple[float, dict]] = {}
    for score, cfg in score_history:
        key = json.dumps(cfg, sort_keys=True)
        existing = deduped.get(key)
        if existing is None or score > existing[0]:
            deduped[key] = (score, cfg)
    ordered = sorted(deduped.values(), key=lambda x: x[0], reverse=True)[:k]
    if not ordered:
        return {"configs": [], "best_macro_f1": 0.0}
    scores = [x[0] for x in ordered]
    cfgs = [x[1] for x in ordered]
    th_var = {
        n: float(statistics.pvariance([c["thresholds"][n] for c in cfgs]))
        for n in _NODES
    } if len(cfgs) > 1 else {n: 0.0 for n in _NODES}
    al_var = {
        n: float(statistics.pvariance([c["alpha_map"][n] for c in cfgs]))
        for n in _NODES
    } if len(cfgs) > 1 else {n: 0.0 for n in _NODES}
    return {
        "best_macro_f1": float(max(scores)),
        "topk_macro_f1_std": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        "topk_threshold_var_per_node": th_var,
        "topk_alpha_var_per_node": al_var,
        "configs": [{"score": s, **c} for s, c in zip(scores, cfgs)],
    }


def calibrate(
    *,
    val_csv_path: str = "data/processed/val.csv",
    bn_model_path: str = "results/phase3/bayesian_network.pkl",
    bert_checkpoint_path: str = "results/phase2/models/legal_bert_phase2.pt",
    label_map_path: str = "results/phase2/label2id.json",
    adapter_path: str | None = "results/phase2/models/legal_bert_lora_adapter",
    out_dir: str = "results/phase3",
    seed: int = 0,
) -> dict:
    """Run the full calibration sweep and persist all artifacts."""
    from src.phase3.hybrid_pipeline import AgastyaHybridPipeline

    rng = random.Random(seed)
    val_df = pd.read_csv(val_csv_path)
    contract_df = build_contract_dataset(val_df)

    pipeline = AgastyaHybridPipeline(
        bn_model_path=ensure_seed_model(bn_model_path),
        bert_checkpoint_path=bert_checkpoint_path,
        label_map_path=label_map_path,
        adapter_path=adapter_path,
        load_calibration=False,  # never read prior calibration during a calibration run
    )

    score_history: list[tuple[float, dict]] = []
    best_overall = CalibConfig()
    best_overall_score = _macro_f1(pipeline, contract_df, best_overall)
    score_history.append((best_overall_score, best_overall.to_dict()))

    for restart in range(RESTARTS):
        cfg = best_overall if restart == 0 else _restart_initial(rng, best_overall)
        for _ in range(COORD_ASCENT_PASSES):
            cfg = _coordinate_ascent_per_node(
                pipeline, contract_df, cfg,
                score_history=score_history, rng=rng,
            )
        score = _macro_f1(pipeline, contract_df, cfg)
        score_history.append((score, cfg.to_dict()))
        if score > best_overall_score:
            best_overall_score = score
            best_overall = cfg

    # Sweep ENTROPY_GATE x ABSENCE_VE_SCALE on top of best per-node config.
    best_with_feedback = best_overall
    best_with_feedback_score = best_overall_score
    for eg in ENTROPY_GATE_GRID:
        for vs in ABSENCE_VE_SCALE_GRID:
            trial = CalibConfig(
                thresholds=dict(best_overall.thresholds),
                alpha_map=dict(best_overall.alpha_map),
                entropy_gate=eg,
                absence_ve_scale=vs,
            )
            score = _macro_f1(pipeline, contract_df, trial)
            score_history.append((score, trial.to_dict()))
            if score > best_with_feedback_score:
                best_with_feedback_score = score
                best_with_feedback = trial

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(
        json.dumps(best_with_feedback.thresholds, indent=2), encoding="utf-8")
    (out / "alpha_map.json").write_text(
        json.dumps(best_with_feedback.alpha_map, indent=2), encoding="utf-8")
    (out / "feedback_config.json").write_text(
        json.dumps({
            "entropy_gate": best_with_feedback.entropy_gate,
            "absence_ve_scale": best_with_feedback.absence_ve_scale,
        }, indent=2), encoding="utf-8")
    (out / "calibration_topk.json").write_text(
        json.dumps(_topk_summary(score_history, k=5), indent=2), encoding="utf-8")

    return {
        "best_macro_f1": float(best_with_feedback_score),
        "thresholds": best_with_feedback.thresholds,
        "alpha_map": best_with_feedback.alpha_map,
        "entropy_gate": best_with_feedback.entropy_gate,
        "absence_ve_scale": best_with_feedback.absence_ve_scale,
    }


def calibrate_conflict_signal(
    *,
    val_csv_path: str = "data/processed/val.csv",
    bn_model_path: str = "results/phase3/bayesian_network.pkl",
    bert_checkpoint_path: str = "results/phase2/models/legal_bert_phase2.pt",
    label_map_path: str = "results/phase2/label2id.json",
    adapter_path: str | None = "results/phase2/models/legal_bert_lora_adapter",
    out_path: str = "results/phase3/conflict_calibration.json",
) -> dict:
    """Persist sorted val-set conflict signals for percentile (ECDF) calibration."""
    from src.phase3.hybrid_pipeline import AgastyaHybridPipeline

    val_df = pd.read_csv(val_csv_path)
    contract_df = build_contract_dataset(val_df)

    pipeline = AgastyaHybridPipeline(
        bn_model_path=ensure_seed_model(bn_model_path),
        bert_checkpoint_path=bert_checkpoint_path,
        label_map_path=label_map_path,
        adapter_path=adapter_path,
        load_calibration=False,
    )

    raw_signals: list[float] = []
    for text in contract_df["text"].tolist():
        signal = pipeline.compute_raw_conflict_signal(text)
        raw_signals.append(float(signal))
    raw_signals.sort()
    payload = {"method": "percentile", "sorted_conflict_signals": raw_signals}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    conflict_payload = calibrate_conflict_signal()
    logger.info("Persisted conflict calibration with %d signals.",
                len(conflict_payload["sorted_conflict_signals"]))
    result = calibrate()
    logger.info("Calibration complete. Best macro-F1=%.3f. Saved per-node + feedback configs.",
                result["best_macro_f1"])


if __name__ == "__main__":
    main()
