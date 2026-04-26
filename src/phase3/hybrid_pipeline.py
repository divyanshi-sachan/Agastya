"""End-to-end orchestration for the Phase 3 hybrid pipeline.

Pipeline:
  Contract text
    -> Clause segmentation
    -> LoRA Legal-BERT (top-K predictions per segment, with embeddings)
    -> Encoder (damped log-odds aggregation, per-node alpha, calibrated
       thresholds, scaled absence penalties, virtual evidence injection)
    -> Conflict signal (cosine of Payment vs Termination embeddings),
       percentile-calibrated against val-set ECDF, injected as virtual CPD
       on Cross_Clause_Conflict
    -> BN inference (state-aware TabularCPD virtual evidence)
    -> Bidirectional entropy-gated feedback loop (boost on high, dampen on low)
       with bounded updates, oscillation/delta abort, max 2 iterations
    -> Smart Report (risk + clause + conflict + top-3 prior-corrected factors)
    -> Distribution-shift guard (KS distance vs val ECDF)

risk_bert is not invoked here — it is only used by the DL_Only ablation.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from pgmpy.inference import BeliefPropagation

from src.phase2.segmentation.clause_splitter import split_clauses
from src.phase3.bayesian.em_trainer import load_model
from src.phase3.bayesian.inference import query_node_posterior, run_inference
from src.phase3.interface.evidence_encoder import (
    CLAUSE_MAP,
    encode_evidence,
    load_absence_ve_scale,
    load_alpha_map,
    load_thresholds,
)
from src.phase3.interface.feature_extractor import extract_cross_clause_features
from src.phase3.interface.phase2_adapter import Phase2BertAdapter, has_phase2_artifacts
from src.phase3.smart_report import build_smart_report

logger = logging.getLogger(__name__)


# --- Feedback-loop knobs ------------------------------------------------------

ENTROPY_GATE_DEFAULT = 0.85
MAX_ITERATIONS = 2
CONVERGENCE_DELTA = 0.01
MAX_POSTERIOR_DELTA = 0.30
OSCILLATION_THRESHOLD = 0.10
BOOST_FACTOR = 1.10
DAMP_FACTOR = 1.10

_FEEDBACK_NODES = list(CLAUSE_MAP.values())


# --- Conflict-signal knobs ----------------------------------------------------

_CONFLICT_CALIB_PATH = Path("results/phase3/conflict_calibration.json")
_CONFLICT_RUNTIME_LOG = Path("results/phase3/conflict_runtime_log.jsonl")
_CONFLICT_RUNTIME_CAP = 5000
_KS_WARN_THRESHOLD = 0.2


def boost_prob(p: float, factor: float = BOOST_FACTOR) -> float:
    """Pull p toward 1, bounded at 0.99."""
    return float(min(0.99, p + (1.0 - p) * (factor - 1.0)))


def damp_prob(p: float, factor: float = DAMP_FACTOR) -> float:
    """Pull p toward 0, bounded at 0.01."""
    return float(max(0.01, p - p * (factor - 1.0)))


def _entropy(probs: dict[str, float]) -> float:
    total = 0.0
    for p in probs.values():
        if p <= 0.0:
            continue
        total -= p * math.log(p + 1e-12)
    return float(total)


def _load_feedback_config() -> dict[str, float]:
    path = Path("results/phase3/feedback_config.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_conflict_calibration() -> list[float]:
    if not _CONFLICT_CALIB_PATH.exists():
        return []
    try:
        payload = json.loads(_CONFLICT_CALIB_PATH.read_text(encoding="utf-8"))
        return list(payload.get("sorted_conflict_signals", []))
    except (json.JSONDecodeError, OSError):
        return []


def _calibrate_conflict(c: float, sorted_vals: list[float]) -> float:
    if not sorted_vals:
        return float(c)
    idx = bisect.bisect_right(sorted_vals, float(c))
    return float(idx) / float(max(len(sorted_vals), 1))


def _ks_distance(sample: list[float], reference: list[float]) -> float:
    """One-sample KS distance between two samples (monotone empirical CDFs)."""
    if not sample or not reference:
        return 0.0
    a = sorted(float(x) for x in sample)
    b = sorted(float(x) for x in reference)
    n_a, n_b = len(a), len(b)
    i = j = 0
    cdf_a = cdf_b = 0.0
    diff = 0.0
    while i < n_a and j < n_b:
        if a[i] <= b[j]:
            i += 1
            cdf_a = i / n_a
        else:
            j += 1
            cdf_b = j / n_b
        diff = max(diff, abs(cdf_a - cdf_b))
    return float(diff)


def _append_runtime_conflict(c: float) -> list[float]:
    """Append c to rolling runtime log (capped)."""
    _CONFLICT_RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
    rolling: list[float] = []
    if _CONFLICT_RUNTIME_LOG.exists():
        try:
            for line in _CONFLICT_RUNTIME_LOG.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rolling.append(float(json.loads(line).get("c", 0.0)))
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            rolling = []
    rolling.append(float(c))
    rolling = rolling[-_CONFLICT_RUNTIME_CAP:]
    try:
        _CONFLICT_RUNTIME_LOG.write_text(
            "\n".join(json.dumps({"c": float(v)}) for v in rolling) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    return rolling


# --- Dummy fallback predictor ------------------------------------------------

@dataclass
class DummyBertPredictor:
    """Lightweight placeholder until a full Phase 2 inference wrapper is added."""

    def predict(self, clause_text: str) -> list[dict]:
        lowered = clause_text.lower()
        if "payment" in lowered or "invoice" in lowered or "price" in lowered:
            clause_type, confidence = "Payment", 0.45
        elif "terminat" in lowered or "notice" in lowered:
            clause_type, confidence = "Termination", 0.42
        elif "liabil" in lowered or "indemnif" in lowered or "warrant" in lowered:
            clause_type, confidence = "Liability", 0.40
        elif "confidential" in lowered or "non-disclosure" in lowered or "non-compete" in lowered:
            clause_type, confidence = "Confidentiality", 0.40
        elif "dispute" in lowered or "governing law" in lowered or "arbitrat" in lowered:
            clause_type, confidence = "Dispute Resolution", 0.40
        else:
            clause_type, confidence = "Other", 0.05
        return [
            {
                "clause_text": clause_text,
                "clause_type": clause_type,
                "confidence": confidence,
                "embedding": np.zeros(768, dtype=float),
                "risk_indicators": [],
                "logits": np.zeros(41, dtype=float),
                "phase2_label": clause_type,
            }
        ]


def _flatten_bert_outputs(raw: list) -> list[dict]:
    flat: list[dict] = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, dict):
            flat.append(item)
    return flat


# --- Pipeline -----------------------------------------------------------------

class AgastyaHybridPipeline:
    """Orchestrates LoRA Legal-BERT + BN with bidirectional feedback loop."""

    def __init__(
        self,
        bn_model_path: str,
        bert_predictor: Callable | None = None,
        bert_checkpoint_path: str | None = None,
        label_map_path: str = "results/phase2/label2id.json",
        adapter_path: str | None = "results/phase2/models/legal_bert_lora_adapter",
        *,
        load_calibration: bool = True,
    ):
        self.bert = bert_predictor or self._build_predictor(
            bert_checkpoint_path=bert_checkpoint_path,
            label_map_path=label_map_path,
            adapter_path=adapter_path,
        )
        self.bn = load_model(bn_model_path)
        self.bp_engine = BeliefPropagation(self.bn)

        # Calibration state.
        if load_calibration:
            self.thresholds = load_thresholds()
            self.alpha_map = load_alpha_map()
            self.absence_ve_scale = load_absence_ve_scale()
        else:
            from src.phase3.interface.evidence_encoder import (
                ALPHA_DEFAULTS, ABSENCE_VE_SCALE_DEFAULT, THRESHOLD_DEFAULTS,
            )
            self.thresholds = dict(THRESHOLD_DEFAULTS)
            self.alpha_map = dict(ALPHA_DEFAULTS)
            self.absence_ve_scale = ABSENCE_VE_SCALE_DEFAULT

        feedback_cfg = _load_feedback_config() if load_calibration else {}
        self.entropy_gate = float(feedback_cfg.get("entropy_gate", ENTROPY_GATE_DEFAULT))

        self.sorted_conflict_calib = _load_conflict_calibration() if load_calibration else []

        self._priors = self._compute_priors_once()

    def set_runtime_overrides(
        self,
        thresholds: dict[str, float] | None = None,
        alpha_map: dict[str, float] | None = None,
        entropy_gate: float | None = None,
        absence_ve_scale: float | None = None,
    ) -> None:
        if thresholds is not None:
            self.thresholds = dict(thresholds)
        if alpha_map is not None:
            self.alpha_map = dict(alpha_map)
        if entropy_gate is not None:
            self.entropy_gate = float(entropy_gate)
        if absence_ve_scale is not None:
            self.absence_ve_scale = float(absence_ve_scale)

    def _build_predictor(
        self,
        bert_checkpoint_path: str | None,
        label_map_path: str,
        adapter_path: str | None,
    ):
        if has_phase2_artifacts(bert_checkpoint_path, label_map_path, adapter_path):
            try:
                return Phase2BertAdapter(
                    checkpoint_path=bert_checkpoint_path,
                    label_map_path=label_map_path,
                    adapter_path=adapter_path,
                )
            except Exception as exc:
                logger.warning("Phase2BertAdapter init failed (%s); falling back to dummy.", exc)
                return DummyBertPredictor()
        logger.info("Phase 2 artifacts missing; using DummyBertPredictor.")
        return DummyBertPredictor()

    # ---- Priors (cached) -----------------------------------------------------

    def _compute_priors_once(self) -> dict[str, dict[str, float]]:
        """Single no-evidence query per node — reused for top_risk_factors."""
        priors: dict[str, dict[str, float]] = {}
        nodes = list(CLAUSE_MAP.values()) + [
            "Payment_Or_Termination_Risky",
            "Liability_Or_Confidentiality_Risky",
            "Cross_Clause_Conflict",
        ]
        for node in nodes:
            try:
                priors[node] = query_node_posterior(self.bn, node)
            except Exception as exc:
                logger.debug("Failed to compute prior for %s: %s", node, exc)
        return priors

    # ---- Conflict signal -----------------------------------------------------

    def _embeddings_by_type(self, bert_outputs: list[dict]) -> dict[str, np.ndarray]:
        return {
            output["clause_type"]: output["embedding"]
            for output in bert_outputs
            if output.get("clause_type") in {"Payment", "Termination"}
            and output.get("embedding") is not None
        }

    def compute_raw_conflict_signal(self, contract_text: str) -> float:
        """Used during calibration — returns raw (uncalibrated) conflict signal."""
        clauses = split_clauses(contract_text)
        raw_outputs = [self.bert.predict(clause) for clause in clauses]
        bert_outputs = _flatten_bert_outputs(raw_outputs)
        embeddings = self._embeddings_by_type(bert_outputs)
        return float(extract_cross_clause_features(embeddings))

    # ---- Feedback loop -------------------------------------------------------

    def _apply_direction(
        self,
        virtual_evidence: dict[str, list[float]],
        direction: str,
    ) -> dict[str, list[float]]:
        updated: dict[str, list[float]] = {}
        for node, probs in virtual_evidence.items():
            if node not in _FEEDBACK_NODES:
                updated[node] = list(probs)
                continue
            p_absent, p_present = float(probs[0]), float(probs[1])
            if direction == "escalate":
                p_present = boost_prob(p_present)
            elif direction == "de-escalate":
                p_present = damp_prob(p_present)
            p_absent = 1.0 - p_present
            updated[node] = [p_absent, p_present]
        return updated

    def _decide_direction(self, probabilities: dict[str, float]) -> str | None:
        ent = _entropy(probabilities)
        high_p = float(probabilities.get("High", 0.0))
        low_p = float(probabilities.get("Low", 0.0))
        if ent >= self.entropy_gate:
            return None
        if high_p > 0.7:
            return "escalate"
        if low_p > 0.7:
            return "de-escalate"
        return None

    def _bidirectional_feedback(
        self,
        hard_evidence: dict[str, str],
        virtual_evidence: dict[str, list[float]],
        bn_result: dict,
    ) -> tuple[dict, list[dict]]:
        """Apply guarded, bounded, bidirectional feedback iterations."""
        trace: list[dict] = []
        current_probs = dict(bn_result["probabilities"])
        current_virtual = {k: list(v) for k, v in virtual_evidence.items()}
        prev_direction: str | None = None
        last_good_result = bn_result

        for iteration in range(MAX_ITERATIONS):
            direction = self._decide_direction(current_probs)
            if direction is None:
                trace.append({
                    "iteration": iteration,
                    "applied": False,
                    "reason": "entropy_gate_or_no_dominant_class",
                    "entropy": _entropy(current_probs),
                    "probabilities": dict(current_probs),
                })
                break

            new_virtual = self._apply_direction(current_virtual, direction)
            new_result = run_inference(
                self.bn,
                hard_evidence,
                virtual_evidence=new_virtual,
                bp_engine=self.bp_engine,
            )
            new_probs = new_result["probabilities"]

            delta = max(abs(new_probs.get(c, 0.0) - current_probs.get(c, 0.0))
                        for c in {"Low", "Medium", "High"})
            if delta > MAX_POSTERIOR_DELTA:
                trace.append({
                    "iteration": iteration,
                    "applied": False,
                    "direction": direction,
                    "reason": "delta_exceeded",
                    "delta": delta,
                    "rolled_back_to": dict(current_probs),
                })
                break
            if (prev_direction is not None
                    and prev_direction != direction
                    and delta > OSCILLATION_THRESHOLD):
                trace.append({
                    "iteration": iteration,
                    "applied": False,
                    "direction": direction,
                    "reason": "oscillation_detected",
                    "delta": delta,
                    "rolled_back_to": dict(current_probs),
                })
                break

            trace.append({
                "iteration": iteration,
                "applied": True,
                "direction": direction,
                "delta": delta,
                "probabilities": dict(new_probs),
            })

            converged = abs(new_probs.get("High", 0.0) - current_probs.get("High", 0.0)) < CONVERGENCE_DELTA
            current_probs = new_probs
            current_virtual = new_virtual
            last_good_result = new_result
            prev_direction = direction
            if converged:
                trace[-1]["converged"] = True
                break

        return last_good_result, trace

    # ---- Main entry ----------------------------------------------------------

    def predict(self, contract_text: str) -> dict:
        clauses = split_clauses(contract_text)
        raw_outputs = [self.bert.predict(clause) for clause in clauses]
        bert_outputs = _flatten_bert_outputs(raw_outputs)

        encoder_payload = encode_evidence(
            bert_outputs,
            thresholds=self.thresholds,
            alpha_map=self.alpha_map,
            absence_ve_scale=self.absence_ve_scale,
        )
        hard_evidence = encoder_payload["hard_evidence"]
        virtual_evidence: dict[str, list[float]] = {
            **encoder_payload["virtual_evidence"]
        }

        embeddings = self._embeddings_by_type(bert_outputs)
        raw_conflict = float(extract_cross_clause_features(embeddings))

        rolling = _append_runtime_conflict(raw_conflict)
        ks = _ks_distance(rolling, self.sorted_conflict_calib)
        calibration_warning: str | None = None
        if self.sorted_conflict_calib and ks > _KS_WARN_THRESHOLD:
            calibration_warning = (
                "Conflict-signal distribution shift detected "
                f"(KS={ks:.2f} > {_KS_WARN_THRESHOLD}). "
                "Recalibrate against current data."
            )
            logger.warning(calibration_warning)

        calibrated_conflict = _calibrate_conflict(raw_conflict, self.sorted_conflict_calib) \
            if self.sorted_conflict_calib else None
        if calibrated_conflict is not None:
            virtual_evidence["Cross_Clause_Conflict"] = [
                1.0 - calibrated_conflict,
                calibrated_conflict,
            ]

        bn_result = run_inference(
            self.bn,
            hard_evidence,
            virtual_evidence=virtual_evidence,
            bp_engine=self.bp_engine,
        )

        bn_result, iteration_trace = self._bidirectional_feedback(
            hard_evidence=hard_evidence,
            virtual_evidence=virtual_evidence,
            bn_result=bn_result,
        )

        cross_post = self._query_node_post("Cross_Clause_Conflict",
                                            hard_evidence, virtual_evidence)
        node_posteriors = self._collect_node_posteriors(hard_evidence, virtual_evidence)

        smart_report = build_smart_report(
            risk_level=bn_result["risk_level"],
            probabilities=bn_result["probabilities"],
            bert_outputs=bert_outputs,
            encoder_payload=encoder_payload,
            raw_conflict=raw_conflict,
            calibrated_conflict=calibrated_conflict,
            cross_clause_posterior=cross_post,
            iteration_trace=iteration_trace,
            calibration_warning=calibration_warning,
            node_posteriors=node_posteriors,
            node_priors=self._priors,
        )

        return {
            "risk_level": bn_result["risk_level"],
            "risk_probabilities": bn_result["probabilities"],
            "clause_evidence": hard_evidence,
            "initial_clause_evidence": dict(hard_evidence),
            "conflict_signal": raw_conflict,
            "calibrated_conflict_signal": calibrated_conflict,
            "bert_details": bert_outputs,
            "bn_trace": str(bn_result["distribution"]),
            "virtual_evidence": virtual_evidence,
            "feedback_loop": {
                "iteration_trace": iteration_trace,
                "applied": any(step.get("applied") for step in iteration_trace),
            },
            "encoder_payload": encoder_payload,
            "smart_report": smart_report,
            "calibration_warning": calibration_warning,
        }

    # ---- Internal helpers ----------------------------------------------------

    def _query_node_post(
        self,
        node: str,
        hard_evidence: dict[str, str],
        virtual_evidence: dict[str, list[float]],
    ) -> dict[str, float]:
        try:
            res = run_inference(
                self.bn,
                hard_evidence,
                virtual_evidence=virtual_evidence,
                query_var=node,
                bp_engine=self.bp_engine,
            )
            return res["probabilities"]
        except Exception:
            return {}

    def _collect_node_posteriors(
        self,
        hard_evidence: dict[str, str],
        virtual_evidence: dict[str, list[float]],
    ) -> dict[str, dict[str, float]]:
        posteriors: dict[str, dict[str, float]] = {}
        nodes = list(CLAUSE_MAP.values()) + [
            "Payment_Or_Termination_Risky",
            "Liability_Or_Confidentiality_Risky",
            "Cross_Clause_Conflict",
        ]
        for node in nodes:
            posteriors[node] = self._query_node_post(node, hard_evidence, virtual_evidence)
        return posteriors
