"""Phase 2 Legal-BERT inference adapter for Phase 3 pipeline.

Loads either:
  (a) a LoRA adapter directory (preferred — base BERT + PEFT adapter), or
  (b) a merged plain checkpoint (legacy / DL_Only baseline).

Quantization: Dynamic INT8 (torch.quantization.quantize_dynamic) is applied
ONLY when LoRA is NOT active (PEFT-wrapped models are incompatible with
``torch.quantization.quantize_dynamic`` because state_dict layouts don't match).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.phase2.models.bert_classifier import BertWithLengthClassifier
from src.phase2.models.bert_lora_classifier import load_lora_adapter

# --- Expanded CUAD label → Phase 3 clause type mapping ---
# Covers 26 of 41 CUAD labels (up from 6).
# All risk-relevant categories now feed BN evidence nodes.
_PHASE2_TO_PHASE3 = {
    # Payment / Financial risk
    "revenue/profit sharing":               "Payment",
    "minimum commitment":                   "Payment",
    "price restrictions":                   "Payment",
    "liquidated damages":                   "Payment",
    "most favored nation":                  "Payment",
    "royalty":                              "Payment",

    # Termination risk
    "termination for convenience":          "Termination",
    "notice period to terminate renewal":   "Termination",
    "post-termination services":            "Termination",
    "renewal term":                         "Termination",

    # Liability / Indemnification risk
    "cap on liability":                     "Liability",
    "uncapped liability":                   "Liability",
    "warranty duration":                    "Liability",
    "insurance":                            "Liability",
    "indemnification":                      "Liability",

    # Confidentiality / IP / Non-compete risk
    "non-compete":                          "Confidentiality",
    "non-disparagement":                    "Confidentiality",
    "no-solicit of customers":              "Confidentiality",
    "no-solicit of employees":              "Confidentiality",
    "ip ownership assignment":              "Confidentiality",
    "joint ip ownership":                   "Confidentiality",
    "source code escrow":                   "Confidentiality",
    "non-transferable license":             "Confidentiality",
    "irrevocable or perpetual license":     "Confidentiality",

    # Dispute resolution / Governance
    "governing law":                        "Dispute Resolution",
    "covenant not to sue":                  "Dispute Resolution",
    "audit rights":                         "Dispute Resolution",
}

# Top-K predictions to emit per clause segment
_TOP_K = 3
# Minimum per-class probability to include in top-K output
_TOP_K_FLOOR = 0.08


logger = logging.getLogger(__name__)


class Phase2BertAdapter:
    """Loads a trained Phase 2 checkpoint and exposes .predict(clause_text).

    Returns a **list** of prediction dicts (top-K predictions), one per
    confident CUAD category. Callers must handle a list, not a single dict.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        label_map_path: str = "results/phase2/label2id.json",
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        max_length: int = 256,
        adapter_path: str | None = "results/phase2/models/legal_bert_lora_adapter",
    ):
        self.device = torch.device("cpu")
        self.max_length = max_length
        self.label2id = _load_label_map(label_map_path)
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        adapter_dir = Path(adapter_path) if adapter_path else None
        self.lora_active = bool(adapter_dir and adapter_dir.exists())

        base_model = BertWithLengthClassifier(
            model_name=model_name,
            num_classes=len(self.label2id),
            use_length_feature=True,
            download_pretrained_backbone=True,
        )

        if self.lora_active:
            self.model = load_lora_adapter(base_model, str(adapter_dir), device=self.device)
            logger.info("Phase2BertAdapter: loaded LoRA adapter from %s", adapter_dir)
        else:
            self.model = base_model.to(self.device)
            if checkpoint_path and Path(checkpoint_path).exists():
                state = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                logger.info("Phase2BertAdapter: loaded merged checkpoint from %s", checkpoint_path)
            self.model.eval()

        # ── Dynamic INT8 Quantization ─────────────────────────────────────────
        # PEFT-wrapped models are not compatible with torch.quantization
        # (state_dict layout differs); we only quantize the merged baseline.
        if self.lora_active:
            logger.info(
                "Phase2BertAdapter: skipping INT8 quantization because "
                "LoRA adapter is active (PEFT-incompatible)."
            )
        else:
            try:
                if "qnnpack" in torch.backends.quantized.supported_engines:
                    torch.backends.quantized.engine = "qnnpack"
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                logger.info(
                    "Legal-BERT quantized to INT8 (dynamic). Device: %s.",
                    self.device,
                )
            except RuntimeError as e:
                logger.warning(
                    "Quantization failed: %s. Running unquantized (FP32).", e,
                )

    def predict(self, clause_text: str) -> list[dict]:
        """Return top-K clause type predictions for a single clause segment.

        Each dict in the returned list has the same schema as the old single
        dict, so downstream consumers can iterate over them.
        """
        encoded = self.tokenizer(
            clause_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        length_feat = torch.tensor(
            [[float(np.log1p(input_ids.shape[1]))]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            bert_out = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = bert_out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                length_feat=length_feat,
            ).squeeze(0)
            probs = torch.softmax(logits, dim=0)

        # Top-K multi-label extraction
        k = min(_TOP_K, len(self.id2label))
        top_k = torch.topk(probs, k=k)
        results: list[dict] = []
        for idx, conf_t in zip(top_k.indices.tolist(), top_k.values.tolist()):
            conf = float(conf_t)
            if conf < _TOP_K_FLOOR:
                continue
            phase2_label = self.id2label[idx]
            clause_type = _map_phase2_to_phase3(phase2_label)
            results.append(
                {
                    "clause_text": clause_text,
                    "clause_type": clause_type,
                    "confidence": conf,
                    "embedding": cls_embedding,
                    "risk_indicators": [],
                    "logits": logits.cpu().numpy(),
                    "phase2_label": phase2_label,
                }
            )

        # Always return at least one result (the argmax prediction)
        if not results:
            top_idx = int(torch.argmax(probs).item())
            results.append(
                {
                    "clause_text": clause_text,
                    "clause_type": _map_phase2_to_phase3(self.id2label[top_idx]),
                    "confidence": float(probs[top_idx].item()),
                    "embedding": cls_embedding,
                    "risk_indicators": [],
                    "logits": logits.cpu().numpy(),
                    "phase2_label": self.id2label[top_idx],
                }
            )

        return results


def _load_label_map(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _map_phase2_to_phase3(label: str) -> str:
    mapped = _PHASE2_TO_PHASE3.get(label.lower())
    return mapped if mapped is not None else "Other"


def has_phase2_artifacts(
    checkpoint_path: str | None = None,
    label_map_path: str = "results/phase2/label2id.json",
    adapter_path: str | None = "results/phase2/models/legal_bert_lora_adapter",
) -> bool:
    """True iff label map exists AND (LoRA adapter dir exists OR merged ckpt exists)."""
    if not Path(label_map_path).exists():
        return False
    if adapter_path and Path(adapter_path).exists():
        return True
    if checkpoint_path and Path(checkpoint_path).exists():
        return True
    return False
