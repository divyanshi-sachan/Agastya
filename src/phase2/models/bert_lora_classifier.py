"""LoRA helpers for ``BertWithLengthClassifier``.

Wraps the underlying HuggingFace BERT encoder with LoRA adapters via the PEFT
library so that Phase 2 fine-tuning trains only a small set of low-rank
matrices plus the classifier head, while the base BERT weights remain frozen.

Usage:
    from src.phase2.models.bert_classifier import BertWithLengthClassifier
    from src.phase2.models.bert_lora_classifier import apply_lora, save_lora_adapter

    model = BertWithLengthClassifier(MODEL_NAME, num_classes=41, dropout=0.1)
    model = apply_lora(model, r=8, alpha=16, dropout=0.1)
    # ... train ...
    save_lora_adapter(model, "results/phase2/models/legal_bert_lora_adapter")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "peft>=0.10.0 is required for LoRA fine-tuning. Install via "
        "`pip install peft`."
    ) from exc


_DEFAULT_TARGET_MODULES: tuple[str, ...] = ("query", "key", "value", "dense")


def _freeze_base(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_classifier_head(model: nn.Module) -> None:
    """Always train the classification MLP head on top of the LoRA backbone."""
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Iterable[str] = _DEFAULT_TARGET_MODULES,
) -> nn.Module:
    """Wrap ``model.bert`` with LoRA adapters in-place.

    Freezes the underlying BERT encoder weights, attaches LoRA adapters to the
    attention projection + output dense layers, and keeps the classifier head
    trainable. The wrapping mutates ``model`` (replaces ``model.bert`` with a
    ``PeftModel``) and also returns it for convenience.
    """
    if not hasattr(model, "bert"):
        raise AttributeError(
            "apply_lora expects an attribute `bert` on the supplied model "
            "(BertWithLengthClassifier-style)."
        )

    _freeze_base(model)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=None,  # generic feature-extraction usage
        target_modules=list(target_modules),
    )
    model.bert = get_peft_model(model.bert, lora_cfg)

    _unfreeze_classifier_head(model)
    return model


def save_lora_adapter(model: nn.Module, adapter_dir: str | Path) -> Path:
    """Persist the LoRA adapter (and classifier head) to disk.

    Saves:
      - ``adapter_dir/`` — PEFT adapter (lora weights + config).
      - ``adapter_dir/classifier_head.pt`` — classifier head state dict.
    """
    adapter_path = Path(adapter_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    if not isinstance(model.bert, PeftModel):
        raise RuntimeError(
            "Cannot save LoRA adapter: model.bert is not a PeftModel. "
            "Did you forget to call apply_lora(model, ...)?"
        )
    model.bert.save_pretrained(str(adapter_path))

    if hasattr(model, "classifier"):
        torch.save(model.classifier.state_dict(), adapter_path / "classifier_head.pt")

    return adapter_path


def load_lora_adapter(
    base_model: nn.Module,
    adapter_dir: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Attach a saved LoRA adapter (and classifier head) onto a freshly built base model.

    The base model must be a ``BertWithLengthClassifier`` (or compatible) with
    the same ``model_name`` / ``num_classes`` used during training.
    """
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")

    base_model.bert = PeftModel.from_pretrained(base_model.bert, str(adapter_path))

    head_path = adapter_path / "classifier_head.pt"
    if head_path.exists() and hasattr(base_model, "classifier"):
        head_state = torch.load(head_path, map_location=device)
        base_model.classifier.load_state_dict(head_state, strict=False)

    base_model.to(device)
    base_model.eval()
    return base_model


def merge_lora_into_base(model: nn.Module) -> nn.Module:
    """Merge LoRA deltas back into the base BERT weights for legacy/ablation use.

    After this call, ``model.bert`` is a plain ``transformers`` model (no PEFT
    wrapper), suitable for INT8 quantization or saving via ``state_dict``.
    """
    if isinstance(model.bert, PeftModel):
        model.bert = model.bert.merge_and_unload()
    return model


__all__ = [
    "apply_lora",
    "save_lora_adapter",
    "load_lora_adapter",
    "merge_lora_into_base",
]
