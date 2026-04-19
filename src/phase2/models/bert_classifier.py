"""BERT encoder with [CLS] + optional log-length fusion and MLP classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertWithLengthClassifier(nn.Module):
    """
    BERT-based classifier with optional length feature fusion.

    Inputs:
        input_ids: (B, L)
        attention_mask: (B, L)
        length_feat: (B, 1) — use log-scaled length (e.g. CSV ``log_length``), float32

    Output:
        logits: (B, num_classes)

    Keyword Args:
        download_pretrained_backbone: If False, build the transformer with random init from
            ``AutoConfig`` only (small Hub fetch), then load weights via ``load_state_dict`` —
            use when restoring a full checkpoint (e.g. Colab) to avoid a duplicate full-model download.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = 0.1,
        use_length_feature: bool = True,
        *,
        download_pretrained_backbone: bool = True,
    ):
        super().__init__()

        if download_pretrained_backbone:
            self.bert = AutoModel.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
            )
        else:
            # Only fetches small config.json from Hub; weights come from load_state_dict (Colab checkpoint).
            cfg = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModel.from_config(cfg)
        hidden_size = self.bert.config.hidden_size

        self.use_length_feature = use_length_feature

        classifier_input_dim = hidden_size + 1 if use_length_feature else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        length_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0]

        if self.use_length_feature:
            if length_feat is None:
                raise ValueError("length_feat required when use_length_feature=True")
            if length_feat.dim() == 1:
                length_feat = length_feat.unsqueeze(1)
            length_feat = length_feat.to(dtype=cls_embedding.dtype, device=cls_embedding.device)
            x = torch.cat([cls_embedding, length_feat], dim=1)
        else:
            x = cls_embedding

        return self.classifier(x)
