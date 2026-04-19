"""PyTorch ``Dataset`` for processed contract CSV rows + tokenizer."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class ContractDataset(Dataset):
    """
    Maps rows from ``dataset_loader.load_split_csv`` (with ``text``, ``label_id``, ``log_length``)
    to tensors for :class:`~src.phase2.models.bert_classifier.BertWithLengthClassifier`.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        text_column: str = "text",
        label_column: str = "label_id",
        log_length_column: str = "log_length",
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.log_length_column = log_length_column
        for col in (text_column, label_column, log_length_column):
            if col not in self.df.columns:
                raise ValueError(f"Missing column {col!r}; columns are {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row[self.text_column],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        log_len = float(row[self.log_length_column])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row[self.label_column]), dtype=torch.long),
            "length_feat": torch.tensor(log_len, dtype=torch.float32).unsqueeze(0),
        }
