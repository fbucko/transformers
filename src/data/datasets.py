"""
datasets.py

This module defines PyTorch Datasets for text classification using
tokenized input text and integer labels. It includes a generic dataset
and aliases for specific data types like domain names, RDAP records, DNS, and GEO.

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextClassificationDataset(Dataset):
    """
    A PyTorch Dataset for any 2-column text+label DataFrame.

    Each __getitem__ returns a dict with:
      - 'input_ids'      : token IDs tensor
      - 'attention_mask' : attention mask tensor
      - 'label'          : the label tensor
    """
    def __init__(
        self,
        dataframe,                      # pandas DataFrame with [text_col, "label"]
        tokenizer: PreTrainedTokenizerBase,
        max_length: int
    ):
        self.texts      = dataframe.iloc[:, 0].astype(str).tolist()
        self.labels     = dataframe["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text  = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids"     : encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label"         : torch.tensor(label, dtype=torch.long),
        }

# also in src/data/datasets.py

class DomainDataset(TextClassificationDataset):
    """Alias for domain-name inputs."""
    pass

class RDAPDataset(TextClassificationDataset):
    """Alias for RDAP-record inputs."""
    pass

class DNSDataset(RDAPDataset):
    """DNS is the same shape as RDAP."""
    pass

class GEODataset(RDAPDataset):
    """GEO is the same shape as RDAP."""
    pass