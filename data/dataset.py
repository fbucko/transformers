# data/dataset.py
import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class DatasetLoader:
    """
    Loads a dataset from a CSV file.
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.dtype = {"domain_name": "string", "label": "int8"}
        self.usecols = ["domain_name", "label"]

    def load(self) -> Tuple[pd.Series, pd.Series]:
        """
        Reads the CSV file and returns the domains and labels.
        """
        df = pd.read_csv(self.file_path, usecols=self.usecols, dtype=self.dtype)
        df = df.reset_index(drop=True)
        return df["domain_name"], df["label"]

class DomainDataset(Dataset):
    """
    PyTorch Dataset for domain data.
    """
    def __init__(self, domains, labels, tokenizer: DistilBertTokenizer, max_length: int):
        self.domains = domains
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx: int):
        domain = str(self.domains.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            domain,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
