# data/dataset.py
import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

import pandas as pd
from typing import Tuple

class DatasetLoader:
    """
    Loads a dataset from a CSV file. The loader adapts based on the data type.
    """
    def __init__(self, file_path: str, data_type: str = "domain") -> None:
        self.file_path = file_path
        self.data_type = data_type
        if self.data_type == "domain":
            self.dtype = {"domain_name": "string", "label": "int8"}
            self.usecols = ["domain_name", "label"]
        elif self.data_type == "rdap" or self.data_type == "dns":
            self.dtype = {"input_string": "string", "label": "int8"}
            self.usecols = ["input_string", "label"]
        else:
            raise ValueError("Unknown data_type. Please use 'domain' or 'rdap'.")

    def load(self) -> pd.DataFrame:
        """
        Reads the CSV file and returns the DataFrame.
        """
        df = pd.read_csv(self.file_path, usecols=self.usecols, dtype=self.dtype)
        return df.reset_index(drop=True)


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

class RDAPDataset(Dataset):
    """
    PyTorch Dataset for RDAP data.
    """
    def __init__(self, df, tokenizer: DistilBertTokenizer, max_length: int):
        self.inputs = df["input_string"]
        self.labels = df["label"]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        text = str(self.inputs.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
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
        
class DNSDataset(RDAPDataset):
    """
    PyTorch Dataset for DNS data.
    """
    pass