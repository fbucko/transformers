"""
dataset_loader.py

This module provides the DatasetLoader class for loading and preprocessing
CSV-based datasets used in text classification tasks.

Supported data types:
- 'domain': expects 'domain_name' and 'label' columns
- 'rdap' / 'dns' / 'geo': expect 'input_string' and 'label' columns

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""
import pandas as pd

class DatasetLoader:
    """
    Loads text classification data from a CSV file.

    Supports three types of data:
      - "domain": uses columns "domain_name" and "label"
      - "rdap":   uses columns "input_string" and "label"
      - "dns":    same as "rdap"
    """
    def __init__(self, file_path: str, data_type: str = "domain"):
        self.file_path = file_path
        self.data_type = data_type

        if data_type == "domain":
            self.text_col = "domain_name"
        elif data_type in ("rdap", "dns", "geo"):
            self.text_col = "input_string"
        else:
            raise ValueError(
                f"Unknown data_type: {data_type}. Must be 'domain', 'rdap', 'dns' or 'geo'."
            )
        self.label_col = "label"

    def load(self) -> pd.DataFrame:
        """
        Reads the CSV and returns a DataFrame with exactly two columns:
        the chosen text column and the label column.
        """
        df = pd.read_csv(
            self.file_path,
            usecols=[self.text_col, self.label_col],
            dtype={self.text_col: "string", self.label_col: "int8"}
        )
        return df.reset_index(drop=True)
