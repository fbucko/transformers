# run.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from config import Config
from data.dataset import DatasetLoader, DomainDataset
from models.transformer_model import get_model
from training.trainer import train

def main():
    # 1) Load
    loader = DatasetLoader(Config.DATA_PATH)
    domains, labels = loader.load()

    # 2) Split: train / val / test (hold out test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(domains)),
        test_size=0.4,
        random_state=42,
        stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=labels.iloc[temp_idx]
    )

    # Save splits for reproducibility
    np.savez_compressed(
        "splits.npz",
        train=train_idx, val=val_idx, test=test_idx
    )

    # 3) Build datasets
    config = Config()
    for model_name in Config.MODEL_NAMES:
        config.PRETRAINED_MODEL_NAME = model_name
        config.MODEL_SAVE_NAME = model_name.replace('/', '-')

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_ds = DomainDataset(
            domains.iloc[train_idx],
            labels.iloc[train_idx],
            tokenizer,
            config.MAX_LENGTH
        )
        val_ds = DomainDataset(
            domains.iloc[val_idx],
            labels.iloc[val_idx],
            tokenizer,
            config.MAX_LENGTH
        )

        train(config, train_ds, val_ds, tokenizer, get_model)

if __name__ == "__main__":
    main()
