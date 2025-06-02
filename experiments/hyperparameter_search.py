#!/usr/bin/env python
"""
hyperparameter_search_ray.py

Performs hyperparameter optimization using HuggingFace Transformers with Ray Tune backend.

NOTICE: This script is designed to be run on a GPU machine
        and within the environment specified in the project setup.

This script:
- Loads a CSV dataset (e.g., DGA, malware, or phishing domains)
- Tokenizes the data using a HuggingFace tokenizer
- Uses HuggingFace's Trainer with Ray Tune for hyperparameter search
- Evaluates models using accuracy and F1-score

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence tokenizer warnings
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from ray import tune

# Dynamically add the project root to PYTHONPATH
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

def main():
    # 1. Setup check
    print(f"CUDA available: {torch.cuda.is_available()}  |  GPUs: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs detected.")

    # 2. Load data
    data_path = (PROJECT_ROOT /".."/ "datasets" / "dga" / "dga_preprocessed.csv").resolve()
    # data_path = (PROJECT_ROOT / "datasets" / "malware" / "malware_preprocessed.csv").resolve()
    # data_path = (PROJECT_ROOT / "datasets" / "phishing" / "phishing_preprocessed.csv").resolve()
    df = pd.read_csv(
        data_path,
        usecols=["domain_name", "label"],
        dtype={"domain_name": "string", "label": "int8"}
    )
    print(df.head())
    train_df, val_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=42
    )
    hf_train = Dataset.from_pandas(train_df).shuffle(seed=42)
    hf_val   = Dataset.from_pandas(val_df).shuffle(seed=42)

    # 3. Tokenize
    # MODEL_NAME = "prajjwal1/bert-tiny"
    MODEL_NAME = "prajjwal1/bert-medium"
    # MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        return tokenizer(
            batch["domain_name"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    hf_train = hf_train.map(preprocess, batched=True)
    hf_val   = hf_val.map(preprocess, batched=True)

    hf_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    hf_val.set_format("torch",   columns=["input_ids", "attention_mask", "label"])

    # 4. Metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")

    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=pred.label_ids)["accuracy"],
            "f1":       f1_metric.compute(predictions=preds, references=pred.label_ids)["f1"],
        }

    # 5. Model initialization
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )

    # 6. Ray Tune search space
    def hp_space_ray(_):
        return {
            "learning_rate":               tune.loguniform(1e-6, 5e-5),
            "per_device_train_batch_size": tune.choice([128, 256]),
            "weight_decay":                tune.uniform(0.0, 0.1),
            "warmup_ratio":                tune.uniform(0.0, 0.2),
            "lr_scheduler_type":           tune.choice(["linear", "cosine"]),
        }

    # 7. Training arguments (single‐GPU per trial)
    training_args = TrainingArguments(
        output_dir=".hp_search_ray",
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=3,
        fp16=True,                  # mixed‐precision
        dataloader_num_workers=4,   # parallel data loading
        report_to="none",
    )

    # 8. Initialize Trainer
    trainer = Trainer(
        args=training_args,
        model_init=model_init,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 9. Hyperparameter search with Ray Tune
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=hp_space_ray,
        n_trials=20,
        resources_per_trial={"cpu": 4, "gpu": 2},  # one GPU per trial
    )

    # 10. Report best run (objective is F1)
    print("\nBest hyper-parameters found:")
    for k, v in best_run.hyperparameters.items():
        print(f" - {k}: {v}")
    print(f"Best validation F1: {best_run.objective:.4f}")


if __name__ == "__main__":
    main()
