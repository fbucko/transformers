# hyperparameter_search_ray.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence tokenizer warnings

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


def main():
    # 0) Sanity check
    print(f"CUDA available: {torch.cuda.is_available()}  |  GPUs: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs detected.")

    # 1) Load & split
    df = pd.read_csv(
        # "../datasets/malware/malware_preprocessed.csv",
        "../datasets/dga/dga_preprocessed.csv",
        # "../datasets/phishing/phishing_preprocessed.csv",
        usecols=["domain_name", "label"],
        dtype={"domain_name": "string", "label": "int8"}
    )
    print(df.head())
    train_df, val_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=42
    )
    hf_train = Dataset.from_pandas(train_df).shuffle(seed=42)
    hf_val   = Dataset.from_pandas(val_df).shuffle(seed=42)

    # 2) Tokenize
    MODEL_NAME = "google/mobilebert-uncased"
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

    # 3) Metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")

    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=pred.label_ids)["accuracy"],
            "f1":       f1_metric.compute(predictions=preds, references=pred.label_ids)["f1"],
        }

    # 4) Model init
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )

    # 5) Ray Tune search space
    def hp_space_ray(_):
        return {
            "learning_rate":               tune.loguniform(1e-6, 5e-5),
            "per_device_train_batch_size": tune.choice([128, 256]),
            "weight_decay":                tune.uniform(0.0, 0.1),
            "warmup_ratio":                tune.uniform(0.0, 0.2),
            "lr_scheduler_type":           tune.choice(["linear", "cosine"]),
        }

    # 6) Training arguments (single‐GPU per trial)
    training_args = TrainingArguments(
        output_dir="hp_search_ray",
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=3,
        fp16=True,                  # mixed‐precision
        dataloader_num_workers=4,   # parallel data loading
        report_to="none",
    )

    # 7) Initialize Trainer
    trainer = Trainer(
        args=training_args,
        model_init=model_init,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 8) Hyperparameter search with Ray Tune
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=hp_space_ray,
        n_trials=20,
        resources_per_trial={"cpu": 4, "gpu": 2},  # one GPU per trial
    )

    # 9) Report best run (objective is F1)
    print("\nBest hyper-parameters found:")
    for k, v in best_run.hyperparameters.items():
        print(f" - {k}: {v}")
    print(f"Best validation F1: {best_run.objective:.4f}")


if __name__ == "__main__":
    main()
