"""
logger.py

Utility for logging training and evaluation metrics to a CSV file.

Appends per-epoch metrics such as loss, accuracy, precision, recall, and F1-score.
Creates the file and headers if it does not already exist.

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""
import csv
from pathlib import Path
from datetime import datetime
from typing import Union, Mapping, Any


def log_metrics(
    run_id: Union[str, int],
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    val_precision: float,
    val_recall: float,
    val_f1: float,
    log_file: Union[str, Path] = "training_logs.csv"
) -> None:
    """
    Append or create a CSV log file with training and validation metrics per epoch.

    Args:
        run_id: Identifier for the training run.
        epoch: Current epoch number (1-indexed).
        train_loss: Training loss for this epoch.
        val_loss: Validation loss for this epoch.
        val_acc: Validation accuracy.
        val_precision: Validation precision.
        val_recall: Validation recall.
        val_f1: Validation F1-score.
        log_file: Path to the CSV log file.
    """
    # Ensure the log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Define the header/order of columns
    fieldnames = [
        "run_id",
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "timestamp"
    ]

    # Check if we need to write the header
    write_header = not log_path.exists() or log_path.stat().st_size == 0

    # Prepare a dictionary of values
    row: Mapping[str, Any] = {
        "run_id": run_id,
        "epoch": epoch,
        "train_loss": f"{train_loss:.6f}",
        "val_loss":   f"{val_loss:.6f}",
        "val_acc":    f"{val_acc:.4f}",
        "val_precision": f"{val_precision:.4f}",
        "val_recall":    f"{val_recall:.4f}",
        "val_f1":        f"{val_f1:.4f}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Append the row to the CSV file
    with log_path.open(mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
