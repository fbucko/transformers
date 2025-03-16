# utils/logger.py
import os
import csv
from datetime import datetime

def log_metrics(run_id, epoch, train_loss, val_acc, val_precision, val_recall, val_f1, log_file="training_logs.csv"):
    """
    Appends training metrics for each epoch to a CSV file.
    """
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run_id", "epoch", "train_loss", "val_acc", "val_precision", "val_recall", "val_f1", "timestamp"])
        writer.writerow([
            run_id, epoch, train_loss, val_acc, val_precision, val_recall, val_f1,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
