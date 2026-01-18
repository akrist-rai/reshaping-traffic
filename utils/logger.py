import csv
from pathlib import Path

class CSVLogger:
    def __init__(self, path="training_metrics.csv"):
        self.path = Path(path)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_mae"])

    def log(self, epoch, train_loss, val_mae):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_mae])
