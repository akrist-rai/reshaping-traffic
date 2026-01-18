import pandas as pd
import matplotlib.pyplot as plt

def plot_training(csv_path="training_metrics.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.show()
