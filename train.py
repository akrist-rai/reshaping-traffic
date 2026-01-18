import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from models.st_mamba import NewtonGraphMamba
from datasets.traffic_dataset import TrafficDataset
from utils.metrics import masked_mae
from utils.seed import set_seed
from utils.early_stopping import EarlyStopping
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import CSVLogger
logger = CSVLogger()



# =======================
# LOGGING
# =======================
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)


def auto_accumulation(device_mem_gb):
    if device_mem_gb <= 4:
        return 4, 8     # effective batch = 32
    elif device_mem_gb <= 6:
        return 8, 4
    else:
        return 8, 8


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================
    # DATA
    # =======================
    data = np.load("data/metr_la/metr_la.npz")["data"]
    adj = torch.tensor(np.load("data/metr_la/adj.npy")).float().to(device)

    T = len(data)
    train, val = data[:int(0.7 * T)], data[int(0.7 * T):int(0.8 * T)]

    train_ds = TrafficDataset(train)
    val_ds = TrafficDataset(val, mean=train_ds.mean, std=train_ds.std)

    batch_size, grad_accum_steps = auto_accumulation(4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=True
    )

    # =======================
    # MODEL
    # =======================
    model = NewtonGraphMamba(
        in_features=data.shape[-1]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # =======================
    # CHECKPOINT + EARLY STOP
    # =======================
    early_stop = EarlyStopping(patience=10)

    start_epoch, best_val, counter = load_checkpoint(
        model, optimizer, scheduler
    )
    early_stop.load(best_val, counter)

    print(f"▶ Resuming from epoch {start_epoch}")

    # =======================
    # TRAINING LOOP
    # =======================
    max_epochs = 500  # big number, early stop will handle it

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0

        # ---- TRAIN ----
        for step, (X, Y) in enumerate(train_loader):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True).permute(0, 2, 1)

            with autocast(enabled=torch.cuda.is_available()):
                pred = model(X, adj)
                loss = masked_mae(pred, Y) / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, Y in val_loader:
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True).permute(0, 2, 1)
                val_loss += masked_mae(model(X, adj), Y).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        logger.log(epoch, running_loss, val_loss)


        # ---- SAVE BEST MODEL ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        # ---- SAVE CHECKPOINT ----
        save_checkpoint({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "early_stop_counter": early_stop.counter
        })

        logging.info(
            f"Epoch {epoch} | Train Loss {running_loss:.4f} | Val MAE {val_loss:.4f}"
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {running_loss:.4f} | "
            f"Val MAE {val_loss:.4f}"
        )

        # ---- EARLY STOP ----
        if early_stop.step(val_loss):
            print("🛑 Early stopping triggered")
            break


if __name__ == "__main__":
    main()
