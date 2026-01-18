import torch
import os

def save_checkpoint(state, path="checkpoint.pt"):
    torch.save(state, path)


def load_checkpoint(model, optimizer, scheduler, path="checkpoint.pt"):
    if not os.path.exists(path):
        return 0, float("inf"), 0

    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt["epoch"], ckpt["best_val"], ckpt["early_stop_counter"]
