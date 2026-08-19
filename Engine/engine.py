"""Mokymo ir vertinimo variklis. Nepriklauso nuo to, koks modelis naudojamas."""

import json
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import shift
from .models import build_model, count_params
from .utils import DEVICE, git_hash, set_seed

EVAL_BATCH = 512


def _loader(X, y, batch_size, shuffle):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, X, y, criterion=None):
    """Grąžina (loss, accuracy). Su criterion=None loss yra None."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in _loader(X, y, EVAL_BATCH, shuffle=False):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        out = model(Xb)
        if criterion is not None:
            total_loss += criterion(out, yb).item() * yb.size(0)
        correct += (out.argmax(dim=1) == yb).sum().item()
        n += yb.size(0)
    return (total_loss / n if criterion else None), correct / n


def train(model, X, y, config, log_prefix=None):
    """Apmoko modelį vietoje. Grąžina apmokytą modelį."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loader = _loader(X, y, config["batch_size"], shuffle=True)

    for epoch in range(config["epochs"]):
        model.train()
        running = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        if log_prefix:
            print(f"{log_prefix} epoch {epoch + 1}/{config['epochs']} "
                  f"| loss {running / len(loader):.4f}")
    return model


def evaluate_shifts(model, X, y, max_shift=4):
    """Tikslumas, kai testiniai vaizdai paslenkami horizontaliai 1..max_shift px.

    Modelis NEMOKOMAS iš naujo — vertinamas tas pats jau apmokytas modelis.
    """
    out = {}
    for dx in range(1, max_shift + 1):
        _, acc = evaluate(model, shift(X, dx=dx), y)
        out[f"acc_shift_{dx}px"] = round(acc, 5)
    return out


def run_one(config, seed, data, eval_shifts=False):
    """Vienas eksperimento paleidimas. Grąžina eilutę rezultatų lentelei."""
    set_seed(seed)
    X_tr, y_tr, X_val, y_val = data

    model = build_model(config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    train(model, X_tr, y_tr, config)
    seconds = time.time() - t0

    val_loss, val_acc = evaluate(model, X_val, y_val, criterion)
    _, train_acc = evaluate(model, X_tr, y_tr)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git_hash(),
        "seed": seed,
        "model": config["model"],
        "model_args": json.dumps(config.get("model_args", {}), sort_keys=True),
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "n_params": count_params(model),
        "val_acc": round(val_acc, 5),
        "val_loss": round(val_loss, 5),
        "train_acc": round(train_acc, 5),
        "train_minus_val": round(train_acc - val_acc, 5),
        "seconds": round(seconds, 1),
        "config": json.dumps(config, sort_keys=True),
    }
    if eval_shifts:
        row.update(evaluate_shifts(model, X_val, y_val))
    return row


@torch.no_grad()
def predict(model, X):
    """Grąžina prognozuotas klases (numpy)."""
    model.eval()
    preds = []
    for i in range(0, len(X), EVAL_BATCH):
        Xb = X[i:i + EVAL_BATCH].to(DEVICE)
        preds.append(model(Xb).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()