"""Modelių registras.

Naujam modeliui pridėti reikia DVIEJŲ dalykų:
  1. funkcijos, kuri grąžina nn.Module,
  2. įrašo REGISTRY žodyne.

Daugiau niekur kode nieko keisti nereikia.
"""

import torch.nn as nn


def build_mlp(hidden_sizes=(256, 128), dropout=0.0):
    """784 -> hidden_sizes -> 10"""
    layers, prev = [], 784
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 10))
    return nn.Sequential(*layers)


def build_cnn(channels=(32, 64), fc_size=128, dropout=0.0, batch_norm=False):
    """(N,784) -> (N,1,28,28) -> conv blokai -> 10

    Kiekvienas blokas: du konvoliuciniai sluoksniai + MaxPool(2).
    Su channels=(32, 64): 28x28 -> 14x14 -> 7x7.
    """
    def conv_pair(in_ch, out_ch):
        block = []
        for a, b in ((in_ch, out_ch), (out_ch, out_ch)):
            block.append(nn.Conv2d(a, b, kernel_size=3, padding=1))
            if batch_norm:
                block.append(nn.BatchNorm2d(b))
            block.append(nn.ReLU())
        return block

    layers = [nn.Unflatten(1, (1, 28, 28))]
    prev = 1
    for ch in channels:
        layers += conv_pair(prev, ch)
        layers.append(nn.MaxPool2d(2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = ch

    side = 28 // (2 ** len(channels))
    layers += [nn.Flatten(), nn.Linear(prev * side * side, fc_size), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(fc_size, 10))
    return nn.Sequential(*layers)


REGISTRY = {
    "mlp": build_mlp,
    "cnn": build_cnn,
}


def build_model(config: dict) -> nn.Module:
    name = config["model"]
    if name not in REGISTRY:
        raise ValueError(
            f"Nežinomas modelis '{name}'. Galimi: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**config.get("model_args", {}))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)