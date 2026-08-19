"""Duomenų užkrovimas ir transformacijos.

Visi tenzorai yra plokšti (N, 784). Konvoliuciniai modeliai patys
persiformuoja į (N, 1, 28, 28) — taip duomenų kodo keisti nereikia.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Skaidymo sėkla NIEKADA nesikeičia tarp eksperimentų — kitaip lygintum
# modelius, įvertintus ant skirtingų validacijos aibių.
SPLIT_SEED = 42


def _to_tensors(df: pd.DataFrame):
    X = df.drop("label", axis=1).values.astype(np.float32) / 255.0
    y = df["label"].values.astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def load_split(val_size: float = 0.2):
    """Grąžina (X_train, y_train, X_val, y_val)."""
    X, y = _to_tensors(pd.read_csv(TRAIN_PATH))
    idx = np.arange(len(y))
    idx_tr, idx_val = train_test_split(
        idx, test_size=val_size, random_state=SPLIT_SEED, stratify=y.numpy()
    )
    return X[idx_tr], y[idx_tr], X[idx_val], y[idx_val]


def load_full():
    """Visas train.csv — galutiniam modeliui prieš submission."""
    return _to_tensors(pd.read_csv(TRAIN_PATH))


def load_test():
    """Kaggle test.csv (be etikečių)."""
    X = pd.read_csv(TEST_PATH).values.astype(np.float32) / 255.0
    return torch.from_numpy(X)


def shift(X: torch.Tensor, dx: int = 0, dy: int = 0) -> torch.Tensor:
    """Ciklinis vaizdų postūmis. (N, 784) -> (N, 784).

    MNIST skaitmenys centruoti, o kraštai juodi, tad ties 1–4 pikselių
    postūmiu ciklinis persivyniojimas praktiškai nesiskiria nuo paprasto
    poslinkio su juodu užpildymu.
    """
    imgs = X.reshape(-1, 28, 28)
    return torch.roll(imgs, shifts=(dy, dx), dims=(1, 2)).reshape(-1, 784)