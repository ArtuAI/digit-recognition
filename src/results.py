"""Rezultatų kaupimas ir suvestinė."""

import json
import os

import pandas as pd

RESULTS_PATH = "results/runs.csv"
SUBMISSION_PATH = "results/submission.csv"


def append_result(row: dict, path: str = RESULTS_PATH) -> None:
    """Prideda eilutę į runs.csv.

    Rašoma per pandas, o ne csv.DictWriter, todėl vėliau atsiradę nauji
    stulpeliai (pvz. acc_shift_1px) senoms eilutėms tampa NaN, o ne
    sugriauna failą.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def load_results(path: str = RESULTS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nėra {path}. Pirma paleisk žvalgomąjį etapą."
        )
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Vidurkis ± std per sėklas kiekvienai konfigūracijai."""
    metrics = ["val_acc", "train_minus_val", "n_params", "seconds"]
    summary = (
        df.groupby(["model", "model_args"])
        .agg(
            val_acc_mean=("val_acc", "mean"),
            val_acc_std=("val_acc", "std"),
            n_seeds=("val_acc", "count"),
            gap=("train_minus_val", "mean"),
            n_params=("n_params", "first"),
            seconds=("seconds", "mean"),
        )
        .sort_values("val_acc_mean", ascending=False)
        .round(5)
    )
    return summary


def best_config(df: pd.DataFrame) -> dict:
    """Konfigūracija su geriausiu VIDUTINIU validacijos tikslumu.

    Renkamasi pagal vidurkį per sėklas, o ne pagal vieną geriausią
    paleidimą — kitaip rinktumeisi sėkmingiausią sėklą, ne modelį.
    """
    means = df.groupby("config")["val_acc"].mean()
    return json.loads(means.idxmax())