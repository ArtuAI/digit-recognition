"""Eksperimentų paleidiklis.

    python experiment.py explore    # paleidžia tinklelį, pildo results/runs.csv
    python experiment.py summary    # tik atspausdina suvestinę
    python experiment.py final      # geriausias config -> visas train -> submission

Etapai atskirti tyčia: galutinis modelis mokomas TIK tada, kai tu pats
pažiūrėjai į suvestinę ir sutinki su pasirinkimu.
"""

import argparse
import copy
import json

import pandas as pd

from src import data as data_mod
from src import results as res
from src.engine import predict, run_one, train
from src.models import build_model, count_params
from src.utils import DEVICE, set_seed

# ---------------------------------------------------------------- dizainas

BASE = {
    "model": "mlp",
    "model_args": {"hidden_sizes": [256, 128], "dropout": 0.0},
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 10,
}

# Kopėčios: kiekviena pakopa prideda VIENĄ veiksnį.
# Mokymo biudžetas (lr, batch_size, epochs) visose pakopose vienodas —
# kitaip lygintum architektūrą, sumaišytą su mokymo trukme.
LADDER = [
    {"model": "mlp", "model_args": {"hidden_sizes": [128]}},
    {"model": "mlp", "model_args": {"hidden_sizes": [256, 128]}},
    {"model": "mlp", "model_args": {"hidden_sizes": [512, 256, 128]}},
    {"model": "cnn", "model_args": {"channels": [32, 64]}},
    {"model": "cnn", "model_args": {"channels": [32, 64], "dropout": 0.25,
                                    "batch_norm": True}},
]

SEEDS = [0, 1, 2, 3, 4]
EVAL_SHIFTS = True   # matuoti tikslumą ir ties 1-4 px postūmiais


# ---------------------------------------------------------------- etapai

def stage_explore():
    dataset = data_mod.load_split()
    print(f"Įrenginys: {DEVICE} | mokymo imtis: {len(dataset[1])} | "
          f"validacija: {len(dataset[3])}")

    for step in LADDER:
        config = copy.deepcopy(BASE)
        config.update(copy.deepcopy(step))
        label = f"{config['model']} {config.get('model_args', {})}"
        print(f"\n--- {label} ---")

        for seed in SEEDS:
            row = run_one(config, seed, dataset, eval_shifts=EVAL_SHIFTS)
            res.append_result(row)
            line = (f"seed {seed} | val_acc {row['val_acc']:.4f} "
                    f"| gap {row['train_minus_val']:+.4f} "
                    f"| {row['n_params']:,} par | {row['seconds']}s")
            if EVAL_SHIFTS:
                line += f" | 2px {row['acc_shift_2px']:.4f}"
            print(line)

    stage_summary()


def stage_summary():
    df = res.load_results()
    print("\n--- Suvestinė (vidurkis per sėklas) ---")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(res.summarize(df))

    shift_cols = [c for c in df.columns if c.startswith("acc_shift_")]
    if shift_cols:
        print("\n--- Tikslumas pagal postūmį ---")
        cols = ["val_acc"] + sorted(shift_cols)
        print(df.groupby(["model", "model_args"])[cols].mean().round(4))


def stage_final(seed=42):
    df = res.load_results()
    config = res.best_config(df)
    print("Geriausia konfigūracija pagal vidutinį validacijos tikslumą:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    set_seed(seed)
    X, y = data_mod.load_full()
    model = build_model(config).to(DEVICE)
    print(f"\nMokoma su visais {len(y)} pavyzdžiais "
          f"({count_params(model):,} parametrų)")
    train(model, X, y, config, log_prefix="  ")

    X_test = data_mod.load_test()
    preds = predict(model, X_test)
    submission = pd.DataFrame({
        "ImageId": range(1, len(preds) + 1),
        "Label": preds,
    })
    submission.to_csv(res.SUBMISSION_PATH, index=False)
    print(f"\nSukurta: {res.SUBMISSION_PATH} ({len(preds)} prognozių)")


# ---------------------------------------------------------------- main

STAGES = {"explore": stage_explore, "summary": stage_summary,
          "final": stage_final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES), default="explore",
                        nargs="?")
    args = parser.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()