"""Rezultatų analizė 3 skyriui.

    python analysis.py

Ką daro:
  1. Suskaičiuoja vidurkius ir standartinius nuokrypius iš results/runs.csv.
  2. Nubraižo du pagrindinius paveikslus su klaidų intervalais.
  3. Atlieka McNemar testus dviem iš anksto suplanuotiems palyginimams.
  4. Sudaro sumaišties matricas ir išsaugo klaidingai klasifikuotų vaizdų tinklelį.

SVARBU: McNemar testui reikia atskirų pavyzdžių prognozių, kurių runs.csv
nesaugo. Todėl 3 ir 4 žingsniai iš naujo apmoko DU modelius (geriausią MLP ir
geriausią CNN) su 5 sėklomis. Prognozės kešuojamos results/preds/, tad antrą
kartą paleidus skaičiuojama tik tai, ko dar nėra.
"""

import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix

from src import data as data_mod
from src import results as res
from src.data import shift
from src.engine import predict, train
from src.models import build_model
from src.utils import DEVICE, set_seed

FIG_DIR = "figures"
PRED_DIR = "results/preds"
SHIFTS = [0, 1, 2, 3, 4]
ALPHA = 0.05

# Iš anksto suplanuoti palyginimai (2.5 poskyris)
COMPARISONS = [(0, "pirminis"), (3, "antrinis")]


# ---------------------------------------------------------------- 1. suvestinė

def shift_columns(df):
    return sorted(
        [c for c in df.columns if c.startswith("acc_shift_")],
        key=lambda c: int(c.split("_")[2].replace("px", "")),
    )


def summary_table(df):
    """Vidurkis ir imties standartinis nuokrypis (ddof=1) per sėklas."""
    metrics = ["val_acc"] + shift_columns(df)
    grouped = df.groupby(["model", "model_args"])
    out = pd.DataFrame(index=grouped.size().index)
    out["n_seeds"] = grouped.size()
    out["n_params"] = grouped["n_params"].first()
    for m in metrics:
        out[f"{m}_mean"] = grouped[m].mean()
        out[f"{m}_std"] = grouped[m].std(ddof=1)
    return out.sort_values("val_acc_mean", ascending=False)


def report_measurement_floor(summary, df):
    """Matavimo riba: kokio dydžio skirtumai apskritai interpretuotini."""
    print("\n=== 3.1 Matavimo neapibrėžtis ===")
    for metric in ["val_acc"] + shift_columns(df):
        stds = summary[f"{metric}_std"]
        print(f"{metric:>18}: std nuo {stds.min():.4f} iki {stds.max():.4f} "
              f"(mediana {stds.median():.4f})")
    floor = 2 * summary["val_acc_std"].max()
    print(f"\nSkirtumai, mažesni nei ~{floor:.4f} ({floor * 100:.2f} pp) "
          f"nepakeistoje validacijoje, neinterpretuotini.")


# ---------------------------------------------------------------- 2. paveikslai

def _label(model, args_json):
    a = json.loads(args_json)
    if model == "mlp":
        return "MLP " + "-".join(map(str, a["hidden_sizes"]))
    extras = []
    if a.get("batch_norm"):
        extras.append("BN")
    if a.get("dropout"):
        extras.append(f"drop {a['dropout']}")
    return "CNN " + "/".join(map(str, a["channels"])) + (
        f" + {' + '.join(extras)}" if extras else "")


def plot_ladder(summary):
    """1 pav. Tikslumas nepakeistoje validacijoje su klaidų intervalais."""
    labels = [_label(m, a) for m, a in summary.index]
    means = summary["val_acc_mean"] * 100
    errs = summary["val_acc_std"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), means, yerr=errs, capsize=4,
           color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Validacijos tikslumas (%)")
    ax.set_ylim(min(means) - 2, 100)
    ax.set_title("Tikslumas nepakeistoje validacijos aibėje (n=5 sėklos)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = f"{FIG_DIR}/fig1_ladder.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Išsaugota: {path}")


def plot_shift_curves(summary, df):
    """2 pav. Tikslumas pagal postūmį — darbo centrinis paveikslas."""
    cols = ["val_acc"] + shift_columns(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (model, args_json) in summary.index:
        means = [summary.loc[(model, args_json), f"{c}_mean"] * 100 for c in cols]
        errs = [summary.loc[(model, args_json), f"{c}_std"] * 100 for c in cols]
        ax.errorbar(SHIFTS, means, yerr=errs, marker="o", capsize=3,
                    linewidth=1.8, label=_label(model, args_json))
    ax.set_xlabel("Horizontalus postūmis (pikseliai)")
    ax.set_ylabel("Tikslumas (%)")
    ax.set_xticks(SHIFTS)
    ax.set_title("Tikslumas pagal postūmį (vidurkis ± std, n=5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIG_DIR}/fig2_shift_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Išsaugota: {path}")


# ---------------------------------------------------------------- 3. prognozės

def _tag(config):
    h = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return f"{config['model']}_{h[:8]}"


def get_predictions(config, seed, X_val, y_val):
    """Grąžina {postūmis: prognozės}. Kešuojama diske."""
    os.makedirs(PRED_DIR, exist_ok=True)
    path = f"{PRED_DIR}/{_tag(config)}_seed{seed}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {int(k): z[k] for k in z.files}

    print(f"  mokoma {_tag(config)} seed={seed} (nėra keše)...")
    set_seed(seed)
    X_tr, y_tr, _, _ = DATA
    model = build_model(config).to(DEVICE)
    train(model, X_tr, y_tr, config)

    preds = {}
    for dx in SHIFTS:
        Xs = X_val if dx == 0 else shift(X_val, dx=dx)
        preds[dx] = predict(model, Xs)
    np.savez_compressed(path, **{str(k): v for k, v in preds.items()})
    return preds


# ---------------------------------------------------------------- 4. McNemar

def mcnemar(pred_a, pred_b, y_true):
    """Dvipusis McNemar. Grąžina (b, c, p).

    b = A teisingai, B klaidingai;  c = A klaidingai, B teisingai.
    Esant mažam b+c naudojamas tikslusis binominis testas, kitu atveju —
    chi kvadrato aproksimacija su tolydumo pataisa.
    """
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n < 25:
        p = stats.binomtest(b, n, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n
        p = stats.chi2.sf(chi2, df=1)
    return b, c, float(p)


def run_comparisons(df, seeds):
    """Geriausias MLP prieš geriausią CNN, atskirai kiekvienai sėklai."""
    _, _, X_val, y_val = DATA
    y = y_val.numpy()

    means = df.groupby(["model", "config"])["val_acc"].mean()
    best_mlp = json.loads(means["mlp"].idxmax())
    best_cnn = json.loads(means["cnn"].idxmax())

    print("\n=== 3.4 Statistinis palyginimas (McNemar) ===")
    print(f"MLP: {json.dumps(best_mlp['model_args'])}")
    print(f"CNN: {json.dumps(best_cnn['model_args'])}")

    rows = []
    preds_mlp = {s: get_predictions(best_mlp, s, X_val, y_val) for s in seeds}
    preds_cnn = {s: get_predictions(best_cnn, s, X_val, y_val) for s in seeds}

    for dx, kind in COMPARISONS:
        print(f"\n-- Postūmis {dx} px ({kind}) --")
        sig, direction = [], []
        for s in seeds:
            pm, pc = preds_mlp[s][dx], preds_cnn[s][dx]
            b, c, p = mcnemar(pm, pc, y)
            sig.append(p < ALPHA)
            direction.append(c > b)   # True = CNN geresnis
            rows.append({"shift_px": dx, "seed": s, "mlp_only_correct": b,
                         "cnn_only_correct": c, "p_value": p,
                         "significant": p < ALPHA})
            print(f"  seed {s}: MLP tik {b:>4} | CNN tik {c:>4} | p = {p:.3g}"
                  f"{'  *' if p < ALPHA else ''}")
        if all(sig) and len(set(direction)) == 1:
            who = "CNN" if direction[0] else "MLP"
            print(f"  -> patvirtinta: {who} geresnis visose 5 sėklose (α={ALPHA})")
        else:
            print("  -> NEpatvirtinta: reikšmingumas arba kryptis nevienoda")

    out = pd.DataFrame(rows)
    out.to_csv("results/mcnemar.csv", index=False)
    print("\nIšsaugota: results/mcnemar.csv")
    return best_mlp, best_cnn, preds_mlp, preds_cnn


# ---------------------------------------------------------------- 5. klaidos

def plot_confusion(preds_mlp, preds_cnn, seed=0):
    """3 pav. Sumaišties matricos nepakeistoje validacijoje."""
    _, _, _, y_val = DATA
    y = y_val.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, preds, name in ((axes[0], preds_mlp[seed][0], "MLP"),
                            (axes[1], preds_cnn[seed][0], "CNN")):
        cm = confusion_matrix(y, preds)
        np.fill_diagonal(cm, 0)          # įstrižainė nuslopina viską kita
        im = ax.imshow(cm, cmap="Reds")
        ax.set_title(f"{name} (sėkla {seed}, be įstrižainės)")
        ax.set_xlabel("Prognozuota")
        ax.set_ylabel("Tikra")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                if cm[i, j] > 0:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = f"{FIG_DIR}/fig3_confusion.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Išsaugota: {path}")

    for preds, name in ((preds_mlp[seed][0], "MLP"), (preds_cnn[seed][0], "CNN")):
        cm = confusion_matrix(y, preds)
        np.fill_diagonal(cm, 0)
        pairs = [(cm[i, j], i, j) for i in range(10) for j in range(10) if cm[i, j]]
        top = sorted(pairs, reverse=True)[:5]
        print(f"\n{name} dažniausios painiavos (tikra -> prognozuota):")
        for n, i, j in top:
            print(f"  {i} -> {j}: {n}")


def plot_error_grid(preds_cnn, seed=0, n=25):
    """4 pav. Klaidingai klasifikuoti vaizdai kokybinei analizei."""
    _, _, X_val, y_val = DATA
    y = y_val.numpy()
    p = preds_cnn[seed][0]
    idx = np.where(p != y)[0][:n]
    if len(idx) == 0:
        print("Klaidų nerasta.")
        return
    side = int(np.ceil(np.sqrt(len(idx))))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.3, side * 1.4))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.array(axes).ravel(), idx):
        ax.imshow(X_val[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"{y[i]} -> {p[i]}", fontsize=8)
    fig.suptitle("CNN klaidos nepakeistoje validacijoje")
    fig.tight_layout()
    path = f"{FIG_DIR}/fig4_errors.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Išsaugota: {path}")
    print(f"Peržiūrėk šį paveikslą akimis ir suskirstyk klaidas: "
          f"neaiški rašysena / ginčytina etiketė / tikra modelio klaida.")


# ---------------------------------------------------------------- main

DATA = None


def main():
    global DATA
    os.makedirs(FIG_DIR, exist_ok=True)
    DATA = data_mod.load_split()

    df = res.load_results()
    print(f"Įrenginys: {DEVICE} | eilučių runs.csv: {len(df)}")
    hashes = df["git"].unique()
    if len(hashes) > 1:
        print(f"ĮSPĖJIMAS: rezultatai gauti su {len(hashes)} kodo versijomis: "
              f"{list(hashes)}")

    summary = summary_table(df)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print("\n=== 3.2 / 3.3 Suvestinė ===")
        print(summary.round(5))
    summary.to_csv("results/summary.csv")
    print("\nIšsaugota: results/summary.csv")

    report_measurement_floor(summary, df)
    plot_ladder(summary)
    plot_shift_curves(summary, df)

    seeds = sorted(df["seed"].unique())
    _, _, preds_mlp, preds_cnn = run_comparisons(df, seeds)
    plot_confusion(preds_mlp, preds_cnn)
    plot_error_grid(preds_cnn)


if __name__ == "__main__":
    main()