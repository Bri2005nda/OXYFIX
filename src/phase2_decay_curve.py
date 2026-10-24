# -*- coding: utf-8 -*-
"""
Phase 2 — Courbe de decroissance phylogenetique
Niveaux: aleatoire -> split genre -> leave-one-genus-out (LOGO)
Cibles : Nif CKSAAP vs RecA CKSAAP
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import load_control_cksaap, load_nif_cksaap  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
SEED = 42


def _fit_eval(Xtr, Xte, ytr, yte, seed=SEED):
    try:
        Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    if len(np.unique(ytr_s)) < 2 or len(np.unique(yte)) < 2:
        return None
    model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, proba)),
        "mcc": float(matthews_corrcoef(yte, pred)),
        "n_test": int(len(yte)),
    }


def eval_random(X, y, name):
    strat = y if min(np.bincount(y.astype(int))) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strat
    )
    m = _fit_eval(Xtr, Xte, ytr, yte)
    return {
        "name": name,
        "level": "1_random",
        "level_label": "Aleatoire",
        "protocol": "random_80_20_SMOTE",
        **m,
    }


def eval_genus_split(X, y, groups, name):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    m = _fit_eval(X[tr], X[te], y[tr], y[te])
    return {
        "name": name,
        "level": "2_genus_split",
        "level_label": "Split genre",
        "protocol": "group_shuffle_genus_SMOTE_train",
        "genres_overlap": int(len(set(groups[tr]) & set(groups[te]))),
        **m,
    }


def eval_logo(X, y, groups, name, min_test=3):
    """Leave-One-Genus-Out: moyenne des genres avec assez de souches en test."""
    logo = LeaveOneGroupOut()
    rows = []
    for tr, te in logo.split(X, y, groups):
        if len(te) < min_test:
            continue
        m = _fit_eval(X[tr], X[te], y[tr], y[te])
        if m is None:
            continue
        rows.append(m)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        "name": name,
        "level": "3_logo",
        "level_label": "Leave-one-genus-out",
        "protocol": "LOGO_mean",
        "n_folds_used": int(len(df)),
        "accuracy": float(df["accuracy"].mean()),
        "auc": float(df["auc"].mean()),
        "mcc": float(df["mcc"].mean()),
        "auc_std": float(df["auc"].std()),
        "n_test": int(df["n_test"].mean()),
    }


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Nif_CKSAAP",) + load_nif_cksaap(),
        ("RecA_CKSAAP",) + load_control_cksaap("RecA"),
    ]

    rows = []
    for name, souches, X, y, groups in datasets:
        print(f"=== {name} n={len(y)} ===")
        r1 = eval_random(X, y, name)
        print(f"  Aleatoire     AUC={r1['auc']:.3f} Acc={r1['accuracy']:.3f}")
        rows.append(r1)

        r2 = eval_genus_split(X, y, groups, name)
        print(f"  Split genre   AUC={r2['auc']:.3f} Acc={r2['accuracy']:.3f}")
        rows.append(r2)

        print("  LOGO (peut prendre plusieurs minutes)...")
        r3 = eval_logo(X, y, groups, name)
        if r3:
            print(
                f"  LOGO mean     AUC={r3['auc']:.3f}±{r3.get('auc_std', 0):.3f} "
                f"folds={r3['n_folds_used']}"
            )
            rows.append(r3)
        else:
            print("  LOGO: aucun fold utilisable")

    df = pd.DataFrame(rows)
    csv_path = OUT_METRICS / "phase2_decay_curve.csv"
    df.to_csv(csv_path, index=False)

    # Figure courbe de decroissance
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = ["Aleatoire", "Split genre", "Leave-one-genus-out"]
    for name, color, marker in [
        ("Nif_CKSAAP", "#1f4e79", "o"),
        ("RecA_CKSAAP", "#8b6914", "s"),
    ]:
        sub = df[df["name"] == name].set_index("level_label").reindex(order)
        ax.plot(
            order,
            sub["auc"].values,
            marker=marker,
            lw=2,
            color=color,
            label=name,
        )
        for i, v in enumerate(sub["auc"].values):
            if pd.notna(v):
                ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=color)
    ax.axhline(0.90, color="red", ls="--", lw=1, label="seuil 0.90")
    ax.axhline(0.70, color="orange", ls="--", lw=1, label="seuil 0.70")
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUC (HGB)")
    ax.set_title("Phase 2 — Courbe de decroissance phylogenetique")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = OUT_FIG / "phase2_decay_curve.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Decision
    nif = df[df["name"] == "Nif_CKSAAP"].set_index("level_label")
    rec = df[df["name"] == "RecA_CKSAAP"].set_index("level_label")
    lines = [
        "PHASE 2 — COURBE DE DECROISSANCE",
        "",
        "Niveaux: Aleatoire -> Split genre -> Leave-one-genus-out",
        "",
        "Nif AUC:",
    ]
    for lv in order:
        if lv in nif.index and pd.notna(nif.loc[lv, "auc"]):
            lines.append(f"  {lv:22s} {nif.loc[lv, 'auc']:.3f}")
    lines.append("RecA AUC:")
    for lv in order:
        if lv in rec.index and pd.notna(rec.loc[lv, "auc"]):
            lines.append(f"  {lv:22s} {rec.loc[lv, 'auc']:.3f}")
    lines += [
        "",
        "LECTURE:",
        "- Si Nif et RecA chutent ensemble: signal surtout taxonomique.",
        "- Si Nif reste au-dessus de RecA aux niveaux durs: residu nif possible.",
        "",
        f"CSV   : {csv_path}",
        f"Figure: {fig_path}",
    ]
    dec = OUT_METRICS / "PHASE2_DECISION.txt"
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nOK -> {dec}")


if __name__ == "__main__":
    main()