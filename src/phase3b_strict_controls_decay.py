# -*- coding: utf-8 -*-
"""
Phase 3b — Controles + courbe de decroissance sur corpus STRICT (489)

Compare Full 787 vs Strict 489:
- Nif vs RecA (et GyrB/RpoB en aleatoire/genre)
- Courbe: aleatoire -> split genre -> LOGO
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
from phase1_premium_complete import (  # noqa: E402
    load_control_cksaap,
    load_nif_cksaap,
)

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
SEED = 42


def strict_set() -> set[str]:
    p3 = pd.read_csv(P3)
    return set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))


def filter_to(souches, X, y, groups, keep: set[str]):
    mask = np.array([s in keep for s in souches])
    return (
        [s for s, m in zip(souches, mask) if m],
        X[mask],
        y[mask],
        groups[mask],
    )


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


def eval_random(X, y, name, corpus):
    strat = y if min(np.bincount(y.astype(int))) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strat
    )
    m = _fit_eval(Xtr, Xte, ytr, yte)
    if m is None:
        return None
    return {
        "name": name,
        "corpus": corpus,
        "level": "1_random",
        "level_label": "Aleatoire",
        "protocol": "random_80_20_SMOTE",
        "n": int(len(y)),
        "n_aerobie": int((y == 1).sum()),
        "n_anaerobie": int((y == 0).sum()),
        **m,
    }


def eval_genus(X, y, groups, name, corpus):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    m = _fit_eval(X[tr], X[te], y[tr], y[te])
    if m is None:
        return None
    return {
        "name": name,
        "corpus": corpus,
        "level": "2_genus_split",
        "level_label": "Split genre",
        "protocol": "group_shuffle_genus_SMOTE_train",
        "n": int(len(y)),
        "n_aerobie": int((y == 1).sum()),
        "n_anaerobie": int((y == 0).sum()),
        "genres_overlap": int(len(set(groups[tr]) & set(groups[te]))),
        **m,
    }


def eval_logo(X, y, groups, name, corpus, min_test=3):
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
        "corpus": corpus,
        "level": "3_logo",
        "level_label": "Leave-one-genus-out",
        "protocol": "LOGO_mean",
        "n": int(len(y)),
        "n_aerobie": int((y == 1).sum()),
        "n_anaerobie": int((y == 0).sum()),
        "n_folds_used": int(len(df)),
        "accuracy": float(df["accuracy"].mean()),
        "auc": float(df["auc"].mean()),
        "mcc": float(df["mcc"].mean()),
        "auc_std": float(df["auc"].std()),
        "n_test": float(df["n_test"].mean()),
    }


def run_feature(name, loader_fn, keep_strict, do_logo=True):
    rows = []
    if name == "Nif_CKSAAP":
        souches, X, y, groups = loader_fn()
    else:
        souches, X, y, groups = loader_fn(name.replace("_CKSAAP", ""))

    for corpus_name, keep in [("full_787", set(souches)), ("strict_489", keep_strict)]:
        s, Xx, yy, gg = filter_to(souches, X, y, groups, keep)
        if len(yy) < 40 or len(np.unique(yy)) < 2:
            print(f"  SKIP {name} {corpus_name} n={len(yy)}")
            continue
        print(f"=== {name} | {corpus_name} n={len(yy)} "
              f"aero={int((yy==1).sum())} ana={int((yy==0).sum())} ===")
        r1 = eval_random(Xx, yy, name, corpus_name)
        if r1:
            print(f"  Aleatoire   AUC={r1['auc']:.3f}")
            rows.append(r1)
        r2 = eval_genus(Xx, yy, gg, name, corpus_name)
        if r2:
            print(f"  Split genre AUC={r2['auc']:.3f}")
            rows.append(r2)
        if do_logo and name in ("Nif_CKSAAP", "RecA_CKSAAP"):
            print("  LOGO...")
            r3 = eval_logo(Xx, yy, gg, name, corpus_name)
            if r3:
                print(
                    f"  LOGO mean   AUC={r3['auc']:.3f}±{r3.get('auc_std', 0):.3f} "
                    f"folds={r3['n_folds_used']}"
                )
                rows.append(r3)
    return rows


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    keep = strict_set()
    print(f"Strict set size: {len(keep)}")

    rows = []
    rows += run_feature("Nif_CKSAAP", load_nif_cksaap, keep, do_logo=True)
    rows += run_feature("RecA_CKSAAP", load_control_cksaap, keep, do_logo=True)
    # controles supplementaires: aleatoire + genre seulement
    for gene in ("GyrB", "RpoB"):
        rows += run_feature(f"{gene}_CKSAAP", load_control_cksaap, keep, do_logo=False)

    df = pd.DataFrame(rows)
    csv_path = OUT_METRICS / "phase3b_strict_controls_decay.csv"
    df.to_csv(csv_path, index=False)

    # Figure: decay Full vs Strict for Nif and RecA
    order = ["Aleatoire", "Split genre", "Leave-one-genus-out"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, corpus, title in zip(
        axes,
        ["full_787", "strict_489"],
        ["A) Corpus Full 787", "B) Corpus Strict 489"],
    ):
        for name, color, marker in [
            ("Nif_CKSAAP", "#1f4e79", "o"),
            ("RecA_CKSAAP", "#8b6914", "s"),
        ]:
            sub = df[(df["name"] == name) & (df["corpus"] == corpus)]
            sub = sub.set_index("level_label").reindex(order)
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
                    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=color)
        ax.axhline(0.90, color="red", ls="--", lw=1)
        ax.axhline(0.70, color="orange", ls="--", lw=1)
        ax.set_ylim(0.45, 1.05)
        ax.set_title(title)
        ax.set_ylabel("AUC (HGB)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Phase 3b — Decroissance phylogenetique Full vs Strict",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = OUT_FIG / "phase3b_strict_decay.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Bar chart controles strict random
    fig2, ax = plt.subplots(figsize=(7.5, 4.2))
    sub = df[
        (df["corpus"] == "strict_489")
        & (df["level_label"] == "Aleatoire")
        & (df["name"].isin(["Nif_CKSAAP", "RecA_CKSAAP", "GyrB_CKSAAP", "RpoB_CKSAAP"]))
    ].copy()
    sub = sub.set_index("name").reindex(
        ["Nif_CKSAAP", "RecA_CKSAAP", "GyrB_CKSAAP", "RpoB_CKSAAP"]
    )
    cols = ["#1f4e79", "#8b6914", "#8b6914", "#8b6914"]
    ax.bar(range(len(sub)), sub["auc"].values, color=cols, edgecolor="black", lw=0.4)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(["Nif", "RecA", "GyrB", "RpoB"])
    ax.axhline(0.90, color="red", ls="--", lw=1)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("Strict 489 — Nif vs controles (split aleatoire)")
    for i, v in enumerate(sub["auc"].values):
        if pd.notna(v):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig2.tight_layout()
    fig2_path = OUT_FIG / "phase3b_strict_controls_random.png"
    fig2.savefig(fig2_path, dpi=170, bbox_inches="tight")
    plt.close(fig2)

    def auc_of(name, corpus, level):
        r = df[
            (df["name"] == name)
            & (df["corpus"] == corpus)
            & (df["level_label"] == level)
        ]
        return float(r["auc"].iloc[0]) if len(r) else float("nan")

    lines = [
        "PHASE 3b — CONTROLES + DECROISSANCE SUR CORPUS STRICT",
        "Corpus strict = 489 documentes BacDive (labels Alida inchanges)",
        "",
        "1) CONTROLES (AUC aleatoire, Strict 489)",
        f"  Nif  = {auc_of('Nif_CKSAAP','strict_489','Aleatoire'):.3f}",
        f"  RecA = {auc_of('RecA_CKSAAP','strict_489','Aleatoire'):.3f}",
        f"  GyrB = {auc_of('GyrB_CKSAAP','strict_489','Aleatoire'):.3f}",
        f"  RpoB = {auc_of('RpoB_CKSAAP','strict_489','Aleatoire'):.3f}",
        "",
        "2) COURBE DE DECROISSANCE (AUC)",
        "  Full 787:",
        f"    Nif  A={auc_of('Nif_CKSAAP','full_787','Aleatoire'):.3f} "
        f"G={auc_of('Nif_CKSAAP','full_787','Split genre'):.3f} "
        f"L={auc_of('Nif_CKSAAP','full_787','Leave-one-genus-out'):.3f}",
        f"    RecA A={auc_of('RecA_CKSAAP','full_787','Aleatoire'):.3f} "
        f"G={auc_of('RecA_CKSAAP','full_787','Split genre'):.3f} "
        f"L={auc_of('RecA_CKSAAP','full_787','Leave-one-genus-out'):.3f}",
        "  Strict 489:",
        f"    Nif  A={auc_of('Nif_CKSAAP','strict_489','Aleatoire'):.3f} "
        f"G={auc_of('Nif_CKSAAP','strict_489','Split genre'):.3f} "
        f"L={auc_of('Nif_CKSAAP','strict_489','Leave-one-genus-out'):.3f}",
        f"    RecA A={auc_of('RecA_CKSAAP','strict_489','Aleatoire'):.3f} "
        f"G={auc_of('RecA_CKSAAP','strict_489','Split genre'):.3f} "
        f"L={auc_of('RecA_CKSAAP','strict_489','Leave-one-genus-out'):.3f}",
        "",
        "LECTURE:",
        "- Si sous Strict les controles restent ~aussi hauts que nif en aleatoire,",
        "  le diagnostic taxonomique de Phase 1 EST ROBUSTE a la proprete des labels.",
        "- Comparer Full vs Strict sur LOGO: si les deux tombent vers ~0.5-0.6,",
        "  la conclusion Phase 2 tient aussi sur corpus defendable.",
        "- Attention: Strict est desequilibre (150 aero / 339 anaero) ->",
        "  interpreter les splits genre avec prudence.",
        "",
        f"CSV    : {csv_path}",
        f"Figure : {fig_path}",
        f"Figure2: {fig2_path}",
    ]
    dec = OUT_METRICS / "PHASE3B_STRICT_DECISION.txt"
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nOK -> {dec}")


if __name__ == "__main__":
    main()
