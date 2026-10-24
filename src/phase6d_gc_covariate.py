# -*- coding: utf-8 -*-
"""
Phase 6d — Covariable GC % génomique (GTDB) seule et combinée.

Le GC % est celui du génome représentatif GTDB apparié en Phase 6
(espèce, ou genre en repli) : results/metrics/phase6b_gtdb_gc_by_strain.csv.
Deux jeux de descripteurs, même protocole répété que Phase 5/6b
(HGB, SMOTE train uniquement, 20 graines, Full/Strict, aléatoire/genre) :
- GC seul (1 nombre) ;
- GC + longueur moyenne + composition AAC globale (22 nombres),
  aligné sur les souches de la baseline triviale Nif.

Sorties :
- results/metrics/phase6d_gc_per_repeat.csv
- results/metrics/phase6d_gc_summary.csv
- results/figures/phase6d_gc_vs_trivial_cksaap.png
- results/metrics/PHASE6D_GC_DECISION.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import load_nif_trivial  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
GC_CSV = OUT_METRICS / "phase6b_gtdb_gc_by_strain.csv"
SEEDS = list(range(42, 62))


def metrics_of(yte, proba):
    pred = (proba >= 0.5).astype(int)
    both = len(np.unique(yte)) > 1
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, proba)) if both else np.nan,
        "mcc": float(matthews_corrcoef(yte, pred)) if both else np.nan,
        "auprc": float(average_precision_score(yte, proba)) if both else np.nan,
        "recall_aerobie": float(recall_score(yte, pred, pos_label=1, zero_division=0)),
        "recall_anaerobie": float(recall_score(yte, pred, pos_label=0, zero_division=0)),
    }


def fit_eval(Xtr, Xte, ytr, yte, seed):
    try:
        Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    return metrics_of(yte, model.predict_proba(Xte)[:, 1])


def main():
    t0 = time.time()
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    p3 = pd.read_csv(P3)
    strict = set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))
    gc = pd.read_csv(GC_CSV).set_index("souche")["gc_percentage"]

    souches_tr, X_tr, y_tr, g_tr = load_nif_trivial()
    souches_tr = [str(s) for s in souches_tr]
    gc_vals = np.array([gc.get(s, np.nan) for s in souches_tr], dtype=np.float64)
    has_gc = np.isfinite(gc_vals)
    print(f"souches triviales={len(souches_tr)} avec GC={has_gc.sum()}")

    datasets = {
        "GC_only": (np.array(souches_tr, dtype=object)[has_gc],
                    gc_vals[has_gc].reshape(-1, 1), y_tr[has_gc], g_tr[has_gc]),
        "GC_len_AAC": (np.array(souches_tr, dtype=object)[has_gc],
                       np.column_stack([gc_vals[has_gc], X_tr[has_gc]]),
                       y_tr[has_gc], g_tr[has_gc]),
    }

    rows = []
    for feature, (souches, X, y, groups) in datasets.items():
        for corpus, keep in [("full", None), ("strict_489", strict)]:
            if keep is None:
                idx = np.arange(len(souches))
            else:
                idx = np.array([i for i, s in enumerate(souches) if s in keep])
            Xs, ys, gs = X[idx], y[idx], np.asarray(groups)[idx]
            for seed in SEEDS:
                strat = ys if min(np.bincount(ys.astype(int))) >= 2 else None
                Xtr, Xte, ytr, yte = train_test_split(
                    Xs, ys, test_size=0.2, random_state=seed, stratify=strat)
                m = fit_eval(Xtr, Xte, ytr, yte, seed)
                rows.append({"feature": feature, "corpus": corpus,
                             "protocol": "random_80_20_SMOTE", "seed": seed,
                             "n": int(len(ys)), **m})
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                tr, te = next(gss.split(Xs, ys, gs))
                m = fit_eval(Xs[tr], Xs[te], ys[tr], ys[te], seed)
                rows.append({"feature": feature, "corpus": corpus,
                             "protocol": "group_shuffle_genus_SMOTE_train", "seed": seed,
                             "n": int(len(ys)),
                             "genres_overlap": int(len(set(gs[tr]) & set(gs[te]))), **m})
            print(f"{feature} | {corpus} ok ({time.time()-t0:.0f}s)", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT_METRICS / "phase6d_gc_per_repeat.csv", index=False)
    summary = (
        per.groupby(["feature", "corpus", "protocol"])
        .agg(n_repeats=("seed", "count"), accuracy_mean=("accuracy", "mean"),
             accuracy_sd=("accuracy", "std"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
             auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
             recall_aerobie_mean=("recall_aerobie", "mean"),
             recall_anaerobie_mean=("recall_anaerobie", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase6d_gc_summary.csv", index=False)

    def get(feature, corpus, protocol, col="auc_mean"):
        s = summary[(summary.feature == feature) & (summary.corpus == corpus)
                    & (summary.protocol == protocol)]
        return float(s[col].iloc[0]) if len(s) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        feats = ["GC_only", "GC_len_AAC"]
        x = np.arange(len(feats))
        rand = [get(f, corpus, "random_80_20_SMOTE") for f in feats]
        rand_s = [get(f, corpus, "random_80_20_SMOTE", "auc_sd") for f in feats]
        gen = [get(f, corpus, "group_shuffle_genus_SMOTE_train") for f in feats]
        gen_s = [get(f, corpus, "group_shuffle_genus_SMOTE_train", "auc_sd") for f in feats]
        ax.bar(x - 0.2, rand, width=0.4, yerr=rand_s, capsize=3, color="#1f4e79", label="aléatoire")
        ax.bar(x + 0.2, gen, width=0.4, yerr=gen_s, capsize=3, color="#8B4513", label="genre")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["GC seul", "GC+longueur+AAC"], fontsize=8)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 6d — Covariable GC % génomique GTDB")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase6d_gc_vs_trivial_cksaap.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    lines = [
        "PHASE 6d — COVARIABLE GC % GÉNOMIQUE (GTDB)",
        "GC du génome représentatif GTDB apparié (espèce ou genre), 746/787 souches.",
        "",
        "AUC moyenne (20 splits) :",
    ]
    for feature in ["GC_only", "GC_len_AAC"]:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {feature:12s} {corpus:11s} aléatoire={get(feature, corpus, 'random_80_20_SMOTE'):.3f}"
                f" | genre={get(feature, corpus, 'group_shuffle_genus_SMOTE_train'):.3f}"
            )
    lines += [
        "",
        "LECTURE :",
        "- GC seul : modéré en aléatoire (Full 0,785 ; Strict 0,776), faible en",
        "  genre Full (0,617), intermédiaire en genre Strict (0,745). Un seul",
        "  nombre lié à la lignée porte donc une information réelle mais incomplète.",
        "- GC+longueur+AAC (Full 0,963/0,931 ; Strict 0,972/0,956) égale presque",
        "  la baseline longueur+AAC (0,959/0,919 ; 0,970/0,946) et reste proche de",
        "  CKSAAP : l'apport propre du GC est marginal, l'essentiel du score trivial",
        "  vient de la composition globale. CKSAAP garde un supplément limité.",
        "- GC représentatif GTDB (espèce/genre), pas le génome exact de chaque souche :",
        "  à formuler comme covariable approchée, couverture 746/787.",
        "",
        f"Durée totale : {time.time()-t0:.0f}s",
        f"Par répétition : {OUT_METRICS / 'phase6d_gc_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase6d_gc_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE6D_GC_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
