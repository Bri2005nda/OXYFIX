# -*- coding: utf-8 -*-
"""
Phase 6b — Baselines triviales répétées (longueur + composition AAC).

Même protocole que Phase 5 (HGB, SMOTE train uniquement, 20 graines,
corpus Full/Strict, splits aléatoire et genre), mais descripteurs
réduits à 21 nombres : longueur moyenne des séquences + proportions
globales des 20 acides aminés. Protéines : Nif (fichier FASTA souche)
et RecA (contrôle). Le GC % génomique est traité séparément dès que
les métadonnées GTDB sont disponibles (voir décision).

Sorties :
- results/metrics/phase6b_trivial_per_repeat.csv
- results/metrics/phase6b_trivial_summary.csv
- results/figures/phase6b_trivial_vs_cksaap.png
- results/metrics/PHASE6B_TRIVIAL_DECISION.txt
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
from phase1_premium_complete import load_control_trivial, load_nif_trivial  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
PHASE5_SUMMARY = OUT_METRICS / "phase5_repeated_splits_summary.csv"
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

    datasets = {}
    s, X, y, g = load_nif_trivial()
    datasets["Nif_trivial_len_AAC"] = (s, X, y, g)
    s, X, y, g = load_control_trivial("RecA")
    datasets["RecA_trivial_len_AAC"] = (s, X, y, g)

    rows = []
    for feature, (souches, X, y, groups) in datasets.items():
        souches = [str(s) for s in souches]
        for corpus, keep in [("full", None), ("strict_489", strict)]:
            if keep is None:
                idx = np.arange(len(souches))
            else:
                idx = np.array([i for i, s in enumerate(souches) if s in keep])
            Xs, ys, gs = X[idx], y[idx], groups[idx]
            for seed in SEEDS:
                strat = ys if min(np.bincount(ys.astype(int))) >= 2 else None
                Xtr, Xte, ytr, yte = train_test_split(
                    Xs, ys, test_size=0.2, random_state=seed, stratify=strat
                )
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
            print(f"{feature} | {corpus} ok ({time.time()-t0:.0f}s)")

    per = pd.DataFrame(rows)
    per.to_csv(OUT_METRICS / "phase6b_trivial_per_repeat.csv", index=False)
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
    summary.to_csv(OUT_METRICS / "phase6b_trivial_summary.csv", index=False)

    # figure vs CKSAAP Phase 5
    ref = pd.read_csv(PHASE5_SUMMARY)
    combos = [
        ("Nif_trivial_len_AAC", "Nif_CKSAAP", "Nif"),
        ("RecA_trivial_len_AAC", "RecA_CKSAAP", "RecA"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (triv_name, ref_name, short) in zip(axes, combos):
        labels, triv_m, triv_s, ref_m, ref_s = [], [], [], [], []
        for corpus in ["full", "strict_489"]:
            for protocol in ["random_80_20_SMOTE", "group_shuffle_genus_SMOTE_train"]:
                t = summary[(summary.feature == triv_name) & (summary.corpus == corpus)
                            & (summary.protocol == protocol)]
                r = ref[(ref.feature == ref_name) & (ref.corpus == corpus)
                        & (ref.protocol == protocol)]
                if len(t) and len(r):
                    labels.append(f"{corpus}\n{'aléatoire' if 'random' in protocol else 'genre'}")
                    triv_m.append(float(t.auc_mean.iloc[0])); triv_s.append(float(t.auc_sd.iloc[0]))
                    ref_m.append(float(r.auc_mean.iloc[0])); ref_s.append(float(r.auc_sd.iloc[0]))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, triv_m, width=0.4, yerr=triv_s, capsize=3, color="#8B4513",
               label="trivial longueur+AAC")
        ax.bar(x + 0.2, ref_m, width=0.4, yerr=ref_s, capsize=3, color="#1f4e79",
               label="CKSAAP")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0.4, 1.02); ax.set_title(short)
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 6b — Baseline triviale contre CKSAAP (même protocole)")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase6b_trivial_vs_cksaap.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    def get(feature, corpus, protocol):
        s = summary[(summary.feature == feature) & (summary.corpus == corpus)
                    & (summary.protocol == protocol)]
        return float(s.auc_mean.iloc[0]) if len(s) else float("nan")

    lines = [
        "PHASE 6b — BASELINES TRIVIALES RÉPÉTÉES (longueur + composition AAC)",
        "HGB, SMOTE train uniquement, 20 graines, descripteurs = 21 nombres.",
        "",
        "AUC moyenne (20 splits) :",
    ]
    for feature in ["Nif_trivial_len_AAC", "RecA_trivial_len_AAC"]:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {feature:22s} {corpus:11s} aléatoire={get(feature, corpus, 'random_80_20_SMOTE'):.3f}"
                f" | genre={get(feature, corpus, 'group_shuffle_genus_SMOTE_train'):.3f}"
            )
    lines += [
        "",
        "LECTURE :",
        "- Comparer chaque valeur à la CKSAAP de la même protéine (Phase 5) :",
        "  si la baseline triviale suit CKSAAP de près, l'ordre local des acides",
        "  aminés (CKSAAP) n'apporte qu'un supplément limité sur cette tâche.",
        "- GC % génomique : non inclus ici ; extraction depuis les métadonnées",
        "  GTDB des génomes représentatifs appariés (Phase 6) en cours, à ajouter",
        "  comme variante GC+longueur+AAC dès disponibilité.",
        "",
        f"Durée totale : {time.time()-t0:.0f}s",
        f"Par répétition : {OUT_METRICS / 'phase6b_trivial_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase6b_trivial_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE6B_TRIVIAL_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
