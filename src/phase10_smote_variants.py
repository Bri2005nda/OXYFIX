# -*- coding: utf-8 -*-
"""
Phase 10b — Variantes de rééchantillonnage (justifier ou abandonner SMOTE).

Même protocole que la Phase 5 (HGB, Full/Strict, aléatoire/genre,
20 graines, Nif et RecA), quatre traitements de la classe minoritaire :
- smote        : SMOTE sur le train uniquement (référence Phase 5)
- none         : aucun rééchantillonnage
- class_weight : HGB class_weight='balanced' (pondération des classes)
- xgb_spw      : XGBoost scale_pos_weight = négatifs/positifs du train

Sorties :
- results/metrics/phase10_smote_variants_per_repeat.csv
- results/metrics/phase10_smote_variants_summary.csv
- results/figures/phase10_smote_variants.png
- results/metrics/PHASE10_SMOTE_VARIANTS_DECISION.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
SEEDS = list(range(42, 62))
VARIANTS = ["smote", "none", "class_weight", "xgb_spw"]


def main():
    t0 = time.time()
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
    from xgboost import XGBClassifier

    from phase1_premium_complete import load_control_cksaap, load_nif_cksaap

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    p3 = pd.read_csv(P3)
    strict = set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))

    datasets = {}
    s, X, y, g = load_nif_cksaap()
    datasets["Nif_CKSAAP"] = ([str(v) for v in s], X, y, np.array([str(v).split("_")[0] for v in s]))
    s, X, y, g = load_control_cksaap("RecA")
    datasets["RecA_CKSAAP"] = ([str(v) for v in s], X, y, np.array([str(v).split("_")[0] for v in s]))

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

    def fit_eval(variant, Xtr, Xte, ytr, yte, seed):
        if variant == "xgb_spw":
            n_pos = max(int((ytr == 1).sum()), 1)
            spw = float((ytr == 0).sum()) / n_pos
            model = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
                eval_metric="logloss", random_state=seed, n_jobs=2)
            model.fit(Xtr, ytr)
            return metrics_of(yte, model.predict_proba(Xte)[:, 1])
        if variant == "smote":
            try:
                Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
            except ValueError:
                pass
            model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
        elif variant == "class_weight":
            model = HistGradientBoostingClassifier(random_state=seed, max_depth=6, class_weight="balanced")
        else:
            model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
        model.fit(Xtr, ytr)
        return metrics_of(yte, model.predict_proba(Xte)[:, 1])

    rows = []
    for variant in VARIANTS:
        for feature, (souches, X, y, groups) in datasets.items():
            for corpus, keep in [("full", None), ("strict_489", strict)]:
                if keep is None:
                    idx = np.arange(len(souches))
                else:
                    idx = np.array([i for i, s in enumerate(souches) if s in keep])
                Xs, ys, gs = X[idx], y[idx], groups[idx]
                for seed in SEEDS:
                    strat = ys if min(np.bincount(ys.astype(int))) >= 2 else None
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xs, ys, test_size=0.2, random_state=seed, stratify=strat)
                    rows.append({"variant": variant, "feature": feature, "corpus": corpus,
                                 "protocol": "random_80_20", "seed": seed, "n": int(len(ys)),
                                 **fit_eval(variant, Xtr, Xte, ytr, yte, seed)})
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    tr, te = next(gss.split(Xs, ys, gs))
                    rows.append({"variant": variant, "feature": feature, "corpus": corpus,
                                 "protocol": "group_shuffle_genus", "seed": seed, "n": int(len(ys)),
                                 **fit_eval(variant, Xs[tr], Xs[te], ys[tr], ys[te], seed)})
                print(f"{variant}|{feature}|{corpus} ok ({time.time() - t0:.0f}s)", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT_METRICS / "phase10_smote_variants_per_repeat.csv", index=False)
    summary = (
        per.groupby(["variant", "feature", "corpus", "protocol"])
        .agg(n_repeats=("seed", "count"),
             accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
             auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
             auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
             recall_aerobie_mean=("recall_aerobie", "mean"),
             recall_anaerobie_mean=("recall_anaerobie", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase10_smote_variants_summary.csv", index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = {"smote": "#BC5A28", "none": "#66742E", "class_weight": "#8A6D3B", "xgb_spw": "#5D4A7A"}
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        labels, means, sds, cols = [], [], [], []
        for variant in VARIANTS:
            for protocol, pname in [("random_80_20", "aléat."), ("group_shuffle_genus", "genre")]:
                sub = summary[(summary.variant == variant) & (summary.feature == "Nif_CKSAAP")
                              & (summary.corpus == corpus) & (summary.protocol == protocol)]
                if len(sub):
                    labels.append(f"{variant}\n{pname}")
                    means.append(float(sub.auc_mean.iloc[0]))
                    sds.append(float(sub.auc_sd.iloc[0]))
                    cols.append(colors[variant])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sds, capsize=3, color=cols)
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Nif — corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    fig.suptitle("Phase 10b — Variantes de rééchantillonnage (Nif, CKSAAP)")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase10_smote_variants.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    def get(variant, corpus, protocol, feature="Nif_CKSAAP"):
        s = summary[(summary.variant == variant) & (summary.feature == feature)
                    & (summary.corpus == corpus) & (summary.protocol == protocol)]
        return float(s.auc_mean.iloc[0]) if len(s) else float("nan")

    lines = [
        "PHASE 10b — VARIANTES DE RÉÉCHANTILLONNAGE (20 tirages, Nif et RecA)",
        "",
        "AUC moyenne Nif (aléatoire | genre) :",
    ]
    for variant in VARIANTS:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {variant:13s} {corpus:11s} {get(variant, corpus, 'random_80_20'):.3f}"
                f" | {get(variant, corpus, 'group_shuffle_genus'):.3f}"
            )
    lines += [
        "",
        "LECTURE : si 'none' et 'class_weight' donnent les mêmes AUC que",
        "'smote' (à l'écart-type près), SMOTE n'apporte rien et peut être",
        "abandonné au profit du protocole le plus simple. 'xgb_spw' teste",
        "la pondération scale_pos_weight sous le classifieur d'Alida.",
        "",
        f"Durée totale : {time.time() - t0:.0f}s",
        f"Par répétition : {OUT_METRICS / 'phase10_smote_variants_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase10_smote_variants_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE10_SMOTE_VARIANTS_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
