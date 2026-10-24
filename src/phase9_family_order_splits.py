# -*- coding: utf-8 -*-
"""
Phase 9 — Splits par famille et par ordre (durcissement du protocole).

Prolonge la courbe de décroissance au-delà du genre : GroupShuffleSplit
sur les groupes famille puis ordre (table_taxonomy_levels.csv), 20
graines, HGB + CKSAAP, SMOTE train uniquement, corpus Full/Strict,
protéines Nif et RecA. Pour chaque répétition, la composition du test
est rapportée (nombre de groupes retirés, part de groupes mono-classe
parmi les groupes du corpus, effectifs par classe) — la preuve que le
découpage est dur.

Sorties :
- results/metrics/phase9_family_order_per_repeat.csv
- results/metrics/phase9_family_order_summary.csv
- results/figures/phase9_family_order_decay.png
- results/metrics/PHASE9_FAMILY_ORDER_DECISION.txt
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
TAXO = ROOT / "data" / "raw" / "metadata" / "table_taxonomy_levels.csv"
SEEDS = list(range(42, 62))


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
    from sklearn.model_selection import GroupShuffleSplit

    from phase1_premium_complete import load_control_cksaap, load_nif_cksaap

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    p3 = pd.read_csv(P3)
    strict = set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))
    taxo = pd.read_csv(TAXO).set_index("souche")

    datasets = {}
    s, X, y, g = load_nif_cksaap()
    datasets["Nif_CKSAAP"] = (s, X, y)
    s, X, y, g = load_control_cksaap("RecA")
    datasets["RecA_CKSAAP"] = (s, X, y)

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

    rows = []
    for feature, (souches, X, y) in datasets.items():
        souches = [str(s) for s in souches]
        fam = np.array([taxo["family"].get(s, "") if s in taxo.index else "" for s in souches], dtype=object)
        order = np.array([taxo["order"].get(s, "") if s in taxo.index else "" for s in souches], dtype=object)
        for level_name, level_groups in [("family", fam), ("order", order)]:
            known = np.array([bool(v) and str(v) != "nan" for v in level_groups])
            for corpus, keep in [("full", None), ("strict_489", strict)]:
                if keep is None:
                    idx = np.where(known)[0]
                else:
                    idx = np.array([i for i, s in enumerate(souches) if s in keep and known[i]])
                Xs, ys, gs = X[idx], y[idx], np.asarray(level_groups)[idx]
                # part de groupes mono-classe dans le corpus (structure)
                df_g = pd.DataFrame({"g": gs, "y": ys}).groupby("g")["y"].agg(["count", "mean"])
                mono_share = float(((df_g["mean"] == 0) | (df_g["mean"] == 1)).mean())
                for seed in SEEDS:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    tr, te = next(gss.split(Xs, ys, gs))
                    m = fit_eval(Xs[tr], Xs[te], ys[tr], ys[te], seed)
                    test_groups = pd.DataFrame({"g": gs[te], "y": ys[te]}).groupby("g")["y"].agg(["count", "mean"])
                    rows.append({
                        "feature": feature, "corpus": corpus,
                        "level": level_name, "seed": seed, "n": int(len(ys)),
                        "n_groups_corpus": int(len(df_g)),
                        "mono_class_group_share_corpus": mono_share,
                        "n_groups_test": int(len(test_groups)),
                        "n_groups_test_mixed": int(((test_groups["mean"] > 0) & (test_groups["mean"] < 1)).sum()),
                        "n_test": int(len(te)),
                        "n_test_aerobie": int((ys[te] == 1).sum()),
                        "n_test_anaerobie": int((ys[te] == 0).sum()),
                        "genres_overlap": int(len(set(gs[tr]) & set(gs[te]))),
                        **m,
                    })
                print(f"{feature}|{corpus}|{level_name} ok ({time.time() - t0:.0f}s)", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT_METRICS / "phase9_family_order_per_repeat.csv", index=False)
    summary = (
        per.groupby(["feature", "corpus", "level"])
        .agg(n_repeats=("seed", "count"), n_mean=("n", "mean"),
             mono_share_mean=("mono_class_group_share_corpus", "mean"),
             accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
             auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
             auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
             recall_aerobie_mean=("recall_aerobie", "mean"),
             recall_anaerobie_mean=("recall_anaerobie", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase9_family_order_summary.csv", index=False)

    import matplotlib.pyplot as plt

    ref = pd.read_csv(OUT_METRICS / "phase5_repeated_splits_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        labels = ["aléatoire", "genre", "famille", "ordre"]
        nif_m, nif_s, rec_m, rec_s = [], [], [], []
        for lab in labels:
            if lab in ("aléatoire", "genre"):
                proto = "random_80_20_SMOTE" if lab == "aléatoire" else "group_shuffle_genus_SMOTE_train"
                rn = ref[(ref.feature == "Nif_CKSAAP") & (ref.corpus == corpus) & (ref.protocol == proto)]
                rr = ref[(ref.feature == "RecA_CKSAAP") & (ref.corpus == corpus) & (ref.protocol == proto)]
                nif_m.append(float(rn.auc_mean.iloc[0])); nif_s.append(float(rn.auc_sd.iloc[0]))
                rec_m.append(float(rr.auc_mean.iloc[0])); rec_s.append(float(rr.auc_sd.iloc[0]))
            else:
                lvl = "family" if lab == "famille" else "order"
                sn = summary[(summary.feature == "Nif_CKSAAP") & (summary.corpus == corpus) & (summary.level == lvl)]
                sr = summary[(summary.feature == "RecA_CKSAAP") & (summary.corpus == corpus) & (summary.level == lvl)]
                nif_m.append(float(sn.auc_mean.iloc[0]) if len(sn) else np.nan)
                nif_s.append(float(sn.auc_sd.iloc[0]) if len(sn) else 0.0)
                rec_m.append(float(sr.auc_mean.iloc[0]) if len(sr) else np.nan)
                rec_s.append(float(sr.auc_sd.iloc[0]) if len(sr) else 0.0)
        x = np.arange(len(labels))
        ax.errorbar(x - 0.07, nif_m, yerr=nif_s, marker="o", capsize=3, color="#BC5A28", label="Nif")
        ax.errorbar(x + 0.07, rec_m, yerr=rec_s, marker="s", capsize=3, color="#66742E", label="RecA")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 9 — Décroissance prolongée : aléatoire, genre, famille, ordre")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase9_family_order_decay.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    def get(feature, corpus, level):
        s = summary[(summary.feature == feature) & (summary.corpus == corpus) & (summary.level == level)]
        return float(s.auc_mean.iloc[0]) if len(s) else float("nan")

    lines = [
        "PHASE 9 — SPLITS PAR FAMILLE ET PAR ORDRE (20 tirages, composition rapportée)",
        "",
        "AUC moyenne (20 splits) :",
    ]
    for feature in ["Nif_CKSAAP", "RecA_CKSAAP"]:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {feature:12s} {corpus:11s} famille={get(feature, corpus, 'family'):.3f}"
                f" | ordre={get(feature, corpus, 'order'):.3f}"
            )
    lines += [
        "",
        "LECTURE : prolonge la courbe de décroissance au-delà du genre.",
        "La part de groupes mono-classe et la composition de chaque test sont",
        "dans phase9_family_order_per_repeat.csv (preuve de dureté des splits).",
        "Taxonomie : GTDB via représentant apparié, consensus de genre sinon",
        "(data/raw/metadata/table_taxonomy_levels.csv).",
        "",
        f"Durée totale : {time.time() - t0:.0f}s",
        f"Par répétition : {OUT_METRICS / 'phase9_family_order_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase9_family_order_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE9_FAMILY_ORDER_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
