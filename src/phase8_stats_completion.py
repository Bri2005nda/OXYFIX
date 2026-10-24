# -*- coding: utf-8 -*-
"""
Phase 8 — Complétion statistique (Annexe B, bloc Statistiques) :

A) Intervalles bootstrap 95 % pour TOUS les chiffres phares rapportés :
   bootstrap (10 000 rééchantillonnages) sur les 20 répétitions de la
   Phase 5, pour accuracy, AUC, MCC, AUPRC et les deux rappels, par
   protéine x corpus x protocole. Complète le bootstrap par split de
   l'AUC déjà présent en Phase 5.

B) Test de permutation harmonisé et conscient de la structure :
   - statistique : AUC en validation croisée 3 plis (plis fixes) du
     pipeline StandardScaler + régression logistique (celui du test
     d'origine), appliqué uniformément à Nif et RecA, Full et Strict ;
   - permutation globale des étiquettes (nulle standard) ;
   - permutation par blocs de genres : les profils de classes des
     genres sont permutés entre genres de même taille, ce qui conserve
     la structure taxonomique des étiquettes (tailles de genres et
     distribution des compositions) en détruisant le lien avec les
     séquences. 1 000 permutations par configuration.

C) Validation croisée imbriquée (nested CV) consciente des genres :
   boucle externe GroupKFold(5) par genre (test jamais touché), boucle
   interne GroupKFold(3) pour choisir les hyperparamètres HGB parmi
   4 combinaisons, SMOTE uniquement dans les entraînements (pipeline).
   Protéines Nif et RecA, corpus Full et Strict.

Sorties :
- results/metrics/phase8_bootstrap_all_metrics.csv
- results/metrics/phase8_permutations.csv (+ _scores.npz)
- results/metrics/phase8_nested_cv_outer_folds.csv
- results/metrics/phase8_nested_cv_summary.csv
- results/figures/phase8_permutation_nulls.png
- results/metrics/PHASE8_STATS_DECISION.txt
"""
from __future__ import annotations

import argparse
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
PHASE5_PER = OUT_METRICS / "phase5_repeated_splits_per_repeat.csv"
SEEDS = list(range(42, 62))


# ---------------------------------------------------------------- A) bootstrap
def stage_bootstrap(rng) -> pd.DataFrame:
    per = pd.read_csv(PHASE5_PER)
    metrics = ["accuracy", "auc", "mcc", "auprc", "recall_aerobie", "recall_anaerobie"]
    rows = []
    B = 10000
    for (feature, corpus, protocol), g in per.groupby(["feature", "corpus", "protocol"]):
        row = {"feature": feature, "corpus": corpus, "protocol": protocol, "n_repeats": len(g)}
        vals_all = {m: g[m].dropna().values for m in metrics}
        for m, vals in vals_all.items():
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            if len(vals) >= 5:
                idx = rng.integers(0, len(vals), size=(B, len(vals)))
                means = vals[idx].mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                row[f"{m}_boot_low"] = float(lo)
                row[f"{m}_boot_high"] = float(hi)
            else:
                row[f"{m}_boot_low"] = np.nan
                row[f"{m}_boot_high"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_METRICS / "phase8_bootstrap_all_metrics.csv", index=False)
    print(f"A) bootstrap terminé : {len(df)} configurations", flush=True)
    return df


# ---------------------------------------------------------------- B) permutations
def cv3_auc(X, y, folds, make_clf):
    from sklearn.metrics import roc_auc_score

    aucs = []
    for tr, te in folds:
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aucs))


def block_permute_labels(y, groups, rng):
    """Permute les profils de classes entre genres de même taille."""
    y_new = y.copy()
    genus_info = {}
    for gname in np.unique(groups):
        idx = np.where(groups == gname)[0]
        genus_info[gname] = (idx, int(y[idx].sum()))
    by_size: dict[int, list[str]] = {}
    for gname, (idx, _k) in genus_info.items():
        by_size.setdefault(len(idx), []).append(gname)
    for size, gnames in by_size.items():
        if len(gnames) < 2:
            # genre seul de sa taille : on mélange seulement en interne
            idx = genus_info[gnames[0]][0]
            y_new[idx] = rng.permutation(y_new[idx])
            continue
        ks = np.array([genus_info[g][1] for g in gnames])
        ks_perm = rng.permutation(ks)
        for gname, k_new in zip(gnames, ks_perm):
            idx = genus_info[gname][0]
            labels = np.zeros(len(idx), dtype=int)
            if k_new > 0:
                labels[rng.choice(len(idx), size=int(k_new), replace=False)] = 1
            y_new[idx] = labels
    return y_new


def stage_permutations(n_perm: int, rng) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from phase1_premium_complete import load_control_cksaap, load_nif_cksaap

    p3 = pd.read_csv(P3)
    strict = set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))

    def make_clf():
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=500, solver="lbfgs", random_state=42),
        )

    datasets = {}
    s, X, y, g = load_nif_cksaap()
    datasets["Nif_CKSAAP"] = (s, X, y, g)
    s, X, y, g = load_control_cksaap("RecA")
    datasets["RecA_CKSAAP"] = (s, X, y, g)

    rows = []
    all_scores = {}
    for feature, (souches, X, y, groups) in datasets.items():
        souches = [str(s) for s in souches]
        for corpus, keep in [("full", None), ("strict_489", strict)]:
            if keep is None:
                idx = np.arange(len(souches))
            else:
                idx = np.array([i for i, s in enumerate(souches) if s in keep])
            Xs, ys, gs = X[idx], y[idx], np.asarray(groups)[idx]
            skf = StratifiedKFold(n_splits=3, shuffle=False)
            folds = list(skf.split(Xs, ys))
            true_auc = cv3_auc(Xs, ys, folds, make_clf)
            for null_name in ["global", "genus_block"]:
                scores = np.empty(n_perm)
                for b in range(n_perm):
                    if null_name == "global":
                        y_perm = rng.permutation(ys)
                    else:
                        y_perm = block_permute_labels(ys, gs, rng)
                    scores[b] = cv3_auc(Xs, y_perm, folds, make_clf)
                    if (b + 1) % 100 == 0:
                        print(f"  perm {feature}|{corpus}|{null_name} {b + 1}/{n_perm}", flush=True)
                pval = float((np.sum(scores >= true_auc) + 1) / (n_perm + 1))
                rows.append({
                    "feature": feature, "corpus": corpus, "null": null_name,
                    "true_auc_cv3": true_auc,
                    "perm_auc_mean": float(np.mean(scores)),
                    "perm_auc_sd": float(np.std(scores)),
                    "perm_auc_p95": float(np.percentile(scores, 95)),
                    "pvalue": pval, "n_permutations": n_perm,
                })
                all_scores[f"{feature}|{corpus}|{null_name}"] = scores
                print(f"B) {feature}|{corpus}|{null_name}: vraie={true_auc:.3f} "
                      f"nulle={np.mean(scores):.3f} p={pval:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_METRICS / "phase8_permutations.csv", index=False)
    np.savez_compressed(OUT_METRICS / "phase8_permutation_scores.npz", **all_scores)
    return df


# ---------------------------------------------------------------- C) nested CV
def stage_nested() -> tuple[pd.DataFrame, pd.DataFrame]:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        matthews_corrcoef,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GridSearchCV, GroupKFold

    from phase1_premium_complete import load_control_cksaap, load_nif_cksaap

    p3 = pd.read_csv(P3)
    strict = set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))
    datasets = {}
    s, X, y, g = load_nif_cksaap()
    datasets["Nif_CKSAAP"] = (s, X, y, g)
    s, X, y, g = load_control_cksaap("RecA")
    datasets["RecA_CKSAAP"] = (s, X, y, g)

    param_grid = {
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.05, 0.1],
    }

    def make_pipe(seed=42):
        return ImbPipeline([
            ("smote", SMOTE(random_state=seed)),
            ("clf", HistGradientBoostingClassifier(random_state=seed)),
        ])

    rows = []
    for feature, (souches, X, y, groups) in datasets.items():
        souches = [str(s) for s in souches]
        groups = np.asarray(groups)
        for corpus, keep in [("full", None), ("strict_489", strict)]:
            if keep is None:
                idx = np.arange(len(souches))
            else:
                idx = np.array([i for i, s in enumerate(souches) if s in keep])
            Xs, ys, gs = X[idx], y[idx], groups[idx]
            outer = GroupKFold(n_splits=5)
            for fold, (tr, te) in enumerate(outer.split(Xs, ys, gs)):
                inner = GroupKFold(n_splits=3)
                search = GridSearchCV(
                    make_pipe(), param_grid, scoring="roc_auc",
                    cv=inner, n_jobs=1, refit=True,
                )
                search.fit(Xs[tr], ys[tr], groups=gs[tr])
                proba = search.predict_proba(Xs[te])[:, 1]
                pred = (proba >= 0.5).astype(int)
                both = len(np.unique(ys[te])) > 1
                rows.append({
                    "feature": feature, "corpus": corpus, "outer_fold": fold,
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "n_test_aerobie": int((ys[te] == 1).sum()),
                    "n_test_anaerobie": int((ys[te] == 0).sum()),
                    "genres_overlap": int(len(set(gs[tr]) & set(gs[te]))),
                    "best_params": str(search.best_params_),
                    "inner_best_auc": float(search.best_score_),
                    "accuracy": float(accuracy_score(ys[te], pred)),
                    "auc": float(roc_auc_score(ys[te], proba)) if both else np.nan,
                    "mcc": float(matthews_corrcoef(ys[te], pred)) if both else np.nan,
                    "auprc": float(average_precision_score(ys[te], proba)) if both else np.nan,
                    "recall_aerobie": float(recall_score(ys[te], pred, pos_label=1, zero_division=0)),
                    "recall_anaerobie": float(recall_score(ys[te], pred, pos_label=0, zero_division=0)),
                })
                print(f"C) nested {feature}|{corpus} pli externe {fold + 1}/5 "
                      f"AUC={rows[-1]['auc']:.3f} params={search.best_params_}", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT_METRICS / "phase8_nested_cv_outer_folds.csv", index=False)
    summary = (
        folds.groupby(["feature", "corpus"])
        .agg(n_outer_folds=("outer_fold", "count"),
             accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
             auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
             auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"))
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase8_nested_cv_summary.csv", index=False)
    return folds, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["bootstrap", "permutations", "nested", "all"], default="all")
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()
    t0 = time.time()
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)

    boot = perm = nested_sum = None
    if args.stage in ("bootstrap", "all"):
        boot = stage_bootstrap(rng)
    if args.stage in ("permutations", "all"):
        perm = stage_permutations(args.n_perm, rng)
    if args.stage in ("nested", "all"):
        _folds, nested_sum = stage_nested()

    if perm is not None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        scores = np.load(OUT_METRICS / "phase8_permutation_scores.npz")
        combos = [("Nif_CKSAAP", "full"), ("RecA_CKSAAP", "full"),
                  ("Nif_CKSAAP", "strict_489"), ("RecA_CKSAAP", "strict_489")]
        for ax, (feature, corpus) in zip(axes.ravel(), combos):
            for null_name, color in [("global", "#BC5A28"), ("genus_block", "#66742E")]:
                key = f"{feature}|{corpus}|{null_name}"
                if key in scores:
                    ax.hist(scores[key], bins=30, alpha=0.55, color=color, label=null_name)
            row = perm[(perm.feature == feature) & (perm.corpus == corpus)]
            if len(row):
                ax.axvline(float(row.true_auc_cv3.iloc[0]), color="#3E2723", lw=2, label="vraie AUC")
            ax.axvline(0.5, color="black", ls="--", lw=1)
            ax.set_title(f"{feature} — {corpus}", fontsize=10)
            ax.set_xlabel("AUC CV3 sous étiquettes permutées")
            ax.set_ylabel("permutations")
            ax.legend(fontsize=7)
        fig.suptitle("Phase 8 — Distributions nulles de permutation (globale et par blocs de genres)")
        fig.tight_layout()
        fig.savefig(OUT_FIG / "phase8_permutation_nulls.png", dpi=160)
        plt.close()

    lines = ["PHASE 8 — COMPLÉTION STATISTIQUE (bootstrap, permutations, nested CV)", ""]
    if boot is not None:
        lines.append(f"A) Bootstrap 10 000 sur les 20 répétitions Phase 5 : {len(boot)} configurations "
                     "(accuracy, AUC, MCC, AUPRC, rappels) -> phase8_bootstrap_all_metrics.csv")
    if perm is not None:
        lines.append("B) Permutations (statistique LR CV3, plis fixes) :")
        for _, r in perm.iterrows():
            lines.append(f"   {r.feature} {r.corpus} nulle={r.null}: vraie AUC={r.true_auc_cv3:.3f}, "
                         f"moyenne nulle={r.perm_auc_mean:.3f}, p={r.pvalue:.4f} (n={r.n_permutations})")
        lines.append("   Lecture : la nulle par blocs de genres conserve la structure taxonomique des "
                     "étiquettes ; un p faible sous cette nulle ne démontre toujours pas un mécanisme "
                     "(le contrôle RecA est soumis au même test).")
    if nested_sum is not None:
        lines.append("C) Validation imbriquée (externe genre GroupKFold 5, interne GroupKFold 3) :")
        for _, r in nested_sum.iterrows():
            lines.append(f"   {r.feature} {r.corpus}: AUC externe={r.auc_mean:.3f} ± {r.auc_sd:.3f}, "
                         f"MCC={r.mcc_mean:.3f}, AUPRC={r.auprc_mean:.3f}")
        lines.append("   Le test externe n'est touché qu'après sélection interne des hyperparamètres.")
    lines += ["", f"Durée totale : {time.time() - t0:.0f}s"]
    (OUT_METRICS / "PHASE8_STATS_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
