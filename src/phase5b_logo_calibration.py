# -*- coding: utf-8 -*-
"""
Phase 5b — Reste de l'etape stats (Annexe B) :
1) LOGO par genre (Full + Strict, Nif/RecA) : AUC par genre, moyenne sur genres
   + bootstrap sur les genres, et AUC poolee sur toutes les predictions LOGO.
2) Courbes de calibration : predictions hors-echantillon poolees sur 5 splits
   (graines 42..46), HGB, random + genre, Full + Strict, Nif/RecA.
3) Test de permutation : deja realise (Phase 1 premium, 1 000 permutations,
   regression logistique CV3, Nif/RecA Full) -> repris dans la decision.

Sorties :
- results/metrics/phase5b_logo_per_fold.csv
- results/metrics/phase5b_logo_predictions.csv
- results/metrics/phase5b_logo_summary.csv
- results/metrics/phase5b_calibration_bins.csv
- results/metrics/phase5b_calibration_summary.csv
- results/figures/phase5b_logo_auc_distribution.png
- results/figures/phase5b_calibration.png
- results/metrics/PHASE5B_LOGO_CALIBRATION_DECISION.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import load_control_cksaap, load_nif_cksaap  # noqa: E402
from phase5_repeated_splits_stats import strict_keep, filter_to  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
SEED = 42
CAL_SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 1000


def fit_proba(Xtr, Xte, ytr, seed=SEED):
    try:
        Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    return model.predict_proba(Xte)[:, 1]


def base_metrics(yte, proba):
    pred = (proba >= 0.5).astype(int)
    both = len(np.unique(yte)) > 1
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, proba)) if both else np.nan,
        "mcc": float(matthews_corrcoef(yte, pred)) if both else np.nan,
        "auprc": float(average_precision_score(yte, proba)) if both else np.nan,
        "n_test": int(len(yte)),
        "n_test_aerobie": int((yte == 1).sum()),
        "n_test_anaerobie": int((yte == 0).sum()),
    }


def bootstrap_mean_ci(values, seed=7):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(vals), len(vals))
        boots.append(float(np.mean(vals[idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_pooled_auc_ci(y_true, proba, seed=11):
    pos = proba[y_true == 1]
    neg = proba[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    comp = (pos[:, None] > neg[None, :]).astype(np.float64) + 0.5 * (
        pos[:, None] == neg[None, :]
    )
    rng = np.random.default_rng(seed)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pi = rng.integers(0, len(pos), len(pos))
        ni = rng.integers(0, len(neg), len(neg))
        vals[b] = comp[np.ix_(pi, ni)].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------- LOGO ----------------

def logo_fold_list(X, y, groups):
    folds = []
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        genus = str(groups[te][0])
        if len(te) < 3:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        folds.append((genus, tr, te))
    return folds


def process_logo_folds(souches, X, y, groups, folds, feature, corpus):
    fold_rows, pred_rows = [], []
    for genus, tr, te in folds:
        proba = fit_proba(X[tr], X[te], y[tr])
        m = base_metrics(y[te], proba)
        fold_rows.append(
            {"feature": feature, "corpus": corpus, "genus_test": genus, **m}
        )
        for s, yt, pr in zip([souches[i] for i in te], y[te], proba):
            pred_rows.append(
                {
                    "feature": feature,
                    "corpus": corpus,
                    "genus_test": genus,
                    "souche": s,
                    "y_true": int(yt),
                    "proba": float(pr),
                }
            )
        print(f"  LOGO [{feature}|{corpus}] genus={genus} n={len(te)}", flush=True)
    return fold_rows, pred_rows


def run_logo(datasets, keep):
    all_folds, all_preds = [], []
    jobs = []
    metas = []
    for feature, souches, X, y, groups in datasets:
        if feature not in ("Nif_CKSAAP", "RecA_CKSAAP"):
            continue
        s_s, X_s, y_s, g_s = filter_to(souches, X, y, groups, keep)
        for corpus, (ss, Xs, ys, gs) in {
            "full": (souches, X, y, groups),
            "strict_489": (s_s, X_s, y_s, g_s),
        }.items():
            folds = logo_fold_list(Xs, ys, gs)
            n_genera = int(len(np.unique(gs)))
            metas.append((feature, corpus, n_genera, len(folds)))
            print(
                f"LOGO {feature}|{corpus}: genres totaux={n_genera}, "
                f"folds retenus (test>=3)={len(folds)}",
                flush=True,
            )
            chunks = [folds[i::4] for i in range(4) if folds[i::4]]
            for ch in chunks:
                jobs.append((feature, corpus, ss, Xs, ys, gs, ch))
    results = Parallel(n_jobs=4)(
        delayed(process_logo_folds)(ss, Xs, ys, gs, ch, feat, corp)
        for (feat, corp, ss, Xs, ys, gs, ch) in jobs
    )
    for fr, pr in results:
        all_folds += fr
        all_preds += pr
    return pd.DataFrame(all_folds), pd.DataFrame(all_preds), metas


def summarise_logo(folds, preds, metas):
    out = []
    meta_map = {(f, c): (ng, nf) for f, c, ng, nf in metas}
    for (feat, corpus), sub in folds.groupby(["feature", "corpus"]):
        mixed = sub.dropna(subset=["auc"])
        lo, hi = bootstrap_mean_ci(mixed["auc"].to_numpy())
        pr = preds[(preds["feature"] == feat) & (preds["corpus"] == corpus)]
        yt = pr["y_true"].to_numpy()
        pb = pr["proba"].to_numpy()
        pooled = base_metrics(yt, pb)
        plo, phi = bootstrap_pooled_auc_ci(yt, pb)
        ng, nf = meta_map.get((feat, corpus), (np.nan, np.nan))
        out.append(
            {
                "feature": feat,
                "corpus": corpus,
                "n_genera_total": ng,
                "n_folds_test_ge3": nf,
                "n_folds_mixed_auc": int(len(mixed)),
                "auc_mean_over_genera": float(mixed["auc"].mean()) if len(mixed) else np.nan,
                "auc_sd_over_genera": float(mixed["auc"].std(ddof=1)) if len(mixed) > 1 else np.nan,
                "auc_median_over_genera": float(mixed["auc"].median()) if len(mixed) else np.nan,
                "auc_mean_boot_low": lo,
                "auc_mean_boot_high": hi,
                "auc_pooled": pooled["auc"],
                "auc_pooled_boot_low": plo,
                "auc_pooled_boot_high": phi,
                "accuracy_pooled": pooled["accuracy"],
                "mcc_pooled": pooled["mcc"],
                "auprc_pooled": pooled["auprc"],
                "n_predictions": int(len(pr)),
            }
        )
    return pd.DataFrame(out).sort_values(["corpus", "feature"])


# ---------------- Calibration ----------------

def cal_one(souches, X, y, groups, feature, corpus, protocol, seed):
    if protocol == "random":
        strat = y if min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xte, ytr, yte, str_tr, str_te = train_test_split(
            X, y, souches, test_size=0.2, random_state=seed, stratify=strat
        )
        g_te = None
        # groups pour info : retrouver via souches
        gmap = {s: g for s, g in zip(souches, groups)}
        g_te = [gmap[s] for s in str_te]
        proba = fit_proba(Xtr, Xte, ytr, seed)
        rows = [
            {
                "feature": feature,
                "corpus": corpus,
                "protocol": protocol,
                "seed": seed,
                "souche": s,
                "genus": g,
                "y_true": int(yt),
                "proba": float(p),
            }
            for s, g, yt, p in zip(str_te, g_te, yte, proba)
        ]
        return rows
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    proba = fit_proba(X[tr], X[te], y[tr], seed)
    return [
        {
            "feature": feature,
            "corpus": corpus,
            "protocol": protocol,
            "seed": seed,
            "souche": souches[i],
            "genus": str(groups[i]),
            "y_true": int(y[i]),
            "proba": float(p),
        }
        for i, p in zip(te, proba)
    ]


def calibration_bins(df, n_bins=10):
    bins_rows, summ_rows = [], []
    for (feat, corpus, proto), sub in df.groupby(["feature", "corpus", "protocol"]):
        yt = sub["y_true"].to_numpy()
        pb = sub["proba"].to_numpy()
        # bins quantile (bords uniques)
        edges = np.unique(np.quantile(pb, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(pb, edges[1:-1], right=False), 0, len(edges) - 2)
        ece = 0.0
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() == 0:
                continue
            mean_pred = float(pb[m].mean())
            frac_pos = float(yt[m].mean())
            bins_rows.append(
                {
                    "feature": feat,
                    "corpus": corpus,
                    "protocol": proto,
                    "bin": int(b),
                    "n": int(m.sum()),
                    "mean_proba": mean_pred,
                    "frac_aerobie": frac_pos,
                }
            )
            ece += (m.sum() / len(pb)) * abs(frac_pos - mean_pred)
        both = len(np.unique(yt)) > 1
        summ_rows.append(
            {
                "feature": feat,
                "corpus": corpus,
                "protocol": proto,
                "n_predictions_pooled": int(len(sub)),
                "brier": float(brier_score_loss(yt, pb)),
                "log_loss": float(log_loss(yt, np.clip(pb, 1e-6, 1 - 1e-6))),
                "ece_quantile10": float(ece),
                "auc_pooled_calibration_runs": float(roc_auc_score(yt, pb)) if both else np.nan,
            }
        )
    return pd.DataFrame(bins_rows), pd.DataFrame(summ_rows)


def run_calibration(datasets, keep):
    jobs = []
    for feature, souches, X, y, groups in datasets:
        if feature not in ("Nif_CKSAAP", "RecA_CKSAAP"):
            continue
        s_s, X_s, y_s, g_s = filter_to(souches, X, y, groups, keep)
        for corpus, (ss, Xs, ys, gs) in {
            "full": (souches, X, y, groups),
            "strict_489": (s_s, X_s, y_s, g_s),
        }.items():
            for proto in ("random", "group_genus"):
                proto_key = "random" if proto == "random" else "genus"
                for seed in CAL_SEEDS:
                    jobs.append((feature, corpus, proto_key, seed, ss, Xs, ys, gs))
    results = Parallel(n_jobs=4)(
        delayed(cal_one)(ss, Xs, ys, gs, feat, corp, proto, seed)
        for (feat, corp, proto, seed, ss, Xs, ys, gs) in jobs
    )
    rows = [r for part in results for r in part]
    df = pd.DataFrame(rows)
    # normaliser le nom de protocole
    df["protocol"] = df["protocol"].replace(
        {"genus": "group_shuffle_genus_SMOTE_train", "random": "random_80_20_SMOTE"}
    )
    return df


# ---------------- Figures ----------------

def fig_logo(folds, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        sub = folds[folds["corpus"] == corpus]
        for feat, color in [("Nif_CKSAAP", "#1f4e79"), ("RecA_CKSAAP", "#8b6914")]:
            vals = sub[(sub["feature"] == feat)]["auc"].dropna()
            if len(vals):
                ax.hist(vals, bins=12, alpha=0.55, color=color, label=f"{feat.split('_')[0]} (n={len(vals)})")
        ax.axvline(0.5, color="red", ls="--", lw=1, label="Hasard 0,5")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("AUC par genre laisse de cote (folds mixtes uniquement)")
        ax.set_ylabel("Nombre de genres")
        ax.set_title(f"LOGO — distribution des AUC par genre ({corpus})")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Phase 5b — Leave-One-Genus-Out : AUC par genre teste", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = OUT_FIG / "phase5b_logo_auc_distribution.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_calibration(bins):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    protos = [("random_80_20_SMOTE", "Aleatoire"), ("group_shuffle_genus_SMOTE_train", "Split genre")]
    corpora = [("full", "Full"), ("strict_489", "Strict 489")]
    for pi, (proto, plab) in enumerate(protos):
        for ci, (corpus, clab) in enumerate(corpora):
            ax = axes[pi][ci]
            ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Parfaite")
            for feat, color in [("Nif_CKSAAP", "#1f4e79"), ("RecA_CKSAAP", "#8b6914")]:
                sub = bins[
                    (bins["feature"] == feat)
                    & (bins["corpus"] == corpus)
                    & (bins["protocol"] == proto)
                ].sort_values("mean_proba")
                if len(sub):
                    ax.plot(
                        sub["mean_proba"],
                        sub["frac_aerobie"],
                        marker="o",
                        color=color,
                        label=feat.split("_")[0],
                    )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Probabilite predite moyenne (bin)")
            ax.set_ylabel("Fraction aerobie observee")
            ax.set_title(f"{clab} — {plab}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Phase 5b — Calibration (predictions poolees, 5 splits, HGB)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    p = OUT_FIG / "phase5b_calibration.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    keep = strict_keep()
    datasets = [
        ("Nif_CKSAAP",) + load_nif_cksaap(),
        ("RecA_CKSAAP",) + load_control_cksaap("RecA"),
    ]

    print("=== LOGO ===", flush=True)
    folds, preds, metas = run_logo(datasets, keep)
    folds_path = OUT_METRICS / "phase5b_logo_per_fold.csv"
    preds_path = OUT_METRICS / "phase5b_logo_predictions.csv"
    folds.to_csv(folds_path, index=False)
    preds.to_csv(preds_path, index=False)
    logo_summary = summarise_logo(folds, preds, metas)
    logo_sum_path = OUT_METRICS / "phase5b_logo_summary.csv"
    logo_summary.to_csv(logo_sum_path, index=False)
    print(logo_summary.to_string(index=False), flush=True)

    print("=== Calibration ===", flush=True)
    cal_df = run_calibration(datasets, keep)
    cal_pred_path = OUT_METRICS / "phase5b_calibration_predictions.csv"
    cal_df.to_csv(cal_pred_path, index=False)
    bins, cal_summ = calibration_bins(cal_df)
    bins_path = OUT_METRICS / "phase5b_calibration_bins.csv"
    cal_sum_path = OUT_METRICS / "phase5b_calibration_summary.csv"
    bins.to_csv(bins_path, index=False)
    cal_summ.to_csv(cal_sum_path, index=False)
    print(cal_summ.to_string(index=False), flush=True)

    p1 = fig_logo(folds, logo_summary)
    p2 = fig_calibration(bins)

    # Permutation deja realisee
    perm_note = "non retrouve"
    perm_path = OUT_METRICS / "phase1_premium_permutations.csv"
    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        perm_note = " ; ".join(
            f"{r['name']}: vraie AUC(CV3)={r['true_auc_cv3']:.3f}, "
            f"perm. moyenne={r['perm_auc_mean']:.3f}, p={r['pvalue']:.4f} "
            f"(n={int(r['n_permutations'])})"
            for _, r in perm.iterrows()
        )

    lines = [
        "PHASE 5b — LOGO PAR GENRE + CALIBRATION (+ permutation deja acquise)",
        "HGB, CKSAAP, SMOTE train uniquement.",
        "",
        "LOGO (moyenne sur genres mixtes + AUC poolee) :",
    ]
    for _, r in logo_summary.iterrows():
        lines.append(
            f"  {r['feature']:12s} {r['corpus']:11s} genres={int(r['n_genera_total'])} "
            f"folds>=3={int(r['n_folds_test_ge3'])} folds mixtes={int(r['n_folds_mixed_auc'])} | "
            f"AUC moyenne/genres={r['auc_mean_over_genera']:.3f} "
            f"(bootstrap IC95 [{r['auc_mean_boot_low']:.3f};{r['auc_mean_boot_high']:.3f}]) | "
            f"AUC poolee={r['auc_pooled']:.3f} "
            f"(IC95 [{r['auc_pooled_boot_low']:.3f};{r['auc_pooled_boot_high']:.3f}])"
        )
    lines += [
        "",
        "ATTENTION LOGO :",
        "- Le chiffre du manuscrit (Nif ~0,549 ; RecA ~0,560) est la MOYENNE des AUC",
        "  par genre, calculable uniquement sur les genres mixtes (aerobie+anaerobie).",
        "  Sur Full, il n'y en a que 8 (sur 69 folds test>=3) ; sur Strict, 0.",
        "  L'intervalle bootstrap est donc tres large : ce resultat est fragile.",
        "- L'AUC POOLEE LOGO (predictions de modeles differents mises bout a bout)",
        "  est elevee (~0,97) mais NE doit PAS etre presentee comme performance LOGO :",
        "  elle compare surtout des souches de genres mono-classe differents, avec",
        "  des probabilites issues de modeles differents. A ne pas reporter telle quelle.",
        "",
        "Calibration (5 splits pooles ; Brier bas = meilleur, ECE bas = meilleur) : voir",
        f"  {cal_sum_path}",
        "",
        "Permutation (deja realisee, Phase 1 premium, regression logistique CV3) :",
        f"  {perm_note}",
        "  Lecture : p~0,001 pour Nif ET RecA = significatif ne veut pas dire mecanistique.",
        "",
        f"Duree : {time.time()-t0:.0f} s",
        f"LOGO folds      : {folds_path}",
        f"LOGO predictions: {preds_path}",
        f"LOGO synthese   : {logo_sum_path}",
        f"Calibration bins: {bins_path}",
        f"Figures         : {p1} ; {p2}",
    ]
    dec = OUT_METRICS / "PHASE5B_LOGO_CALIBRATION_DECISION.txt"
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nOK -> {dec}", flush=True)


if __name__ == "__main__":
    main()
