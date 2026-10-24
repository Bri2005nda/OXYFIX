# -*- coding: utf-8 -*-
"""
Phase 5 (Etape 1 manuscrit) — Splits repetes + intervalles + MCC/AUPRC

Demande Annexe B du manuscrit :
- repeter chaque decoupage >= 20 fois ; rapporter moyenne +/- ecart-type
- intervalles de confiance bootstrap 95 % (1 000 reechantillonnages) sur l'AUC
- ajouter MCC et AUPRC a cote de l'AUC et de l'accuracy

Perimetre de cette etape :
- Classifieur HGB (classifieur des chiffres phares), descripteurs CKSAAP
- Proteines : Nif (cible), RecA / GyrB / RpoB (controles negatifs)
- Corpus : Full (~787) et Strict 489
- Protocoles : aleatoire 80/20 stratifie, split genre (GroupShuffleSplit)
- 20 graines (42..61), SMOTE sur le train uniquement
- LOGO Full (Nif/RecA) : 1 passage deterministe par genre + bootstrap sur les folds
  (le decoupage LOGO n'est pas aleatoire ; sa variabilite est entre genres)

Sorties :
- results/metrics/phase5_repeated_splits_per_repeat.csv
- results/metrics/phase5_repeated_splits_summary.csv
- results/metrics/phase5_logo_per_fold.csv
- results/metrics/phase5_logo_summary.csv
- results/figures/phase5_repeated_auc.png
- results/metrics/PHASE5_REPEATED_STATS_DECISION.txt
"""
from __future__ import annotations

import argparse
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
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import load_control_cksaap, load_nif_cksaap  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"

SEEDS = list(range(42, 62))  # 20 graines
N_BOOT = 1000


def strict_keep() -> set[str]:
    p3 = pd.read_csv(P3)
    return set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))


def filter_to(souches, X, y, groups, keep):
    mask = np.array([s in keep for s in souches])
    return [s for s, m in zip(souches, mask) if m], X[mask], y[mask], groups[mask]


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
        "n_test": int(len(yte)),
        "n_test_aerobie": int((yte == 1).sum()),
        "n_test_anaerobie": int((yte == 0).sum()),
    }


def bootstrap_auc_ci(yte, proba, seed, n_boot=N_BOOT):
    """IC bootstrap 95 % de l'AUC (1 000 reech.) — Mann-Whitney vectorise.

    AUC = P(score_pos > score_neg) + 0,5 * P(egalite). La matrice de
    comparaison pos x neg est calculee une fois ; chaque reechantillonnage
    tire des indices dans les deux classes et moyenne la sous-matrice.
    """
    pos = proba[yte == 1]
    neg = proba[yte == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    comp = (pos[:, None] > neg[None, :]).astype(np.float64) + 0.5 * (
        pos[:, None] == neg[None, :]
    )
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pi = rng.integers(0, len(pos), len(pos))
        ni = rng.integers(0, len(neg), len(neg))
        vals[b] = comp[np.ix_(pi, ni)].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fit_eval(Xtr, Xte, ytr, yte, seed):
    try:
        Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(Xte)[:, 1]
    m = metrics_of(yte, proba)
    lo, hi = bootstrap_auc_ci(yte, proba, seed + 99991)
    m["auc_boot_low"] = lo
    m["auc_boot_high"] = hi
    return m


def eval_repeated(souches, X, y, groups, feature, corpus, seeds):
    rows = []
    for seed in seeds:
        # aleatoire
        strat = y if min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=strat
        )
        m = fit_eval(Xtr, Xte, ytr, yte, seed)
        rows.append(
            {
                "feature": feature,
                "corpus": corpus,
                "protocol": "random_80_20_SMOTE",
                "seed": seed,
                "n": int(len(y)),
                **m,
            }
        )
        # genre
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        m = fit_eval(X[tr], X[te], y[tr], y[te], seed)
        rows.append(
            {
                "feature": feature,
                "corpus": corpus,
                "protocol": "group_shuffle_genus_SMOTE_train",
                "seed": seed,
                "n": int(len(y)),
                "genres_overlap": int(len(set(groups[tr]) & set(groups[te]))),
                **m,
            }
        )
        print(
            f"  [{feature:12s}|{corpus:11s}] seed={seed} done "
            f"(rows={len(rows)})",
            flush=True,
        )
    return rows


def eval_logo(souches, X, y, groups, feature, corpus, seed=42):
    """LOGO deterministe : un fold par genre (test >=3, 2 classes des 2 cotes)."""
    rows = []
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        genus = str(groups[te][0])
        if len(te) < 3:
            continue
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        try:
            Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(X[tr], y[tr])
        except ValueError:
            Xtr_s, ytr_s = X[tr], y[tr]
        model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
        model.fit(Xtr_s, ytr_s)
        proba = model.predict_proba(X[te])[:, 1]
        m = metrics_of(y[te], proba)
        m.pop("auc_boot_low", None)
        rows.append(
            {
                "feature": feature,
                "corpus": corpus,
                "protocol": "LOGO",
                "genus_test": genus,
                **m,
            }
        )
    return rows


def summarise(df):
    keys = ["feature", "corpus", "protocol"]
    out = []
    for (feat, corpus, proto), sub in df.groupby(keys):
        row = {"feature": feat, "corpus": corpus, "protocol": proto, "n_repeats": len(sub)}
        for col in [
            "accuracy",
            "auc",
            "mcc",
            "auprc",
            "recall_aerobie",
            "recall_anaerobie",
        ]:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_sd"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        row["auc_rep_q025"] = float(sub["auc"].quantile(0.025))
        row["auc_rep_q975"] = float(sub["auc"].quantile(0.975))
        if "auc_boot_low" in sub:
            row["auc_boot_low_mean"] = float(sub["auc_boot_low"].mean())
            row["auc_boot_high_mean"] = float(sub["auc_boot_high"].mean())
        out.append(row)
    return pd.DataFrame(out).sort_values(["corpus", "protocol", "feature"])


def summarise_logo(df):
    out = []
    for (feat, corpus), sub in df.groupby(["feature", "corpus"]):
        rng = np.random.default_rng(123)
        aucs = sub["auc"].to_numpy(dtype=float)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(aucs), len(aucs))
            boots.append(float(np.mean(aucs[idx])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(
            {
                "feature": feat,
                "corpus": corpus,
                "protocol": "LOGO_mean_over_genera",
                "n_folds": int(len(sub)),
                "auc_mean": float(np.mean(aucs)),
                "auc_sd_over_genera": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "auc_boot_low": float(lo),
                "auc_boot_high": float(hi),
                "accuracy_mean": float(sub["accuracy"].mean()),
                "mcc_mean": float(sub["mcc"].mean()),
                "auprc_mean": float(sub["auprc"].mean()),
            }
        )
    return pd.DataFrame(out)


def make_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    protos = [
        ("random_80_20_SMOTE", "Aleatoire"),
        ("group_shuffle_genus_SMOTE_train", "Split genre"),
    ]
    corpora = [("full", "Full (~787)"), ("strict_489", "Strict 489")]
    feats = ["Nif_CKSAAP", "RecA_CKSAAP", "GyrB_CKSAAP", "RpoB_CKSAAP"]
    labels = ["Nif", "RecA", "GyrB", "RpoB"]
    colors = ["#1f4e79", "#8b6914", "#6a994e", "#9d4edd"]
    for ci, (corpus, clab) in enumerate(corpora):
        for pi, (proto, plab) in enumerate(protos):
            ax = axes[pi][ci]
            sub = summary[
                (summary["corpus"] == corpus) & (summary["protocol"] == proto)
            ].set_index("feature")
            ys = np.arange(len(feats))[::-1]
            for ypos, feat, lab, col in zip(ys, feats, labels, colors):
                if feat not in sub.index:
                    continue
                r = sub.loc[feat]
                ax.errorbar(
                    r["auc_mean"],
                    ypos,
                    xerr=r["auc_sd"],
                    fmt="o",
                    color=col,
                    ecolor=col,
                    capsize=4,
                    markersize=6,
                )
                ax.text(
                    min(r["auc_mean"] + r["auc_sd"] + 0.015, 1.01),
                    ypos,
                    f"{r['auc_mean']:.3f} +/- {r['auc_sd']:.3f}",
                    va="center",
                    fontsize=8,
                    color=col,
                )
            ax.axvline(0.5, color="grey", ls=":", lw=1)
            ax.set_yticks(ys)
            ax.set_yticklabels(labels)
            ax.set_xlim(0.4, 1.08)
            ax.set_title(f"{clab} — {plab} (AUC moyenne +/- ecart-type, 20 splits)")
            ax.set_xlabel("AUC (HGB)")
            ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(
        "Phase 5 — AUC repetees (20 splits) : Nif vs controles housekeeping",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    p = OUT_FIG / "phase5_repeated_auc.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--no-logo", action="store_true")
    args = ap.parse_args()
    if args.repeats < 1 or args.repeats > len(SEEDS):
        raise SystemExit(f"--repeats doit etre entre 1 et {len(SEEDS)}")

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    keep = strict_keep()

    datasets = [
        ("Nif_CKSAAP",) + load_nif_cksaap(),
        ("RecA_CKSAAP",) + load_control_cksaap("RecA"),
        ("GyrB_CKSAAP",) + load_control_cksaap("GyrB"),
        ("RpoB_CKSAAP",) + load_control_cksaap("RpoB"),
    ]

    # Parallelisation : HGB est lent sur cette machine (~40 s/fit).
    # On decoupe les graines en 4 paquets par (proteine, corpus) -> joblib (4 coeurs).
    from joblib import Parallel, delayed

    seeds_all = SEEDS[: args.repeats]
    chunks = [seeds_all[i::4] for i in range(4)]
    jobs = []
    for feature, souches, X, y, groups in datasets:
        s_s, X_s, y_s, g_s = filter_to(souches, X, y, groups, keep)
        for corpus, (ss, Xs, ys, gs) in {
            "full": (souches, X, y, groups),
            "strict_489": (s_s, X_s, y_s, g_s),
        }.items():
            print(
                f"=== {feature} | {corpus} n={len(ys)} "
                f"aero={int((ys==1).sum())} ana={int((ys==0).sum())} ===",
                flush=True,
            )
            for ch in chunks:
                if ch:
                    jobs.append((feature, corpus, ss, Xs, ys, gs, ch))
    print(f"Jobs paralleles : {len(jobs)} paquets (n_jobs=4)", flush=True)
    results = Parallel(n_jobs=4)(
        delayed(eval_repeated)(ss, Xs, ys, gs, feat, corp, ch)
        for (feat, corp, ss, Xs, ys, gs, ch) in jobs
    )
    all_rows = [r for part in results for r in part]
    print(f"Splits repetes termines : {len(all_rows)} lignes", flush=True)

    logo_rows = []
    if not args.no_logo:
        for feature, souches, X, y, groups in datasets:
            if feature in ("Nif_CKSAAP", "RecA_CKSAAP"):
                print(f"  LOGO Full {feature} ...", flush=True)
                logo_rows += eval_logo(souches, X, y, groups, feature, "full")

    df = pd.DataFrame(all_rows)
    per_path = OUT_METRICS / "phase5_repeated_splits_per_repeat.csv"
    df.to_csv(per_path, index=False)

    summary = summarise(df)
    sum_path = OUT_METRICS / "phase5_repeated_splits_summary.csv"
    summary.to_csv(sum_path, index=False)

    logo_sum_path = None
    logo_path = None
    logo_summary = pd.DataFrame()
    if logo_rows:
        ldf = pd.DataFrame(logo_rows)
        logo_path = OUT_METRICS / "phase5_logo_per_fold.csv"
        ldf.to_csv(logo_path, index=False)
        logo_summary = summarise_logo(ldf)
        logo_sum_path = OUT_METRICS / "phase5_logo_summary.csv"
        logo_summary.to_csv(logo_sum_path, index=False)

    fig_path = make_figure(summary)

    def get(corpus, proto, feat, col="auc_mean"):
        sub = summary[
            (summary["corpus"] == corpus)
            & (summary["protocol"] == proto)
            & (summary["feature"] == feat)
        ]
        return float(sub.iloc[0][col]) if len(sub) else float("nan")

    lines = [
        "PHASE 5 — ETAPE 1 : SPLITS REPETES (20) + IC BOOTSTRAP + MCC/AUPRC",
        "Classifieur HGB, CKSAAP, SMOTE train uniquement, graines 42..61.",
        "",
        "AUC moyenne +/- ecart-type (20 splits) :",
    ]
    for corpus in ("full", "strict_489"):
        for proto, plab in (
            ("random_80_20_SMOTE", "aleatoire"),
            ("group_shuffle_genus_SMOTE_train", "genre"),
        ):
            nif = get(corpus, proto, "Nif_CKSAAP")
            nif_sd = get(corpus, proto, "Nif_CKSAAP", "auc_sd")
            ctrls = {
                c: get(corpus, proto, c)
                for c in ("RecA_CKSAAP", "GyrB_CKSAAP", "RpoB_CKSAAP")
            }
            best_ctrl = max(ctrls.values())
            lines.append(
                f"  {corpus:11s} {plab:10s} Nif={nif:.3f}+/-{nif_sd:.3f} | "
                + " ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in ctrls.items())
                + f" | ecart Nif-meilleur controle={nif-best_ctrl:+.3f}"
            )
    if len(logo_summary):
        lines += ["", "LOGO Full (moyenne sur genres, bootstrap sur les folds) :"]
        for _, r in logo_summary.iterrows():
            lines.append(
                f"  {r['feature']:12s} folds={int(r['n_folds'])} "
                f"AUC={r['auc_mean']:.3f} IC95[{r['auc_boot_low']:.3f};{r['auc_boot_high']:.3f}]"
            )
    lines += [
        "",
        "LECTURE :",
        "- Si les ecarts-types se chevauchent entre Nif et controles, la parite",
        "  controles ~ Nif n'est pas un effet d'un seul decoupage.",
        "- Les chiffres a rapporter dans le manuscrit = moyennes +/- ecart-type",
        "  (tableau de synthese), plus IC bootstrap par split (colonnes auc_boot_*).",
        "",
        f"Duree totale : {time.time()-t0:.0f} s",
        f"Par repetition : {per_path}",
        f"Synthese       : {sum_path}",
    ]
    if logo_path is not None:
        lines.append(f"LOGO folds     : {logo_path}")
        lines.append(f"LOGO synthese  : {logo_sum_path}")
    lines.append(f"Figure         : {fig_path}")

    dec = OUT_METRICS / "PHASE5_REPEATED_STATS_DECISION.txt"
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nOK -> {dec}", flush=True)


if __name__ == "__main__":
    main()
