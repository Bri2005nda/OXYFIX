# -*- coding: utf-8 -*-
"""
Piste 2 + 3 (en parallele du diagnostic OXYFIX)
2) Modeles paralleles vs reference Alida (Acc 0.9225 / AUC 0.9772)
3) Meme evaluation sur corpus BacDive documente (n=489)

Ne remplace PAS Phase 1 : complete avec un resultat de performance nouveau.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import genus_of, load_nif_cksaap  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"
SEED = 42
ALIDA_ACC = 0.9225
ALIDA_AUC = 0.9772


def make_models(seed: int = SEED):
    return {
        "HGB": HistGradientBoostingClassifier(random_state=seed, max_depth=6),
        "RF": RandomForestClassifier(
            n_estimators=400, random_state=seed, n_jobs=-1, max_depth=None
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1, max_depth=None
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def eval_split(X, y, groups, protocol: str, corpus: str):
    rows = []
    if protocol == "random":
        strat = y if min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=strat
        )
        proto = "random_80_20_SMOTE"
        meta = {"genres_overlap": np.nan}
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(X, y, groups))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        proto = "group_shuffle_genus_SMOTE_train"
        meta = {"genres_overlap": int(len(set(groups[tr]) & set(groups[te])))}

    try:
        Xtr_s, ytr_s = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr

    for mname, model in make_models().items():
        model.fit(Xtr_s, ytr_s)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = float(accuracy_score(yte, pred))
        auc = float(roc_auc_score(yte, proba))
        mcc = float(matthews_corrcoef(yte, pred))
        f1_a = float(f1_score(yte, pred, pos_label=1, zero_division=0))
        f1_n = float(f1_score(yte, pred, pos_label=0, zero_division=0))
        rows.append(
            {
                "corpus": corpus,
                "n": int(len(y)),
                "protocol": proto,
                "model": mname,
                "accuracy": acc,
                "auc": auc,
                "mcc": mcc,
                "f1_aerobie": f1_a,
                "f1_anaerobie": f1_n,
                "delta_acc_vs_alida": acc - ALIDA_ACC,
                "delta_auc_vs_alida": auc - ALIDA_AUC,
                "beats_alida_acc": bool(acc > ALIDA_ACC),
                "beats_alida_auc": bool(auc > ALIDA_AUC),
                **meta,
            }
        )
        print(
            f"  [{corpus:12s}] {proto[:12]:12s} {mname:10s} "
            f"Acc={acc:.4f} AUC={auc:.4f} dAcc={acc - ALIDA_ACC:+.4f}"
        )
    return rows


def subset_documente(souches, X, y, groups):
    prov = pd.read_csv(PROV)
    ok = set(prov.loc[prov["statut_provenance"] == "documente", "souche"].astype(str))
    mask = np.array([s in ok for s in souches])
    return (
        [s for s, m in zip(souches, mask) if m],
        X[mask],
        y[mask],
        groups[mask],
    )


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    souches, X, y, groups = load_nif_cksaap()
    print(f"Full corpus n={len(y)} pos={int((y==1).sum())} neg={int((y==0).sum())}")

    all_rows = []
    for protocol in ("random", "genus"):
        all_rows.extend(eval_split(X, y, groups, protocol, corpus="full_787"))

    s489, X489, y489, g489 = subset_documente(souches, X, y, groups)
    print(
        f"Documente corpus n={len(y489)} pos={int((y489==1).sum())} "
        f"neg={int((y489==0).sum())}"
    )
    if len(np.unique(y489)) < 2 or len(y489) < 50:
        print("WARN: corpus documente trop petit / mono-classe — skip")
    else:
        for protocol in ("random", "genus"):
            all_rows.extend(eval_split(X489, y489, g489, protocol, corpus="documente_489"))

    df = pd.DataFrame(all_rows)
    csv_path = OUT_METRICS / "phase1b_parallel_models.csv"
    df.to_csv(csv_path, index=False)

    # Best random full vs Alida
    rnd = df[(df["corpus"] == "full_787") & (df["protocol"] == "random_80_20_SMOTE")]
    best = rnd.sort_values(["accuracy", "auc"], ascending=False).iloc[0]

    # Figure bars Acc random full
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    sub = rnd.sort_values("model")
    colors = ["#1f4e79" if m != best["model"] else "#1a7f37" for m in sub["model"]]
    ax.bar(sub["model"], sub["accuracy"], color=colors, edgecolor="black", lw=0.4)
    ax.axhline(ALIDA_ACC, color="red", ls="--", lw=1.2, label=f"Alida Acc {ALIDA_ACC}")
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Corpus 787 — split aleatoire")
    ax.legend(fontsize=8)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r["accuracy"] + 0.002, f"{r['accuracy']:.3f}", ha="center", fontsize=8)

    ax = axes[1]
    subg = df[
        (df["corpus"] == "full_787")
        & (df["protocol"] == "group_shuffle_genus_SMOTE_train")
    ].sort_values("model")
    ax.bar(subg["model"], subg["accuracy"], color="#8b6914", edgecolor="black", lw=0.4)
    ax.axhline(ALIDA_ACC, color="red", ls="--", lw=1.2, label=f"Alida Acc {ALIDA_ACC}")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Corpus 787 — split genre")
    ax.legend(fontsize=8)
    for i, (_, r) in enumerate(subg.iterrows()):
        ax.text(i, r["accuracy"] + 0.01, f"{r['accuracy']:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "Piste parallele — modeles vs reference Alida (Nif CKSAAP)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = OUT_FIG / "phase1b_parallel_models_vs_alida.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Decision
    lines = []
    lines.append("PISTES 2+3 — MODELES PARALLELES + CORPUS DOCUMENTE")
    lines.append("Reference Alida: Acc=0.9225 | AUC=0.9772")
    lines.append("")
    lines.append("MEILLEUR MODELE (corpus 787, aleatoire):")
    lines.append(
        f"  {best['model']} | Acc={best['accuracy']:.4f} | AUC={best['auc']:.4f} | "
        f"dAcc={best['delta_acc_vs_alida']:+.4f} | dAUC={best['delta_auc_vs_alida']:+.4f}"
    )
    if best["beats_alida_acc"] or best["beats_alida_auc"]:
        lines.append("  -> Depasse Alida sur au moins une metrique (Acc et/ou AUC).")
    else:
        lines.append(
            "  -> Ne depasse pas Alida (normal si le plafond est surtout taxonomique)."
        )
        lines.append(
            "  -> Le resultat reste utile: on a TESTE des modeles plus forts en parallele."
        )
    lines.append("")
    # documente summary if present
    doc = df[df["corpus"] == "documente_489"]
    if len(doc):
        best_doc = doc[doc["protocol"] == "random_80_20_SMOTE"].sort_values(
            ["accuracy", "auc"], ascending=False
        ).iloc[0]
        lines.append("CORPUS DOCUMENTE (n=489), aleatoire — meilleur:")
        lines.append(
            f"  {best_doc['model']} | Acc={best_doc['accuracy']:.4f} | "
            f"AUC={best_doc['auc']:.4f} | dAcc={best_doc['delta_acc_vs_alida']:+.4f}"
        )
        lines.append(
            "  Lecture: si le score change beaucoup vs 787, les labels flous comptent."
        )
    lines.append("")
    lines.append("ORIENTATION:")
    lines.append("- OXYFIX (diagnostic parenté) reste le coeur.")
    lines.append("- La piste parallele documente l'essai de gain de performance.")
    lines.append("- Suite: Phase 2 courbe de decroissance + eventuelle nouvelle base.")
    lines.append("")
    lines.append(f"Fichiers: {csv_path}")
    lines.append(f"Figure : {fig_path}")

    dec = OUT_METRICS / "PHASE1B_PARALLEL_DECISION.txt"
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nOK -> {dec}")


if __name__ == "__main__":
    main()
