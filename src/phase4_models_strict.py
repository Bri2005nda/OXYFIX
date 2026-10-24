# -*- coding: utf-8 -*-
"""
Phase 4 — Modeles alternatifs sur corpus STRICT (489)

Objectif (piste parallele, sans abandonner OXYFIX):
tester si un modele plus fort / un ensemble depasse Alida
sur labels documentes, sous aleatoire ET split genre.

Modeles: HGB (baseline), XGBoost, ExtraTrees, MLP, Voting soft
Cibles : Nif CKSAAP + RecA CKSAAP (controle)
Note   : embeddings ESM non dispo (torch absent) — piste DL reportée.
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
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import load_control_cksaap, load_nif_cksaap  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
SEED = 42
ALIDA_ACC = 0.9225
ALIDA_AUC = 0.9772


def strict_keep() -> set[str]:
    p3 = pd.read_csv(P3)
    return set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))


def filter_to(souches, X, y, groups, keep):
    mask = np.array([s in keep for s in souches])
    return (
        [s for s, m in zip(souches, mask) if m],
        X[mask],
        y[mask],
        groups[mask],
    )


def make_models(seed=SEED):
    hgb = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    et = ExtraTreesClassifier(
        n_estimators=600, random_state=seed, n_jobs=-1, max_depth=None, min_samples_leaf=1
    )
    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-3,
                    max_iter=400,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    vote = VotingClassifier(
        estimators=[("hgb", hgb), ("xgb", xgb), ("et", et)],
        voting="soft",
        weights=[1, 1, 1],
    )
    return {
        "HGB": hgb,
        "XGBoost": xgb,
        "ExtraTrees": et,
        "MLP": mlp,
        "VotingSoft": vote,
    }


def eval_protocol(X, y, groups, protocol: str, feature_name: str, corpus: str):
    rows = []
    if protocol == "random":
        strat = y if min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=strat
        )
        proto = "random_80_20_SMOTE"
        meta = {}
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
        # rebuild fresh model each time
        model = make_models()[mname]
        model.fit(Xtr_s, ytr_s)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xte)[:, 1]
        else:
            # unlikely
            proba = model.decision_function(Xte)
            proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
        pred = (proba >= 0.5).astype(int)
        acc = float(accuracy_score(yte, pred))
        auc = float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else np.nan
        rows.append(
            {
                "feature": feature_name,
                "corpus": corpus,
                "n": int(len(y)),
                "n_aerobie": int((y == 1).sum()),
                "n_anaerobie": int((y == 0).sum()),
                "protocol": proto,
                "model": mname,
                "accuracy": acc,
                "auc": auc,
                "mcc": float(matthews_corrcoef(yte, pred)),
                "f1_aerobie": float(f1_score(yte, pred, pos_label=1, zero_division=0)),
                "f1_anaerobie": float(f1_score(yte, pred, pos_label=0, zero_division=0)),
                "delta_acc_vs_alida": acc - ALIDA_ACC,
                "delta_auc_vs_alida": auc - ALIDA_AUC if not np.isnan(auc) else np.nan,
                "beats_alida_acc": bool(acc > ALIDA_ACC),
                "beats_alida_auc": bool(auc > ALIDA_AUC) if not np.isnan(auc) else False,
                **meta,
            }
        )
        print(
            f"  [{feature_name:12s}|{corpus}|{proto[:10]:10s}] {mname:10s} "
            f"Acc={acc:.4f} AUC={auc:.4f} dAcc={acc-ALIDA_ACC:+.4f}"
        )
    return rows


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    keep = strict_keep()

    all_rows = []
    # Nif strict
    s, X, y, g = load_nif_cksaap()
    s, X, y, g = filter_to(s, X, y, g, keep)
    print(f"Nif Strict n={len(y)} aero={int((y==1).sum())} ana={int((y==0).sum())}")
    for proto in ("random", "genus"):
        all_rows += eval_protocol(X, y, g, proto, "Nif_CKSAAP", "strict_489")

    # RecA strict control
    s, X, y, g = load_control_cksaap("RecA")
    s, X, y, g = filter_to(s, X, y, g, keep)
    print(f"RecA Strict n={len(y)}")
    for proto in ("random", "genus"):
        all_rows += eval_protocol(X, y, g, proto, "RecA_CKSAAP", "strict_489")

    df = pd.DataFrame(all_rows)
    csv_path = OUT_METRICS / "phase4_models_strict.csv"
    df.to_csv(csv_path, index=False)

    # Best nif random
    nif_r = df[
        (df["feature"] == "Nif_CKSAAP")
        & (df["protocol"] == "random_80_20_SMOTE")
    ].sort_values(["accuracy", "auc"], ascending=False)
    best = nif_r.iloc[0]
    nif_g = df[
        (df["feature"] == "Nif_CKSAAP")
        & (df["protocol"] == "group_shuffle_genus_SMOTE_train")
    ].sort_values(["auc", "accuracy"], ascending=False)
    best_g = nif_g.iloc[0]
    rec_r = df[
        (df["feature"] == "RecA_CKSAAP")
        & (df["protocol"] == "random_80_20_SMOTE")
        & (df["model"] == best["model"])
    ]
    rec_auc = float(rec_r["auc"].iloc[0]) if len(rec_r) else np.nan

    # Figure — Acc vs Acc (aléatoire) et AUC vs AUC (genre)
    # Alida = protocole aléatoire seulement : sa ligne AUC à droite est une RÉFÉRENCE,
    # pas un score Alida en split genre (Alida n'a pas fait le split genre).
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes[0]
    sub = nif_r.sort_values("model")
    colors = ["#1a7f37" if m == best["model"] else "#1f4e79" for m in sub["model"]]
    ax.bar(sub["model"], sub["accuracy"], color=colors, edgecolor="black", lw=0.4)
    ax.axhline(
        ALIDA_ACC,
        color="red",
        ls="--",
        lw=1.2,
        label=f"Alida Acc {ALIDA_ACC:.4f} (aléatoire)",
    )
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Strict 489 — Nif — Accuracy (split aléatoire)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, loc="lower right")
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r["accuracy"] + 0.002, f"{r['accuracy']:.3f}", ha="center", fontsize=7)

    ax = axes[1]
    subg = nif_g.sort_values("model")
    ax.bar(subg["model"], subg["auc"], color="#8b6914", edgecolor="black", lw=0.4)
    ax.axhline(
        ALIDA_AUC,
        color="red",
        ls="--",
        lw=1.2,
        label=f"Alida AUC {ALIDA_AUC:.4f} (réf. aléatoire, pas genre)",
    )
    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Strict 489 — Nif — AUC (split genre ; Alida n'a pas ce protocole)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, loc="lower right")
    for i, (_, r) in enumerate(subg.iterrows()):
        ax.text(i, r["auc"] + 0.002, f"{r['auc']:.3f}", ha="center", fontsize=7)

    fig.suptitle(
        "Phase 4 — Modèles sur Strict 489 (Nif CKSAAP)\n"
        "Ne pas comparer Acc Alida (0,92) à l'AUC genre (~0,97) : métriques/protocoles différents",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = OUT_FIG / "phase4_models_strict.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "PHASE 4 — MODELES ALTERNATIFS SUR CORPUS STRICT (489)",
        "Piste parallele: tenter un gain de performance SANS abandonner OXYFIX.",
        "Embeddings ESM non testes ici (torch/transformers absents).",
        "",
        f"Reference Alida: Acc={ALIDA_ACC} | AUC={ALIDA_AUC}",
        "",
        "MEILLEUR Nif (aleatoire, Strict):",
        f"  Modele = {best['model']}",
        f"  Acc={best['accuracy']:.4f} | AUC={best['auc']:.4f} | "
        f"dAcc={best['delta_acc_vs_alida']:+.4f} | dAUC={best['delta_auc_vs_alida']:+.4f}",
        f"  Depasse Alida Acc? {best['beats_alida_acc']} | AUC? {best['beats_alida_auc']}",
        f"  RecA meme modele (aleatoire) AUC={rec_auc:.4f}",
        "",
        "MEILLEUR Nif (split genre, Strict) selon AUC:",
        f"  Modele = {best_g['model']} | Acc={best_g['accuracy']:.4f} | AUC={best_g['auc']:.4f}",
        "",
        "LECTURE:",
        "- Si Acc/AUC aleatoire > Alida MAIS RecA reste comparable: gain non mecanistique.",
        "- Si le modele reste haut en genre: a discuter (desequilibre Strict 150/339).",
        "- Phase 4 documente l'essai de modeles plus forts; OXYFIX (diagnostic) reste le coeur.",
        "",
        "SUITE POSSIBLE:",
        "- embeddings proteines (ESM) si environnement DL installe",
        "- sensibilite corpus Etendu 559",
        "",
        f"CSV   : {csv_path}",
        f"Figure: {fig_path}",
    ]
    dec = OUT_METRICS / "PHASE4_DECISION.txt"
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nOK -> {dec}")


if __name__ == "__main__":
    main()
