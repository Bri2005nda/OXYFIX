# -*- coding: utf-8 -*-
"""
Phase 4b — Contôle « est-ce juste parce qu'on a réduit à 489 ? »

Même combinaison type Alida : Nif CKSAAP + XGBoost
(mêmes hyperparamètres que Phase 1b), même seed, même protocole,
sur Full (~787) ET Strict (489).

Si Strict >> Full uniquement à cause de n plus petit / corpus filtré,
on le voit. Sinon, le beau score n'est pas « magie de la réduction ».
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import load_nif_cksaap  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
SEED = 42
ALIDA_ACC = 0.9225
ALIDA_AUC = 0.9772


def make_alida_xgb(seed=SEED):
    """Meilleure famille Alida côté projet : CKSAAP + XGBoost (params Phase 1b)."""
    return XGBClassifier(
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
    )


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


def eval_once(X, y, groups, protocol: str, corpus: str) -> dict:
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

    model = make_alida_xgb()
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(yte, pred))
    auc = float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else float("nan")
    return {
        "feature": "Nif_CKSAAP",
        "model": "XGBoost_AlidaLike",
        "corpus": corpus,
        "n": int(len(y)),
        "n_aerobie": int((y == 1).sum()),
        "n_anaerobie": int((y == 0).sum()),
        "protocol": proto,
        "accuracy": acc,
        "auc": auc,
        "mcc": float(matthews_corrcoef(yte, pred)),
        "delta_acc_vs_alida_ref": acc - ALIDA_ACC,
        "delta_auc_vs_alida_ref": auc - ALIDA_AUC if auc == auc else np.nan,
        **meta,
    }


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    souches, X, y, groups = load_nif_cksaap()
    keep = strict_keep()
    s_s, X_s, y_s, g_s = filter_to(souches, X, y, groups, keep)

    print(f"Full  n={len(y)} aero={(y==1).sum()} ana={(y==0).sum()}")
    print(f"Strict n={len(y_s)} aero={(y_s==1).sum()} ana={(y_s==0).sum()}")

    rows = []
    for corpus, Xs, ys, gs in (
        ("full", X, y, groups),
        ("strict_489", X_s, y_s, g_s),
    ):
        for protocol in ("random", "genus"):
            r = eval_once(Xs, ys, gs, protocol, corpus)
            rows.append(r)
            print(
                f"  [{corpus:11s}|{r['protocol'][:12]:12s}] "
                f"Acc={r['accuracy']:.4f} AUC={r['auc']:.4f}"
            )

    df = pd.DataFrame(rows)
    csv_path = OUT_METRICS / "phase4b_alida_combo_full_vs_strict.csv"
    df.to_csv(csv_path, index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    order_c = ["full", "strict_489"]
    labels_c = ["Full (~787)", "Strict 489"]
    colors = ["#5B7C99", "#8B6239"]

    panels = [
        (axes[0], "random_80_20_SMOTE", "Split aleatoire — Accuracy", "accuracy", ALIDA_ACC),
        (axes[1], "group_shuffle_genus_SMOTE_train", "Split genre — AUC", "auc", ALIDA_AUC),
    ]
    for ax, proto_key, title, metric, ref in panels:
        vals = []
        for c in order_c:
            sub = df[(df["corpus"] == c) & (df["protocol"] == proto_key)]
            vals.append(float(sub.iloc[0][metric]))
        bars = ax.bar(labels_c, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(ref, color="#B33", ls="--", lw=1.2, label=f"Ref. Alida {ref:.4f}")
        ax.set_ylim(0.5, 1.02)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylabel(metric.upper())

    fig.suptitle(
        "Contrôle : même combo Alida (Nif CKSAAP + XGBoost) — Full vs Strict 489",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = OUT_FIG / "phase4b_alida_combo_full_vs_strict.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Décision
    r_full = df[(df["corpus"] == "full") & (df["protocol"] == "random_80_20_SMOTE")].iloc[0]
    r_str = df[(df["corpus"] == "strict_489") & (df["protocol"] == "random_80_20_SMOTE")].iloc[0]
    g_full = df[(df["corpus"] == "full") & (df["protocol"] == "group_shuffle_genus_SMOTE_train")].iloc[0]
    g_str = df[(df["corpus"] == "strict_489") & (df["protocol"] == "group_shuffle_genus_SMOTE_train")].iloc[0]

    d_acc = float(r_str["accuracy"] - r_full["accuracy"])
    d_auc_r = float(r_str["auc"] - r_full["auc"])
    d_auc_g = float(g_str["auc"] - g_full["auc"])

    lines = [
        "PHASE 4b — MEME COMBO ALIDA (Nif CKSAAP + XGBoost) : FULL vs STRICT 489",
        "Question : le beau score sur 489 vient-il seulement du fait qu'on a réduit les données ?",
        "",
        "Protocole identique : seed=42, SMOTE sur train seulement, XGBoost params Phase 1b.",
        f"Reference publiee Alida (autre run historique) : Acc={ALIDA_ACC} | AUC={ALIDA_AUC}",
        "",
        "SPLIT ALEATOIRE",
        f"  Full   n={int(r_full['n'])}  Acc={r_full['accuracy']:.4f}  AUC={r_full['auc']:.4f}",
        f"  Strict n={int(r_str['n'])}  Acc={r_str['accuracy']:.4f}  AUC={r_str['auc']:.4f}",
        f"  Delta Strict-Full : dAcc={d_acc:+.4f}  dAUC={d_auc_r:+.4f}",
        "",
        "SPLIT GENRE",
        f"  Full   Acc={g_full['accuracy']:.4f}  AUC={g_full['auc']:.4f}",
        f"  Strict Acc={g_str['accuracy']:.4f}  AUC={g_str['auc']:.4f}",
        f"  Delta Strict-Full AUC genre : {d_auc_g:+.4f}",
        "",
        "LECTURE",
    ]
    if abs(d_acc) < 0.02 and abs(d_auc_r) < 0.02:
        lines.append(
            "- En aleatoire, Full et Strict sont proches : on ne peut pas dire que le score "
            "vient seulement du fait d'avoir diminue a 489."
        )
    elif d_acc > 0.02 or d_auc_r > 0.02:
        lines.append(
            "- Strict est un peu au-dessus du Full en aleatoire : le filtrage des labels "
            "(corps plus propre) peut aider, ce n'est pas seulement « moins de lignes »."
        )
    else:
        lines.append(
            "- Strict n'est pas superieur au Full en aleatoire : reduire a 489 ne « cree » "
            "pas automatiquement un meilleur modele."
        )
    lines += [
        "- Comparer a la reference Alida historique reste indicatif (autre code/seed possible).",
        "- Le point OXYFIX reste : controles housekeeping + splits taxonomiques.",
        "",
        f"CSV   : {csv_path}",
        f"Figure: {fig_path}",
    ]
    dec = OUT_METRICS / "PHASE4B_ALIDA_COMBO_DECISION.txt"
    dec.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("OK", csv_path)
    print("OK", fig_path)


if __name__ == "__main__":
    main()
