# -*- coding: utf-8 -*-
"""
Métriques PAR CLASSE (aérobie / anaérobie)
— répond à la question encadreur : où chute le score sous split genre ?

Protocoles : aléatoire 80/20 + SMOTE train | GroupShuffleSplit par genre
Modèle     : HistGradientBoosting (même famille que Phase 1)
Cibles     : Nif CKSAAP, RecA/GyrB/RpoB CKSAAP
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import (  # noqa: E402
    load_control_cksaap,
    load_nif_cksaap,
)

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
SEED = 42

# label 1 = aerobie, label 0 = anaerobie
CLASS_NAMES = {0: "anaerobie", 1: "aerobie"}


def _fit_predict(X, y, groups=None, protocol: str = "random"):
    if protocol == "random":
        strat = y if min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=strat
        )
        proto_name = "random_80_20_SMOTE"
        meta = {}
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(X, y, groups))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        proto_name = "group_shuffle_genus_SMOTE_train"
        gtr, gte = set(groups[tr]), set(groups[te])
        meta = {
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "genres_train": int(len(gtr)),
            "genres_test": int(len(gte)),
            "genres_overlap": int(len(gtr & gte)),
        }
    try:
        Xtr_s, ytr_s = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    model = HistGradientBoostingClassifier(random_state=SEED, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proto_name, yte, pred, proba, meta


def per_class_rows(name: str, protocol: str, yte, pred, proba, meta: dict) -> list[dict]:
    rows = []
    # global
    base = {
        "name": name,
        "protocol": protocol,
        "n_test": int(len(yte)),
        "n_test_aerobie": int((yte == 1).sum()),
        "n_test_anaerobie": int((yte == 0).sum()),
        "accuracy": float(accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else np.nan,
        "mcc": float(matthews_corrcoef(yte, pred)),
        **meta,
    }
    for cls, cname in CLASS_NAMES.items():
        rows.append(
            {
                **base,
                "classe": cname,
                "label_id": int(cls),
                "precision": float(precision_score(yte, pred, pos_label=cls, zero_division=0)),
                "recall": float(recall_score(yte, pred, pos_label=cls, zero_division=0)),
                "f1": float(f1_score(yte, pred, pos_label=cls, zero_division=0)),
                "support": int((yte == cls).sum()),
            }
        )
    # also dump sklearn text for archive
    report = classification_report(
        yte, pred, target_names=["anaerobie(0)", "aerobie(1)"], zero_division=0
    )
    return rows, report


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    datasets = [("Nif_CKSAAP",) + load_nif_cksaap()]
    for gene in ("RecA", "GyrB", "RpoB"):
        datasets.append((f"{gene}_CKSAAP",) + load_control_cksaap(gene))

    all_rows = []
    reports = []

    for name, souches, X, y, groups in datasets:
        print(f"=== {name} n={len(y)} ===")
        for protocol_key in ("random", "genus"):
            proto_name, yte, pred, proba, meta = _fit_predict(
                X, y, groups=groups, protocol=protocol_key
            )
            rows, report = per_class_rows(name, proto_name, yte, pred, proba, meta)
            all_rows.extend(rows)
            reports.append(f"\n#### {name} | {proto_name}\n{report}")
            # console brief
            for r in rows:
                print(
                    f"  {protocol_key:6s} {r['classe']:10s} "
                    f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
                    f"supp={r['support']}"
                )

    df = pd.DataFrame(all_rows)
    csv_path = OUT_METRICS / "phase1_per_class_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Pivot-friendly wide table for nif only (easy for prof)
    nif = df[df["name"] == "Nif_CKSAAP"].copy()
    wide_rows = []
    for proto in nif["protocol"].unique():
        sub = nif[nif["protocol"] == proto]
        row = {"name": "Nif_CKSAAP", "protocol": proto}
        for _, r in sub.iterrows():
            c = r["classe"]
            row[f"precision_{c}"] = r["precision"]
            row[f"recall_{c}"] = r["recall"]
            row[f"f1_{c}"] = r["f1"]
            row[f"support_{c}"] = r["support"]
        row["accuracy"] = sub["accuracy"].iloc[0]
        row["auc"] = sub["auc"].iloc[0]
        row["mcc"] = sub["mcc"].iloc[0]
        wide_rows.append(row)
    df_wide = pd.DataFrame(wide_rows)
    wide_path = OUT_METRICS / "phase1_per_class_nif_wide.csv"
    df_wide.to_csv(wide_path, index=False)

    # Figure: recall par classe, nif + RecA, random vs genus
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, feat in zip(axes, ["Nif_CKSAAP", "RecA_CKSAAP"]):
        sub = df[df["name"] == feat]
        # order: random aerobie, random anaerobie, genus aerobie, genus anaerobie
        labels = []
        recalls = []
        colors = []
        for proto_label, proto in [
            ("Aleatoire", "random_80_20_SMOTE"),
            ("Genre", "group_shuffle_genus_SMOTE_train"),
        ]:
            for cls, col in [("aerobie", "#1f4e79"), ("anaerobie", "#8b6914")]:
                r = sub[(sub["protocol"] == proto) & (sub["classe"] == cls)]
                if r.empty:
                    continue
                labels.append(f"{proto_label}\n{cls}")
                recalls.append(float(r["recall"].iloc[0]))
                colors.append(col if proto_label == "Aleatoire" else ("#5b8fc9" if cls == "aerobie" else "#c4a35a"))
        x = np.arange(len(labels))
        ax.bar(x, recalls, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("Recall (rappel)")
        ax.set_title(feat)
        for i, v in enumerate(recalls):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle(
        "Rappel par classe — aléatoire vs split par genre\n"
        "(aérobie = label 1, anaérobie = label 0)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = OUT_FIG / "phase1_per_class_recall.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Decision text for Brinda / prof
    # Compare nif genus drops
    def get_rec(name, proto, classe):
        r = df[(df["name"] == name) & (df["protocol"] == proto) & (df["classe"] == classe)]
        return float(r["recall"].iloc[0]) if len(r) else float("nan")

    pr_r, pr_g = "random_80_20_SMOTE", "group_shuffle_genus_SMOTE_train"
    lines = []
    lines.append("METRIQUES PAR CLASSE — Phase 1 (complement)")
    lines.append("Date: calcul pour repondre a la question encadreur")
    lines.append("")
    lines.append("Classes: aerobie (label=1) | anaerobie (label=0)")
    lines.append("")
    lines.append("Nif CKSAAP — Recall (rappel)")
    lines.append(
        f"  Aleatoire : aerobie={get_rec('Nif_CKSAAP', pr_r, 'aerobie'):.3f} | "
        f"anaerobie={get_rec('Nif_CKSAAP', pr_r, 'anaerobie'):.3f}"
    )
    lines.append(
        f"  Genre     : aerobie={get_rec('Nif_CKSAAP', pr_g, 'aerobie'):.3f} | "
        f"anaerobie={get_rec('Nif_CKSAAP', pr_g, 'anaerobie'):.3f}"
    )
    drop_a = get_rec("Nif_CKSAAP", pr_r, "aerobie") - get_rec("Nif_CKSAAP", pr_g, "aerobie")
    drop_n = get_rec("Nif_CKSAAP", pr_r, "anaerobie") - get_rec("Nif_CKSAAP", pr_g, "anaerobie")
    lines.append(f"  Chute recall aerobie   : {drop_a:+.3f}")
    lines.append(f"  Chute recall anaerobie : {drop_n:+.3f}")
    if drop_a > drop_n + 0.03:
        verdict = "La baisse touche surtout la classe AEROBIE (rappel)."
    elif drop_n > drop_a + 0.03:
        verdict = "La baisse touche surtout la classe ANAEROBIE (rappel)."
    else:
        verdict = "La baisse touche les DEUX classes de facon comparable."
    lines.append("")
    lines.append("LECTURE:")
    lines.append(f"- {verdict}")
    lines.append(
        "- La baisse sous split genre reste d'abord un effet de PROTOCOLE "
        "(perte du raccourci taxonomique), visible sur le score global."
    )
    lines.append(
        "- Le detail par classe precise si une des deux etiquettes souffre plus."
    )
    lines.append("")
    lines.append("Fichiers:")
    lines.append(f"- {csv_path}")
    lines.append(f"- {wide_path}")
    lines.append(f"- {fig_path}")

    dec_path = OUT_METRICS / "PHASE1_PER_CLASS_DECISION.txt"
    dec_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_METRICS / "phase1_per_class_reports.txt").write_text(
        "\n".join(reports), encoding="utf-8"
    )

    print("\n" + "\n".join(lines))
    print(f"\nOK -> {dec_path}")


if __name__ == "__main__":
    main()
