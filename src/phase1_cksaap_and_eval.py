# -*- coding: utf-8 -*-
"""
Phase 1 — CKSAAP (gap 0-3, 1600 dims comme Alida) + comparaison Nif vs contrôles.
Protocole Alida-like: SMOTE global + split 80/20, RF + HistGradientBoosting.
Agrégation au niveau souche (moyenne des séquences Nif).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
CTRL_META = ROOT / "data" / "raw" / "controls_phase1" / "controls_fetch_table.csv"
CTRL_FASTA = ROOT / "data" / "raw" / "controls_phase1" / "fasta"
CK_AERO = ROOT / "data" / "raw" / "Resultats" / "Fixateurs_aerobie" / "CKSAAP type 1.csv"
CK_ANA = ROOT / "data" / "raw" / "Resultats" / "Fixateurs_anaerobie" / "CKSAAP type 1.csv"
PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
GAPS = (0, 1, 2, 3)  # 20*20*4 = 1600


def cksaap_vec(seq: str) -> np.ndarray:
    seq = "".join(ch for ch in seq.upper() if ch in AA_SET)
    L = len(seq)
    vec = np.zeros(20 * 20 * len(GAPS), dtype=np.float64)
    if L < 2:
        return vec
    idx = {a: i for i, a in enumerate(AA)}
    for gi, gap in enumerate(GAPS):
        base = gi * 400
        denom = max(L - gap - 1, 1)
        for i in range(L - gap - 1):
            a, b = seq[i], seq[i + gap + 1]
            if a in idx and b in idx:
                vec[base + idx[a] * 20 + idx[b]] += 1.0
        vec[base : base + 400] /= denom
    return vec


def read_fasta_seq(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "".join(l.strip() for l in lines if l and not l.startswith(">"))


def load_nif_cksaap_strain() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Moyenne CKSAAP par souche + labels depuis provenance."""
    a = pd.read_csv(CK_AERO)
    b = pd.read_csv(CK_ANA)
    df = pd.concat([a, b], ignore_index=True)
    feat_cols = [c for c in df.columns if c.startswith("CKSAAP_")]
    # aligner à 1600 si besoin
    X = df.groupby("bacterie")[feat_cols].mean()
    prov = pd.read_csv(PROV).set_index("souche")
    common = X.index.intersection(prov.index)
    X = X.loc[common]
    y = prov.loc[common, "label"].astype(int).values
    meta = prov.loc[common, ["label_texte"]].copy()
    meta["souche"] = common
    return meta, X.values.astype(np.float64), y


def load_control_matrix(gene: str) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    meta = pd.read_csv(CTRL_META)
    sub = meta[(meta["gene"] == gene) & (meta["status"].astype(str).str.startswith("ok"))]
    if len(sub) < 40:
        print(f"{gene}: trop peu de séquences OK ({len(sub)})")
        return None
    Xs, ys, souches = [], [], []
    for _, row in sub.iterrows():
        path = Path(row["fasta_path"]) if row["fasta_path"] else CTRL_FASTA / gene / f"{row['souche']}.fasta"
        if not path.exists():
            continue
        seq = read_fasta_seq(path)
        Xs.append(cksaap_vec(seq))
        ys.append(int(row["label"]))
        souches.append(row["souche"])
    if len(Xs) < 40:
        return None
    return souches, np.vstack(Xs), np.array(ys)


def eval_protocol(X, y, name: str, seed: int = 42) -> dict:
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"{name}: une seule classe dans y (besoin aéro+anaéro). n={len(y)}")
    strat = y if min(np.bincount(y.astype(int))) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=strat
    )
    if len(np.unique(ytr)) < 2:
        raise ValueError(f"{name}: train mono-classe après split")
    try:
        Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    models = {
        "RF": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1, class_weight=None
        ),
        "HGB": HistGradientBoostingClassifier(random_state=seed, max_depth=6),
    }
    out = {"name": name, "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    curves = {}
    for mname, model in models.items():
        model.fit(Xtr_s, ytr_s)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[f"{mname}_accuracy"] = float(accuracy_score(yte, pred))
        out[f"{mname}_auc"] = float(roc_auc_score(yte, proba))
        out[f"{mname}_mcc"] = float(matthews_corrcoef(yte, pred))
        fpr, tpr, _ = roc_curve(yte, proba)
        curves[mname] = (fpr, tpr, out[f"{mname}_auc"])
    out["_curves"] = curves
    return out


def decide(auc_nif: float, auc_ctrl_best: float) -> str:
    if auc_ctrl_best >= 0.90:
        return (
            "Signal surtout taxonomique (controles AUC ~0.90+). "
            "Orientation: papier / memoire de mise en garde methodologique."
        )
    if auc_ctrl_best >= 0.70:
        return (
            "Signal mixte (controles AUC 0.70-0.80). "
            "Quantifier l'ecart Nif - controles; Phase 2 indispensable."
        )
    return (
        "Signal plus specifique a Nif (controles AUC ~0.55-0.65). "
        "Interpretation mecanistique possible sous reserve Phase 2."
    )


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("=== Nif CKSAAP (souche) ===")
    meta_nif, X_nif, y_nif = load_nif_cksaap_strain()
    print(X_nif.shape, "pos", (y_nif == 1).sum(), "neg", (y_nif == 0).sum())
    r_nif = eval_protocol(X_nif, y_nif, "Nif_CKSAAP")

    results = [ {k: v for k, v in r_nif.items() if k != "_curves"} ]
    all_curves = {"Nif_CKSAAP": r_nif["_curves"]}

    if not CTRL_META.exists():
        print("Contrôles absents — lance d'abord phase1_fetch_controls_uniprot.py")
        pd.DataFrame(results).to_csv(OUT_METRICS / "phase1_nif_only.csv", index=False)
        return

    for gene in ["RecA", "GyrB", "RpoB"]:
        print(f"=== Contrôle {gene} ===")
        loaded = load_control_matrix(gene)
        if loaded is None:
            continue
        souches, Xc, yc = loaded
        print(gene, Xc.shape)
        r = eval_protocol(Xc, yc, f"{gene}_CKSAAP")
        results.append({k: v for k, v in r.items() if k != "_curves"})
        all_curves[f"{gene}_CKSAAP"] = r["_curves"]

    df = pd.DataFrame(results)
    df.to_csv(OUT_METRICS / "phase1_controls_vs_nif.csv", index=False)
    df.to_json(OUT_METRICS / "phase1_controls_vs_nif.json", orient="records", indent=2)

    # figure 2 panneaux: barres AUC + ROC HGB
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = df["name"].tolist()
    aucs = df["HGB_auc"].tolist() if "HGB_auc" in df else df["RF_auc"].tolist()
    axes[0].barh(names, aucs, color=["#1f4e79" if "Nif" in n else "#8B4513" for n in names])
    axes[0].axvline(0.9, color="red", ls="--", lw=1, label="seuil taxonomique 0.90")
    axes[0].axvline(0.7, color="orange", ls="--", lw=1, label="seuil mixte 0.70")
    axes[0].set_xlabel("AUC (HGB)")
    axes[0].set_title("Phase 1 — Nif vs contrôles")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0.4, 1.0)

    for name, curves in all_curves.items():
        if "HGB" in curves:
            fpr, tpr, auc = curves["HGB"]
            axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("Courbes ROC (HGB)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig_path = OUT_FIG / "phase1_nif_vs_controls.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    ctrl_aucs = [r["HGB_auc"] for r in results if r["name"] != "Nif_CKSAAP" and "HGB_auc" in r]
    decision = decide(r_nif.get("HGB_auc", 0), max(ctrl_aucs) if ctrl_aucs else 0)
    note = OUT_METRICS / "PHASE1_DECISION.txt"
    note.write_text(
        "PHASE 1 — DÉCISION D'ORIENTATION\n\n"
        + df.to_string(index=False)
        + "\n\nDÉCISION:\n"
        + decision
        + "\n\nFigure: "
        + str(fig_path)
        + "\n",
        encoding="utf-8",
    )
    print(decision)
    print("OK ->", OUT_METRICS / "phase1_controls_vs_nif.csv")
    print("OK ->", fig_path)
    print("OK ->", note)


if __name__ == "__main__":
    main()
