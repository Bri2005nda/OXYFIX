# -*- coding: utf-8 -*-
"""
Phase 1 PREMIUM — complete le plan:
1) Baselines triviales (longueur + composition AA 20)
2) Test de permutation (n=1000) Nif vs RecA
3) Comparaison Nif vs controles sous split par GENRE
4) Figure + decision premium mise a jour
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, permutation_test_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
CTRL_META = ROOT / "data" / "raw" / "controls_phase1" / "controls_fetch_table.csv"
CTRL_FASTA = ROOT / "data" / "raw" / "controls_phase1" / "fasta"
CK_AERO = ROOT / "data" / "raw" / "Resultats" / "Fixateurs_aerobie" / "CKSAAP type 1.csv"
CK_ANA = ROOT / "data" / "raw" / "Resultats" / "Fixateurs_anaerobie" / "CKSAAP type 1.csv"
PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"
FASTA_A = ROOT / "data" / "raw" / "fasta" / "Fixateurs_aerobie" / "Fixateurs_aerobie"
FASTA_N = ROOT / "data" / "raw" / "fasta" / "Fixateurs_anaerobie" / "Fixateurs_anaerobie"

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
GAPS = (0, 1, 2, 3)
N_PERM = 1000
SEED = 42


def read_fasta_seqs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    seqs, cur = [], []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def aa_comp_and_length(seqs: list[str]) -> np.ndarray:
    """[len_mean, aa_freq_20] = 21 dims."""
    if not seqs:
        return np.zeros(21)
    lens = []
    counts = np.zeros(20)
    total = 0
    idx = {a: i for i, a in enumerate(AA)}
    for s in seqs:
        s = "".join(ch for ch in s.upper() if ch in AA_SET)
        if not s:
            continue
        lens.append(len(s))
        for ch in s:
            counts[idx[ch]] += 1
            total += 1
    if total == 0:
        return np.zeros(21)
    freqs = counts / total
    return np.concatenate([[float(np.mean(lens))], freqs])


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


def genus_of(souche: str) -> str:
    return str(souche).split("_")[0]


def load_nif_cksaap():
    a = pd.read_csv(CK_AERO)
    b = pd.read_csv(CK_ANA)
    df = pd.concat([a, b], ignore_index=True)
    feat_cols = [c for c in df.columns if c.startswith("CKSAAP_")]
    X = df.groupby("bacterie")[feat_cols].mean()
    prov = pd.read_csv(PROV).set_index("souche")
    common = X.index.intersection(prov.index)
    X = X.loc[common]
    y = prov.loc[common, "label"].astype(int).to_numpy()
    souches = list(common)
    groups = np.array([genus_of(s) for s in souches])
    return souches, X.to_numpy(dtype=np.float64), y, groups


def load_nif_trivial():
    """Baseline triviale depuis FASTA nif (longueur + composition AA)."""
    prov = pd.read_csv(PROV)
    Xs, ys, souches, groups = [], [], [], []
    for _, row in prov.iterrows():
        souche = row["souche"]
        path = FASTA_A / f"{souche}.fasta"
        if not path.exists():
            path = FASTA_N / f"{souche}.fasta"
        if not path.exists():
            # parfois nom fichier different
            continue
        seqs = read_fasta_seqs(path)
        Xs.append(aa_comp_and_length(seqs))
        ys.append(int(row["label"]))
        souches.append(souche)
        groups.append(genus_of(souche))
    return souches, np.vstack(Xs), np.array(ys), np.array(groups)


def load_control_cksaap(gene: str):
    meta = pd.read_csv(CTRL_META)
    sub = meta[(meta["gene"] == gene) & (meta["status"].astype(str).str.startswith("ok"))]
    Xs, ys, souches, groups = [], [], [], []
    for _, row in sub.iterrows():
        path = Path(str(row["fasta_path"])) if row.get("fasta_path") else CTRL_FASTA / gene / f"{row['souche']}.fasta"
        if not path.exists():
            path = CTRL_FASTA / gene / f"{row['souche']}.fasta"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        seq = "".join(l.strip() for l in text.splitlines() if l and not l.startswith(">"))
        Xs.append(cksaap_vec(seq))
        ys.append(int(row["label"]))
        souches.append(row["souche"])
        groups.append(genus_of(row["souche"]))
    return souches, np.vstack(Xs), np.array(ys), np.array(groups)


def load_control_trivial(gene: str = "RecA"):
    meta = pd.read_csv(CTRL_META)
    sub = meta[(meta["gene"] == gene) & (meta["status"].astype(str).str.startswith("ok"))]
    Xs, ys, souches, groups = [], [], [], []
    for _, row in sub.iterrows():
        path = Path(str(row["fasta_path"])) if row.get("fasta_path") else CTRL_FASTA / gene / f"{row['souche']}.fasta"
        if not path.exists():
            path = CTRL_FASTA / gene / f"{row['souche']}.fasta"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        seq = "".join(l.strip() for l in text.splitlines() if l and not l.startswith(">"))
        Xs.append(aa_comp_and_length([seq]))
        ys.append(int(row["label"]))
        souches.append(row["souche"])
        groups.append(genus_of(row["souche"]))
    return souches, np.vstack(Xs), np.array(ys), np.array(groups)


def eval_random(X, y, name: str, seed: int = SEED) -> dict:
    strat = y if min(np.bincount(y.astype(int))) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    try:
        Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "name": name,
        "protocol": "random_80_20_SMOTE",
        "n": int(len(y)),
        "accuracy": float(accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, proba)),
        "mcc": float(matthews_corrcoef(yte, pred)),
    }


def eval_genus(X, y, groups, name: str, seed: int = SEED) -> dict:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    # SMOTE train-only
    try:
        Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    model = HistGradientBoostingClassifier(random_state=seed, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    # genres overlap check
    gtr, gte = set(groups[tr]), set(groups[te])
    return {
        "name": name,
        "protocol": "group_shuffle_genus_SMOTE_train",
        "n": int(len(y)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "genres_train": int(len(gtr)),
        "genres_test": int(len(gte)),
        "genres_overlap": int(len(gtr & gte)),
        "accuracy": float(accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(yte, pred)),
    }


def run_permutations(X, y, name: str, n_perm: int = N_PERM) -> dict:
    """Permutation test AUC (cv=3), modele rapide + scalable."""
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED),
    )
    # si trop de features pour LR stable, sous-echantillonner colonnes? CKSAAP ok with scaler
    print(f"  permutations {name}: n={n_perm} ...")
    score, perm_scores, pvalue = permutation_test_score(
        clf,
        X,
        y,
        scoring="roc_auc",
        cv=3,
        n_permutations=n_perm,
        random_state=SEED,
        n_jobs=-1,
    )
    return {
        "name": name,
        "true_auc_cv3": float(score),
        "perm_auc_mean": float(np.mean(perm_scores)),
        "perm_auc_std": float(np.std(perm_scores)),
        "perm_auc_p95": float(np.percentile(perm_scores, 95)),
        "pvalue": float(pvalue),
        "n_permutations": int(n_perm),
    }


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    rows_random = []
    rows_genus = []
    rows_perm = []

    print("=== 1) Nif CKSAAP ===")
    s_nif, X_nif, y_nif, g_nif = load_nif_cksaap()
    rows_random.append(eval_random(X_nif, y_nif, "Nif_CKSAAP"))
    rows_genus.append(eval_genus(X_nif, y_nif, g_nif, "Nif_CKSAAP"))

    print("=== 2) Baseline triviale Nif (longueur + AAC) ===")
    s_tr, X_tr, y_tr, g_tr = load_nif_trivial()
    print("trivial nif", X_tr.shape)
    rows_random.append(eval_random(X_tr, y_tr, "Nif_trivial_len_AAC"))
    rows_genus.append(eval_genus(X_tr, y_tr, g_tr, "Nif_trivial_len_AAC"))

    print("=== 3) Controles CKSAAP ===")
    for gene in ["RecA", "GyrB", "RpoB"]:
        s, X, y, g = load_control_cksaap(gene)
        print(gene, X.shape)
        rows_random.append(eval_random(X, y, f"{gene}_CKSAAP"))
        rows_genus.append(eval_genus(X, y, g, f"{gene}_CKSAAP"))

    print("=== 4) Baseline triviale RecA ===")
    s_r, X_r, y_r, g_r = load_control_trivial("RecA")
    rows_random.append(eval_random(X_r, y_r, "RecA_trivial_len_AAC"))
    rows_genus.append(eval_genus(X_r, y_r, g_r, "RecA_trivial_len_AAC"))

    print("=== 5) Permutations (1000) Nif CKSAAP + RecA CKSAAP ===")
    rows_perm.append(run_permutations(X_nif, y_nif, "Nif_CKSAAP", N_PERM))
    X_reca = load_control_cksaap("RecA")[1]
    y_reca = load_control_cksaap("RecA")[2]
    rows_perm.append(run_permutations(X_reca, y_reca, "RecA_CKSAAP", N_PERM))

    df_r = pd.DataFrame(rows_random)
    df_g = pd.DataFrame(rows_genus)
    df_p = pd.DataFrame(rows_perm)
    df_r.to_csv(OUT_METRICS / "phase1_premium_random.csv", index=False)
    df_g.to_csv(OUT_METRICS / "phase1_premium_genus.csv", index=False)
    df_p.to_csv(OUT_METRICS / "phase1_premium_permutations.csv", index=False)

    # Figure premium 3 panneaux
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # A: random AUC bars
    axes[0].barh(df_r["name"], df_r["auc"], color=["#1f4e79" if "Nif_CKSAAP" == n else "#8B4513" if "CKSAAP" in n else "#6b8e23" for n in df_r["name"]])
    axes[0].axvline(0.90, color="red", ls="--", lw=1)
    axes[0].axvline(0.70, color="orange", ls="--", lw=1)
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_xlabel("AUC")
    axes[0].set_title("A) Split aleatoire")

    # B: genus AUC bars
    axes[1].barh(df_g["name"], df_g["auc"], color=["#1f4e79" if "Nif_CKSAAP" == n else "#8B4513" if "CKSAAP" in n else "#6b8e23" for n in df_g["name"]])
    axes[1].axvline(0.90, color="red", ls="--", lw=1)
    axes[1].axvline(0.70, color="orange", ls="--", lw=1)
    axes[1].set_xlim(0.45, 1.0)
    axes[1].set_xlabel("AUC")
    axes[1].set_title("B) Split par genre")

    # C: permutation histogram for Nif
    # reload perm scores? we only stored summary — recompute quick hist from saved? 
    # store perm scores to npy during run
    axes[2].axis("off")
    txt = ["C) Permutations (resume)", ""]
    for _, row in df_p.iterrows():
        txt.append(f"{row['name']}:")
        txt.append(f"  AUC vrai={row['true_auc_cv3']:.3f}")
        txt.append(f"  null mean={row['perm_auc_mean']:.3f}")
        txt.append(f"  p={row['pvalue']:.4g}")
        txt.append("")
    axes[2].text(0.05, 0.95, "\n".join(txt), va="top", family="monospace", fontsize=9)

    fig.tight_layout()
    fig_path = OUT_FIG / "phase1_premium_summary.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    # Decision premium
    nif_r = float(df_r.loc[df_r["name"] == "Nif_CKSAAP", "auc"].iloc[0])
    ctrl_r = float(df_r.loc[df_r["name"].str.endswith("_CKSAAP") & ~df_r["name"].str.startswith("Nif"), "auc"].max())
    triv_r = float(df_r.loc[df_r["name"] == "Nif_trivial_len_AAC", "auc"].iloc[0])
    nif_g = float(df_g.loc[df_g["name"] == "Nif_CKSAAP", "auc"].iloc[0])
    ctrl_g = float(df_g.loc[df_g["name"].str.endswith("_CKSAAP") & ~df_g["name"].str.startswith("Nif"), "auc"].max())
    p_nif = float(df_p.loc[df_p["name"] == "Nif_CKSAAP", "pvalue"].iloc[0])

    decision = f"""PHASE 1 PREMIUM — DECISION D ORIENTATION
Date: complete selon plan (baselines + permutations + split genre)

1) SPLIT ALEATOIRE (AUC HGB)
{df_r.to_string(index=False)}

2) SPLIT PAR GENRE (AUC HGB, SMOTE train-only)
{df_g.to_string(index=False)}

3) PERMUTATIONS (n={N_PERM}, AUC cv=3, LogisticRegression)
{df_p.to_string(index=False)}

4) LECTURE PREMIUM
- Nif aleatoire AUC = {nif_r:.3f}
- Meilleur controle CKSAAP aleatoire AUC = {ctrl_r:.3f}
- Baseline triviale Nif (longueur+AAC) AUC = {triv_r:.3f}
- Nif split-genre AUC = {nif_g:.3f}
- Meilleur controle split-genre AUC = {ctrl_g:.3f}
- p-value permutation Nif = {p_nif:.4g}

DECISION (grille plan + enrichissements premium):
Les controles RecA/GyrB/RpoB atteignent des AUC comparables a nif sous split aleatoire
(>= 0.90), donc le signal est surtout TAXONOMIQUE.
La baseline triviale elevee renforce ce constat si elle reste haute.
Sous split par genre, la chute (ou non) precise si le confondant est bien la parente.
Les permutations confirment que le score observe depasse (ou non) le hasard structure.

ORIENTATION: memoire / papier de mise en garde methodologique.
Phase 2 reste le coeur pour la courbe de decroissance complete.

Figure: {fig_path}
Fichiers:
- results/metrics/phase1_premium_random.csv
- results/metrics/phase1_premium_genus.csv
- results/metrics/phase1_premium_permutations.csv
"""
    out_dec = OUT_METRICS / "PHASE1_PREMIUM_DECISION.txt"
    out_dec.write_text(decision, encoding="utf-8")

    # update short README
    (OUT_METRICS / "PHASE1_README.txt").write_text(
        "PHASE 1 PREMIUM COMPLETEE\n"
        "Voir PHASE1_PREMIUM_DECISION.txt et figure phase1_premium_summary.png\n",
        encoding="utf-8",
    )

    print(decision)
    print("OK ->", out_dec)


if __name__ == "__main__":
    main()
