# -*- coding: utf-8 -*-
"""
Phase 6c — Modèles par famille Nif : NifH, NifD, NifK séparément.

Les FASTA souches contiennent tout le cluster nif (et des homologues).
Classification d'une séquence dans une famille structurelle, par le
produit annoté UniProt (en-tête FASTA), avec contrôle du gène :
- NifH : "Nitrogenase iron protein" / NifH, hors protéines de maturation ;
- NifD : chaîne alpha de la nitrogénase (Mo-Fe), hors biosynthèse du
  cofacteur (NifE/NifN) ;
- NifK : chaîne bêta de la nitrogénase (Mo-Fe), hors biosynthèse.
Les nitrogénases alternatives anf/vnf et les homologues bch/chl sont
exclus. Si une souche possède plusieurs copies, leurs vecteurs CKSAAP
sont moyennés (une ligne = une souche), comme le modèle combiné.

Même protocole que Phase 5 : HGB, CKSAAP écarts 0-3, SMOTE train
uniquement, 20 graines, corpus Full/Strict, splits aléatoire et genre.

Sorties :
- results/metrics/phase6c_nif_family_counts.csv (famille x classe)
- results/metrics/phase6c_nif_families_per_repeat.csv
- results/metrics/phase6c_nif_families_summary.csv
- results/figures/phase6c_nif_families.png
- results/metrics/PHASE6C_NIF_FAMILIES_DECISION.txt
"""
from __future__ import annotations

import re
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import cksaap_vec  # noqa: E402

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
FASTA_DIRS = [
    ROOT / "data" / "raw" / "fasta" / "Fixateurs_aerobie" / "Fixateurs_aerobie",
    ROOT / "data" / "raw" / "fasta" / "Fixateurs_anaerobie" / "Fixateurs_anaerobie",
]
PHASE5_SUMMARY = OUT_METRICS / "phase5_repeated_splits_summary.csv"
SEEDS = list(range(42, 62))
FAMILIES = ["NifH", "NifD", "NifK"]


def read_fasta_records(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    records = []
    for part in text.split(">")[1:]:
        if "\n" not in part:
            continue
        head, seq = part.split("\n", 1)
        records.append((head.strip(), "".join(seq.split())))
    return records


def classify_family(head: str) -> str | None:
    low = head.lower()
    gn_m = re.search(r"gn=([a-z0-9_-]+)", low)
    gn = gn_m.group(1) if gn_m else ""
    # exclusions : biosynthèse du cofacteur, maturation, homologues photosynthèse
    if "cofactor biosynthesis" in low or "nifen" in low.replace(" ", ""):
        return None
    if "maturation protein" in low:
        return None
    if gn.startswith(("anf", "vnf", "bch", "chl")):
        return None
    if "nitrogenase iron protein" in low or re.search(r"\bnifh\b", low):
        if gn.startswith("nifh") or gn == "" :
            return "NifH"
    if ("alpha chain" in low or "subunit alpha" in low) and "nitrogenase" in low:
        if gn.startswith("nifd") or gn == "":
            return "NifD"
    if ("beta chain" in low or "subunit beta" in low) and "nitrogenase" in low:
        if gn.startswith("nifk") or gn == "":
            return "NifK"
    return None


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


def main():
    t0 = time.time()
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    prov = pd.read_csv(P3)
    strict = set(prov.loc[prov["corpus_strict"] == "oui", "souche"].astype(str))
    label_of = dict(zip(prov["souche"].astype(str), prov["label"].astype(int)))

    fasta_path = {}
    for d in FASTA_DIRS:
        for f in d.glob("*.fasta"):
            fasta_path[f.name] = f

    # vecteurs par famille et par souche + comptages séquences
    fam_vectors: dict[str, dict[str, list[np.ndarray]]] = {fam: {} for fam in FAMILIES}
    seq_counts = []
    for souche in prov["souche"].astype(str):
        path = fasta_path.get(f"{souche}.fasta")
        if path is None:
            continue
        per_fam: dict[str, int] = {fam: 0 for fam in FAMILIES}
        for head, seq in read_fasta_records(path):
            fam = classify_family(head)
            if fam is None:
                continue
            fam_vectors[fam].setdefault(souche, []).append(cksaap_vec(seq))
            per_fam[fam] += 1
        seq_counts.append({"souche": souche, "label": label_of[souche],
                           **{f"n_seq_{fam}": per_fam[fam] for fam in FAMILIES}})
    counts = pd.DataFrame(seq_counts)

    # tableau famille x classe (souches ayant au moins une séquence)
    cross_rows = []
    for corpus, keep in [("full", None), ("strict_489", strict)]:
        sub = counts if keep is None else counts[counts["souche"].isin(keep)]
        for fam in FAMILIES:
            has = sub[sub[f"n_seq_{fam}"] > 0]
            cross_rows.append({
                "famille": fam, "corpus": corpus,
                "n_souches": int(len(has)),
                "n_aerobie": int((has["label"] == 1).sum()),
                "n_anaerobie": int((has["label"] == 0).sum()),
                "n_sequences_total": int(has[f"n_seq_{fam}"].sum()),
                "copies_par_souche_moy": float(has[f"n_seq_{fam}"].mean()) if len(has) else np.nan,
            })
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(OUT_METRICS / "phase6c_nif_family_counts.csv", index=False)
    print(cross.to_string(index=False))

    rows = []
    for fam in FAMILIES:
        souches = sorted(fam_vectors[fam].keys())
        X = np.vstack([np.mean(fam_vectors[fam][s], axis=0) for s in souches])
        y = np.array([label_of[s] for s in souches], dtype=int)
        groups = np.array([s.split("_")[0] for s in souches])
        for corpus, keep in [("full", None), ("strict_489", strict)]:
            if keep is None:
                idx = np.arange(len(souches))
            else:
                idx = np.array([i for i, s in enumerate(souches) if s in keep])
            Xs, ys, gs = X[idx], y[idx], groups[idx]
            if len(np.unique(ys)) < 2 or len(ys) < 40:
                continue
            for seed in SEEDS:
                strat = ys if min(np.bincount(ys.astype(int))) >= 2 else None
                Xtr, Xte, ytr, yte = train_test_split(
                    Xs, ys, test_size=0.2, random_state=seed, stratify=strat)
                m = fit_eval(Xtr, Xte, ytr, yte, seed)
                rows.append({"feature": f"{fam}_CKSAAP", "corpus": corpus,
                             "protocol": "random_80_20_SMOTE", "seed": seed,
                             "n": int(len(ys)), **m})
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                tr, te = next(gss.split(Xs, ys, gs))
                m = fit_eval(Xs[tr], Xs[te], ys[tr], ys[te], seed)
                rows.append({"feature": f"{fam}_CKSAAP", "corpus": corpus,
                             "protocol": "group_shuffle_genus_SMOTE_train", "seed": seed,
                             "n": int(len(ys)),
                             "genres_overlap": int(len(set(gs[tr]) & set(gs[te]))), **m})
            print(f"{fam} | {corpus} ok ({time.time()-t0:.0f}s)")

    per = pd.DataFrame(rows)
    per.to_csv(OUT_METRICS / "phase6c_nif_families_per_repeat.csv", index=False)
    summary = (
        per.groupby(["feature", "corpus", "protocol"])
        .agg(n_repeats=("seed", "count"), accuracy_mean=("accuracy", "mean"),
             accuracy_sd=("accuracy", "std"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
             auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
             recall_aerobie_mean=("recall_aerobie", "mean"),
             recall_anaerobie_mean=("recall_anaerobie", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase6c_nif_families_summary.csv", index=False)

    ref = pd.read_csv(PHASE5_SUMMARY)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        feats = ["Nif_CKSAAP", "NifH_CKSAAP", "NifD_CKSAAP", "NifK_CKSAAP"]
        labels = ["Nif combiné", "NifH", "NifD", "NifK"]
        rand_m, rand_s, gen_m, gen_s = [], [], [], []
        for feat in feats:
            if feat == "Nif_CKSAAP":
                src, name = ref, "Nif_CKSAAP"
            else:
                src, name = summary, feat
            r = src[(src.feature == name) & (src.corpus == corpus)
                    & (src.protocol == "random_80_20_SMOTE")]
            g = src[(src.feature == name) & (src.corpus == corpus)
                    & (src.protocol == "group_shuffle_genus_SMOTE_train")]
            rand_m.append(float(r.auc_mean.iloc[0]) if len(r) else np.nan)
            rand_s.append(float(r.auc_sd.iloc[0]) if len(r) else 0.0)
            gen_m.append(float(g.auc_mean.iloc[0]) if len(g) else np.nan)
            gen_s.append(float(g.auc_sd.iloc[0]) if len(g) else 0.0)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, rand_m, width=0.4, yerr=rand_s, capsize=3, color="#1f4e79", label="aléatoire")
        ax.bar(x + 0.2, gen_m, width=0.4, yerr=gen_s, capsize=3, color="#8B4513", label="genre")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 6c — NifH / NifD / NifK séparément contre Nif combiné")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase6c_nif_families.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    def get(feature, corpus, protocol):
        s = summary[(summary.feature == feature) & (summary.corpus == corpus)
                    & (summary.protocol == protocol)]
        return float(s.auc_mean.iloc[0]) if len(s) else float("nan")

    lines = [
        "PHASE 6c — NifH / NifD / NifK SÉPARÉMENT (CKSAAP, protocole Phase 5 répété)",
        "",
        "Tableau famille x classe : voir phase6c_nif_family_counts.csv",
        "",
        "AUC moyenne (20 splits) :",
    ]
    for fam in FAMILIES:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {fam}_CKSAAP {corpus:11s} aléatoire={get(fam+'_CKSAAP', corpus, 'random_80_20_SMOTE'):.3f}"
                f" | genre={get(fam+'_CKSAAP', corpus, 'group_shuffle_genus_SMOTE_train'):.3f}"
            )
    lines += [
        "",
        "LECTURE :",
        "- Comparer chaque famille au Nif combiné (Phase 5) et, mentalement, aux",
        "  contrôles housekeeping : une famille seule qui égale le combiné indique",
        "  où se concentre le signal (ou la confusion) lu par le modèle.",
        "- Classification des séquences par produit annoté UniProt, anf/vnf/bch",
        "  exclus, biosynthèse du cofacteur exclue ; copies moyennées par souche.",
        "",
        f"Durée totale : {time.time()-t0:.0f}s",
        f"Comptages : {OUT_METRICS / 'phase6c_nif_family_counts.csv'}",
        f"Par répétition : {OUT_METRICS / 'phase6c_nif_families_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase6c_nif_families_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE6C_NIF_FAMILIES_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
