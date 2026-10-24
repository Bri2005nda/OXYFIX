# -*- coding: utf-8 -*-
"""
Phase 7 — Embeddings ESM-2 650M (gelés) sous le protocole identique.

Descripteur remplacé : CKSAAP -> représentation ESM-2
(facebook/esm2_t33_650M_UR50D, modèle gelé, moyenne des vecteurs
contextuels des résidus, hors tokens spéciaux). Le reste du protocole
est celui de la Phase 5 : corpus Full/Strict, splits aléatoire et
genre, 20 graines (42..61), HGB, SMOTE train uniquement.

Jeux de séquences :
- Nif_ESM : séquences classées NifH/NifD/NifK (Phase 6c) des FASTA
  souches ; moyenne des embeddings de séquences par souche.
- RecA_ESM : séquence RecA de contrôle (une par souche).

Extraction par morceaux avec reprise : chaque morceau de séquences est
sauvegardé dans results/models/esm_cache/ ; un redémarrage saute les
morceaux déjà calculés.

Sorties :
- results/models/esm_cache/ (morceaux d'embeddings par séquence)
- results/metrics/phase7_esm_per_repeat.csv
- results/metrics/phase7_esm_summary.csv
- results/figures/phase7_esm_vs_cksaap.png
- results/metrics/PHASE7_ESM_DECISION.txt
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
CACHE = ROOT / "results" / "models" / "esm_cache"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
CTRL_META = ROOT / "data" / "raw" / "controls_phase1" / "controls_fetch_table.csv"
CTRL_FASTA = ROOT / "data" / "raw" / "controls_phase1" / "fasta"
PHASE5_SUMMARY = OUT_METRICS / "phase5_repeated_splits_summary.csv"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
SEEDS = list(range(42, 62))
CHUNK = 64


# ------------------------------------------------------------ séquences
def collect_sequences() -> dict[str, list[tuple[str, str, str]]]:
    """dataset -> liste (souche, tag, sequence)."""
    from phase6c_nif_families import FASTA_DIRS, classify_family, read_fasta_records

    prov = pd.read_csv(P3)
    fasta_path = {}
    for d in FASTA_DIRS:
        for f in d.glob("*.fasta"):
            fasta_path[f.name] = f
    nif: list[tuple[str, str, str]] = []
    for souche in prov["souche"].astype(str):
        path = fasta_path.get(f"{souche}.fasta")
        if path is None:
            continue
        for head, seq in read_fasta_records(path):
            fam = classify_family(head)
            if fam is not None:
                nif.append((souche, fam, seq))

    meta = pd.read_csv(CTRL_META)
    sub = meta[(meta["gene"] == "RecA") & (meta["status"].astype(str).str.startswith("ok"))]
    reca: list[tuple[str, str, str]] = []
    for _, row in sub.iterrows():
        path = Path(str(row["fasta_path"])) if row.get("fasta_path") else CTRL_FASTA / "RecA" / f"{row['souche']}.fasta"
        if not path.exists():
            path = CTRL_FASTA / "RecA" / f"{row['souche']}.fasta"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        seq = "".join(l.strip() for l in text.splitlines() if l and not l.startswith(">"))
        if seq:
            reca.append((str(row["souche"]), "RecA", seq))
    return {"Nif_ESM": nif, "RecA_ESM": reca}


# ------------------------------------------------------------ extraction
def embed_dataset(name: str, records: list[tuple[str, str, str]], tok, model, torch):
    """Embeddings par séquence, par morceaux sauvegardés (reprise auto)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # tri par longueur pour des batchs homogènes (ordre retenu dans les métadonnées)
    order = np.argsort([len(r[2]) for r in records])
    sorted_records = [records[i] for i in order]
    meta_rows = []
    n_chunks = (len(sorted_records) + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        chunk = sorted_records[ci * CHUNK:(ci + 1) * CHUNK]
        chunk_file = CACHE / f"{name}_chunk{ci:04d}.npz"
        if chunk_file.exists():
            data = np.load(chunk_file)
            embs = data["emb"]
            print(f"{name} morceau {ci + 1}/{n_chunks} déjà calculé", flush=True)
        else:
            seqs = [r[2] for r in chunk]
            embs_list = []
            with torch.no_grad():
                for start in range(0, len(seqs), 8):
                    batch = seqs[start:start + 8]
                    inp = tok(batch, return_tensors="pt", padding=True)
                    out = model(**inp).last_hidden_state
                    mask = inp["attention_mask"]
                    for j in range(len(batch)):
                        # tokens réels hors CLS/EOS : positions 1..len-2
                        L = int(mask[j].sum().item())
                        vec = out[j, 1:L - 1].mean(dim=0).cpu().numpy()
                        embs_list.append(vec)
            embs = np.vstack(embs_list).astype(np.float32)
            np.savez_compressed(chunk_file, emb=embs)
            print(f"{name} morceau {ci + 1}/{n_chunks} calculé ({len(chunk)} séq.)", flush=True)
        for (souche, tag, _seq), vec in zip(chunk, embs):
            meta_rows.append({"dataset": name, "souche": souche, "tag": tag, "vec": vec})
    return meta_rows


def strain_matrices(meta_rows: list[dict], label_of: dict[str, int]):
    """Moyenne des embeddings de séquences -> une ligne par souche."""
    datasets: dict[str, tuple[list[str], np.ndarray, np.ndarray, np.ndarray]] = {}
    by_ds: dict[str, dict[str, list[np.ndarray]]] = {}
    for r in meta_rows:
        by_ds.setdefault(r["dataset"], {}).setdefault(r["souche"], []).append(r["vec"])
    for ds, per_strain in by_ds.items():
        souches = sorted(per_strain.keys())
        X = np.vstack([np.mean(per_strain[s], axis=0) for s in souches])
        y = np.array([label_of[s] for s in souches], dtype=int)
        groups = np.array([s.split("_")[0] for s in souches])
        datasets[ds] = (souches, X, y, groups)
    return datasets


# ------------------------------------------------------------ évaluation
def evaluate(datasets, strict: set[str]):
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

    rows = []
    for feature, (souches, X, y, groups) in datasets.items():
        for corpus, keep in [("full", None), ("strict_489", strict)]:
            if keep is None:
                idx = np.arange(len(souches))
            else:
                idx = np.array([i for i, s in enumerate(souches) if s in keep])
            Xs, ys, gs = X[idx], y[idx], groups[idx]
            for seed in SEEDS:
                strat = ys if min(np.bincount(ys.astype(int))) >= 2 else None
                Xtr, Xte, ytr, yte = train_test_split(
                    Xs, ys, test_size=0.2, random_state=seed, stratify=strat)
                m = fit_eval(Xtr, Xte, ytr, yte, seed)
                rows.append({"feature": feature, "corpus": corpus,
                             "protocol": "random_80_20_SMOTE", "seed": seed,
                             "n": int(len(ys)), **m})
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                tr, te = next(gss.split(Xs, ys, gs))
                m = fit_eval(Xs[tr], Xs[te], ys[tr], ys[te], seed)
                rows.append({"feature": feature, "corpus": corpus,
                             "protocol": "group_shuffle_genus_SMOTE_train", "seed": seed,
                             "n": int(len(ys)),
                             "genres_overlap": int(len(set(gs[tr]) & set(gs[te]))), **m})
            print(f"éval {feature} | {corpus} ok", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"chargement {MODEL_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"modèle chargé ({time.time() - t0:.0f}s)", flush=True)

    seqs = collect_sequences()
    for name, records in seqs.items():
        print(f"{name}: {len(records)} séquences, {len(set(r[0] for r in records))} souches", flush=True)

    meta_rows: list[dict] = []
    for name, records in seqs.items():
        meta_rows.extend(embed_dataset(name, records, tok, model, torch))
        print(f"{name}: extraction terminée ({time.time() - t0:.0f}s)", flush=True)

    if args.extract_only:
        print("extraction seule demandée — fin")
        return

    prov = pd.read_csv(P3)
    label_of = dict(zip(prov["souche"].astype(str), prov["label"].astype(int)))
    strict = set(prov.loc[prov["corpus_strict"] == "oui", "souche"].astype(str))
    datasets = strain_matrices(meta_rows, label_of)
    for ds, (souches, X, y, _g) in datasets.items():
        print(f"{ds}: matrice souches {X.shape}", flush=True)

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    per = evaluate(datasets, strict)
    per.to_csv(OUT_METRICS / "phase7_esm_per_repeat.csv", index=False)
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
    summary.to_csv(OUT_METRICS / "phase7_esm_summary.csv", index=False)

    import matplotlib.pyplot as plt

    ref = pd.read_csv(PHASE5_SUMMARY)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        feats = [("Nif_ESM", "Nif_CKSAAP", "Nif"), ("RecA_ESM", "RecA_CKSAAP", "RecA")]
        labels, esm_m, esm_s, ck_m, ck_s = [], [], [], [], []
        for esm_name, ck_name, short in feats:
            for protocol, pname in [("random_80_20_SMOTE", "aléatoire"),
                                    ("group_shuffle_genus_SMOTE_train", "genre")]:
                e = summary[(summary.feature == esm_name) & (summary.corpus == corpus)
                            & (summary.protocol == protocol)]
                c = ref[(ref.feature == ck_name) & (ref.corpus == corpus)
                        & (ref.protocol == protocol)]
                if len(e) and len(c):
                    labels.append(f"{short}\n{pname}")
                    esm_m.append(float(e.auc_mean.iloc[0])); esm_s.append(float(e.auc_sd.iloc[0]))
                    ck_m.append(float(c.auc_mean.iloc[0])); ck_s.append(float(c.auc_sd.iloc[0]))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, esm_m, width=0.4, yerr=esm_s, capsize=3, color="#5B2C6F", label="ESM-2 650M")
        ax.bar(x + 0.2, ck_m, width=0.4, yerr=ck_s, capsize=3, color="#1f4e79", label="CKSAAP")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 7 — Embeddings ESM-2 650M contre CKSAAP (protocole identique)")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase7_esm_vs_cksaap.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    def get(feature, corpus, protocol):
        s = summary[(summary.feature == feature) & (summary.corpus == corpus)
                    & (summary.protocol == protocol)]
        return float(s.auc_mean.iloc[0]) if len(s) else float("nan")

    lines = [
        "PHASE 7 — EMBEDDINGS ESM-2 650M (gelés) CONTRE CKSAAP",
        "Protocole Phase 5 répété ; moyenne des embeddings par souche.",
        "",
        "AUC moyenne (20 splits) :",
    ]
    for feature in ["Nif_ESM", "RecA_ESM"]:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {feature:10s} {corpus:11s} aléatoire={get(feature, corpus, 'random_80_20_SMOTE'):.3f}"
                f" | genre={get(feature, corpus, 'group_shuffle_genus_SMOTE_train'):.3f}"
            )
    lines += [
        "",
        "LECTURE :",
        "- Comparer Nif_ESM à RecA_ESM (contrôle) avant de comparer à CKSAAP :",
        "  un gain qui touche aussi le contrôle est un gain de lecture de",
        "  l'organisme, pas un signal propre à la nitrogénase.",
        "- Embeddings gelés (pas de fine-tuning) ; modèle préentraîné sur des",
        "  protéines publiques, sans nos étiquettes : pas de fuite d'étiquettes,",
        "  mais les représentations contiennent l'histoire évolutive des protéines.",
        "",
        f"Durée totale : {time.time() - t0:.0f}s",
        f"Par répétition : {OUT_METRICS / 'phase7_esm_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase7_esm_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE7_ESM_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
