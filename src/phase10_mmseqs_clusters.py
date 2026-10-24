# -*- coding: utf-8 -*-
"""
Phase 10 — Split par clusters de séquences MMseqs2 (30 % et 50 % d'identité).

Toutes les séquences des FASTA souches (cluster nif complet, 6 710
séquences) sont regroupées par MMseqs2 easy-cluster à --min-seq-id
0.3 puis 0.5. Deux souches reliées dès qu'une de leurs séquences tombe
dans le même cluster (union-find) ; les composantes connexes forment
les groupes du split : aucun cluster de séquences ne peut chevaucher
train et test. GroupShuffleSplit sur ces composantes, 20 graines,
HGB + CKSAAP (Nif combiné historique et RecA), corpus Full/Strict,
SMOTE train uniquement. Composition de chaque test rapportée.

Prérequis : tools/**/mmseqs.exe (src/download_mmseqs.py).
Sorties :
- results/metrics/phase10_mmseqs_components_{30,50}.csv
- results/metrics/phase10_mmseqs_per_repeat.csv
- results/metrics/phase10_mmseqs_summary.csv
- results/figures/phase10_mmseqs_decay.png
- results/metrics/PHASE10_MMSEQS_DECISION.txt
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
WORK = ROOT / "results" / "models" / "mmseqs_work"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
SEEDS = list(range(42, 62))
THRESHOLDS = [0.3, 0.5]


def find_mmseqs() -> Path:
    exes = sorted((ROOT / "tools").rglob("mmseqs.exe"))
    if not exes:
        raise FileNotFoundError("mmseqs.exe introuvable sous tools/ — lancer src/download_mmseqs.py")
    return exes[0]


def build_all_fasta() -> Path:
    from phase6c_nif_families import FASTA_DIRS, read_fasta_records

    prov = pd.read_csv(P3)
    fasta_path = {}
    for d in FASTA_DIRS:
        for f in d.glob("*.fasta"):
            fasta_path[f.name] = f
    out = WORK / "all_sequences.fasta"
    WORK.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for souche in prov["souche"].astype(str):
            path = fasta_path.get(f"{souche}.fasta")
            if path is None:
                continue
            for i, (_head, seq) in enumerate(read_fasta_records(path)):
                seq = "".join(seq.split()).upper()
                if len(seq) < 30:
                    continue
                fh.write(f">{souche}|{i}\n{seq}\n")
                n += 1
    print(f"FASTA global : {n} séquences -> {out}", flush=True)
    return out


def run_cluster(mmseqs: Path, fasta: Path, thr: float) -> dict[str, int]:
    tag = f"id{int(thr * 100)}"
    prefix = WORK / f"clu_{tag}"
    tsv = Path(str(prefix) + "_cluster.tsv")
    if not tsv.exists():
        # MODIFIÉ : easy-cluster/cluster/search lancent des scripts shell
        # (inutilisables avec ce binaire Windows sans /bin/sh). Enchaînement
        # des sous-commandes binaires natives équivalentes, workflow bas
        # niveau documenté de MMseqs2 : createdb -> prefilter -> align
        # -> clust -> createtsv. Threads limités à 2 (machine partagée).
        seqdb = WORK / "seqdb"
        if not Path(str(seqdb) + ".dbtype").exists():
            subprocess.run([str(mmseqs), "createdb", str(fasta), str(seqdb), "-v", "1"], check=True)
        pre = WORK / f"pre_{tag}"
        aln = WORK / f"aln_{tag}"
        cludb = WORK / f"cludb_{tag}"
        # identité/couverture sont appliquées à l'étape align ; clust ne
        # prend que les trois bases. Reprise : align existant = on saute.
        if not Path(str(aln) + ".dbtype").exists():
            subprocess.run([str(mmseqs), "prefilter", str(seqdb), str(seqdb), str(pre),
                            "--add-self-matches", "1",
                            "--threads", "2", "-v", "1"], check=True)
            subprocess.run([str(mmseqs), "align", str(seqdb), str(seqdb), str(pre), str(aln),
                            "--min-seq-id", str(thr), "-c", "0.8", "--cov-mode", "0",
                            "--threads", "2", "-v", "1"], check=True)
        subprocess.run([str(mmseqs), "clust", str(seqdb), str(aln), str(cludb), "-v", "1"], check=True)
        subprocess.run([str(mmseqs), "createtsv", str(seqdb), str(seqdb), str(cludb), str(tsv), "-v", "1"], check=True)
    seq_cluster: dict[str, int] = {}
    cid = 0
    with tsv.open(encoding="utf-8") as fh:
        current_rep = None
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")[:2]
            if rep != current_rep:
                current_rep = rep
                cid += 1
            seq_cluster[member] = cid
    print(f"seuil {thr} : {cid} clusters pour {len(seq_cluster)} séquences", flush=True)
    return seq_cluster


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def strain_components(seq_cluster: dict[str, int], souches: list[str]) -> np.ndarray:
    dsu = DSU()
    by_cluster: dict[int, list[str]] = {}
    for seq_id, cid in seq_cluster.items():
        by_cluster.setdefault(cid, []).append(seq_id.split("|")[0])
    for strains in by_cluster.values():
        uniq = sorted(set(strains))
        for other in uniq[1:]:
            dsu.union(uniq[0], other)
    comp_ids = {}
    groups = []
    for s in souches:
        r = dsu.find(s) if s in dsu.p else f"seule:{s}"
        if r not in comp_ids:
            comp_ids[r] = len(comp_ids)
        groups.append(comp_ids[r])
    return np.array(groups)


def main():
    t0 = time.time()
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        matthews_corrcoef,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit

    from phase1_premium_complete import load_control_cksaap, load_nif_cksaap

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    p3 = pd.read_csv(P3)
    strict = set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))

    mmseqs = find_mmseqs()
    print("mmseqs :", mmseqs, flush=True)
    fasta = build_all_fasta()

    datasets = {}
    s, X, y, g = load_nif_cksaap()
    datasets["Nif_CKSAAP"] = ([str(v) for v in s], X, y)
    s, X, y, g = load_control_cksaap("RecA")
    datasets["RecA_CKSAAP"] = ([str(v) for v in s], X, y)

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
    comp_tables = {}
    for thr in THRESHOLDS:
        seq_cluster = run_cluster(mmseqs, fasta, thr)
        for feature, (souches, X, y) in datasets.items():
            groups_all = strain_components(seq_cluster, souches)
            comp_tables.setdefault(thr, {})[feature] = (souches, groups_all)
            for corpus, keep in [("full", None), ("strict_489", strict)]:
                if keep is None:
                    idx = np.arange(len(souches))
                else:
                    idx = np.array([i for i, s in enumerate(souches) if s in keep])
                Xs, ys, gs = X[idx], y[idx], groups_all[idx]
                df_g = pd.DataFrame({"g": gs, "y": ys}).groupby("g")["y"].agg(["count", "mean"])
                mono_share = float(((df_g["mean"] == 0) | (df_g["mean"] == 1)).mean())
                for seed in SEEDS:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    tr, te = next(gss.split(Xs, ys, gs))
                    m = fit_eval(Xs[tr], Xs[te], ys[tr], ys[te], seed)
                    rows.append({
                        "feature": feature, "corpus": corpus, "min_seq_id": thr,
                        "seed": seed, "n": int(len(ys)),
                        "n_components_corpus": int(len(df_g)),
                        "mono_class_component_share": mono_share,
                        "largest_component_share": float(df_g["count"].max() / len(ys)),
                        "n_components_test": int(len(set(gs[te]))),
                        "n_test": int(len(te)),
                        "n_test_aerobie": int((ys[te] == 1).sum()),
                        "n_test_anaerobie": int((ys[te] == 0).sum()),
                        "components_overlap": int(len(set(gs[tr]) & set(gs[te]))),
                        **m,
                    })
                print(f"mmseqs {thr} | {feature} | {corpus} ok ({time.time() - t0:.0f}s)", flush=True)

    for thr, per_feature in comp_tables.items():
        souches, groups_all = per_feature["Nif_CKSAAP"]
        pd.DataFrame({"souche": souches, "component": groups_all}).to_csv(
            OUT_METRICS / f"phase10_mmseqs_components_{int(thr * 100)}.csv", index=False)

    per = pd.DataFrame(rows)
    per.to_csv(OUT_METRICS / "phase10_mmseqs_per_repeat.csv", index=False)
    summary = (
        per.groupby(["feature", "corpus", "min_seq_id"])
        .agg(n_repeats=("seed", "count"),
             accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
             auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
             auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
             recall_aerobie_mean=("recall_aerobie", "mean"),
             recall_anaerobie_mean=("recall_anaerobie", "mean"),
             mono_share_mean=("mono_class_component_share", "mean"),
             largest_share_mean=("largest_component_share", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase10_mmseqs_summary.csv", index=False)

    import matplotlib.pyplot as plt

    # Phase 9 peut encore tourner : sa synthèse est optionnelle pour la figure.
    try:
        ref9 = pd.read_csv(OUT_METRICS / "phase9_family_order_summary.csv")
    except FileNotFoundError:
        ref9 = None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        labels = ["genre", "famille", "ordre", "MMseqs 50 %", "MMseqs 30 %"]
        nif_m, nif_s, rec_m, rec_s = [], [], [], []
        for lab in labels:
            if lab.startswith("MMseqs"):
                thr = 0.5 if "50" in lab else 0.3
                sn = summary[(summary.feature == "Nif_CKSAAP") & (summary.corpus == corpus) & (summary.min_seq_id == thr)]
                sr = summary[(summary.feature == "RecA_CKSAAP") & (summary.corpus == corpus) & (summary.min_seq_id == thr)]
            else:
                lvl = {"genre": None, "famille": "family", "ordre": "order"}[lab]
                if lvl is None:
                    ref5 = pd.read_csv(OUT_METRICS / "phase5_repeated_splits_summary.csv")
                    sn = ref5[(ref5.feature == "Nif_CKSAAP") & (ref5.corpus == corpus) & (ref5.protocol == "group_shuffle_genus_SMOTE_train")]
                    sr = ref5[(ref5.feature == "RecA_CKSAAP") & (ref5.corpus == corpus) & (ref5.protocol == "group_shuffle_genus_SMOTE_train")]
                elif ref9 is not None:
                    sn = ref9[(ref9.feature == "Nif_CKSAAP") & (ref9.corpus == corpus) & (ref9.level == lvl)]
                    sr = ref9[(ref9.feature == "RecA_CKSAAP") & (ref9.corpus == corpus) & (ref9.level == lvl)]
                else:
                    sn = sr = pd.DataFrame()
            nif_m.append(float(sn.auc_mean.iloc[0]) if len(sn) else np.nan)
            nif_s.append(float(sn.auc_sd.iloc[0]) if len(sn) else 0.0)
            rec_m.append(float(sr.auc_mean.iloc[0]) if len(sr) else np.nan)
            rec_s.append(float(sr.auc_sd.iloc[0]) if len(sr) else 0.0)
        x = np.arange(len(labels))
        ax.errorbar(x - 0.07, nif_m, yerr=nif_s, marker="o", capsize=3, color="#BC5A28", label="Nif")
        ax.errorbar(x + 0.07, rec_m, yerr=rec_s, marker="s", capsize=3, color="#66742E", label="RecA")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 10 — Décroissance : genre, famille, ordre, clusters MMseqs2")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase10_mmseqs_decay.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    def get(feature, corpus, thr):
        s = summary[(summary.feature == feature) & (summary.corpus == corpus) & (summary.min_seq_id == thr)]
        return float(s.auc_mean.iloc[0]) if len(s) else float("nan")

    # Garde-fou : composante géante = découpage dégénéré, AUC non citable.
    max_largest = float(summary["largest_share_mean"].max()) if len(summary) else 0.0
    lines = [
        "PHASE 10 — SPLITS PAR CLUSTERS MMseqs2 (30 % et 50 % d'identité, 20 tirages)",
        "Groupes = composantes de souches reliées par un cluster de séquences commun.",
    ]
    if max_largest > 0.5:
        lines += [
            "",
            "⚠️ RÉSULTAT DÉGÉNÉRÉ — NE PAS CITER COMME UNE PERFORMANCE.",
            f"Composante géante = {max_largest:.1%} des souches : le train est",
            "réduit à la composante minuscule, l'AUC de 0,500 est un artefact",
            "du découpage. Voir l'analyse dans ce fichier (version complète",
            "dans l'historique de génération) et le protocole par protéine.",
        ]
    lines += [
        "",
        "AUC moyenne (20 splits) :",
    ]
    for feature in ["Nif_CKSAAP", "RecA_CKSAAP"]:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {feature:12s} {corpus:11s} MMseqs50={get(feature, corpus, 0.5):.3f}"
                f" | MMseqs30={get(feature, corpus, 0.3):.3f}"
            )
    lines += [
        "",
        "LECTURE : le split par cluster supprime le dernier canal de fuite",
        "(séquences quasi identiques partagées entre train et test).",
        "Si la performance s'effondre à 30 %, le score reposait sur la",
        "reconnaissance de séquences proches, pas sur une signature généralisable.",
        "",
        f"Durée totale : {time.time() - t0:.0f}s",
        f"Par répétition : {OUT_METRICS / 'phase10_mmseqs_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase10_mmseqs_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE10_MMSEQS_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
