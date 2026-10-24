# -*- coding: utf-8 -*-
"""
Phase 12 — Splits par clusters MMseqs2 PAR PROTÉINE (protocole corrigé).

La Phase 10 (union sur les 6 705 séquences du cluster complet) était
dégénérée : une composante géante réunissait 99,9 % des souches. Ici,
le clustering est fait séparément sur les séquences de chaque protéine
descripteur (NifH, NifD, NifK, RecA), aux seuils 30 % et 50 %
d'identité (couverture 0,8), par les sous-commandes natives MMseqs2
(createdb -> prefilter -> align -> clust -> createtsv, threads = 2).
Deux souches sont reliées si une de leurs séquences de CETTE protéine
tombe dans le même cluster (union-find) ; split GroupShuffleSplit sur
les composantes, 20 graines, HGB + CKSAAP de la même protéine,
SMOTE train uniquement, corpus Full/Strict. Un bloc « Nif_combine »
relie les souches via l'union des clusters H, D et K et évalue le
CKSAAP historique combiné. Garde-fou : composante géante > 50 % =
résultat marqué dégénéré, non citable.

Sorties :
- results/metrics/phase12_mmseqs_protein_per_repeat.csv
- results/metrics/phase12_mmseqs_protein_summary.csv
- results/figures/phase12_mmseqs_by_protein.png
- results/metrics/PHASE12_MMSEQS_PROTEIN_DECISION.txt
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
WORK = ROOT / "results" / "models" / "mmseqs_by_protein"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
SEEDS = list(range(42, 62))
THRESHOLDS = [0.3, 0.5]
PROTEINS = ["NifH", "NifD", "NifK", "RecA"]


def find_mmseqs() -> Path:
    exes = sorted((ROOT / "tools").rglob("mmseqs.exe"))
    if not exes:
        raise FileNotFoundError("mmseqs.exe introuvable sous tools/")
    return exes[0]


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


def cluster_protein(mmseqs: Path, protein: str, records: list[tuple[str, str]], thr: float) -> dict[str, int]:
    """records : (souche, sequence). Retourne seq_id -> cluster."""
    tag = f"{protein}_id{int(thr * 100)}"
    wdir = WORK / tag
    wdir.mkdir(parents=True, exist_ok=True)
    tsv = wdir / "clusters.tsv"
    if not tsv.exists():
        fasta = wdir / "seqs.fasta"
        with fasta.open("w", encoding="utf-8") as fh:
            for i, (souche, seq) in enumerate(records):
                fh.write(f">{souche}|{i}\n{''.join(seq.split()).upper()}\n")
        seqdb = wdir / "seqdb"
        if not Path(str(seqdb) + ".dbtype").exists():
            subprocess.run([str(mmseqs), "createdb", str(fasta), str(seqdb), "-v", "1"], check=True)
        pre, aln, cludb = wdir / "pre", wdir / "aln", wdir / "cludb"
        if not Path(str(aln) + ".dbtype").exists():
            subprocess.run([str(mmseqs), "prefilter", str(seqdb), str(seqdb), str(pre),
                            "--add-self-matches", "1", "--threads", "2", "-v", "1"], check=True)
            subprocess.run([str(mmseqs), "align", str(seqdb), str(seqdb), str(pre), str(aln),
                            "--min-seq-id", str(thr), "-c", "0.8", "--cov-mode", "0",
                            "--threads", "2", "-v", "1"], check=True)
        subprocess.run([str(mmseqs), "clust", str(seqdb), str(aln), str(cludb), "-v", "1"], check=True)
        subprocess.run([str(mmseqs), "createtsv", str(seqdb), str(seqdb), str(cludb), str(tsv), "-v", "1"], check=True)
    seq_cluster: dict[str, int] = {}
    cid, current = 0, None
    with tsv.open(encoding="utf-8") as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")[:2]
            if rep != current:
                current, cid = rep, cid + 1
            seq_cluster[member] = cid
    print(f"{tag} : {cid} clusters / {len(seq_cluster)} séquences", flush=True)
    return seq_cluster


def components(seq_cluster: dict[str, int], souches: list[str]) -> np.ndarray:
    dsu = DSU()
    by_cluster: dict[int, list[str]] = {}
    for seq_id, cid in seq_cluster.items():
        by_cluster.setdefault(cid, []).append(seq_id.split("|")[0])
    for strains in by_cluster.values():
        uniq = sorted(set(strains))
        for other in uniq[1:]:
            dsu.union(uniq[0], other)
    ids, groups = {}, []
    for s in souches:
        r = dsu.find(s) if s in dsu.p else f"seule:{s}"
        if r not in ids:
            ids[r] = len(ids)
        groups.append(ids[r])
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

    from phase1_premium_complete import cksaap_vec, load_control_cksaap, load_nif_cksaap
    from phase7_esm_embeddings import collect_sequences

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    prov = pd.read_csv(P3)
    label_of = dict(zip(prov["souche"].astype(str), prov["label"].astype(int)))
    strict = set(prov.loc[prov["corpus_strict"] == "oui", "souche"].astype(str))

    seqs = collect_sequences()
    per_protein: dict[str, list[tuple[str, str]]] = {p: [] for p in PROTEINS}
    for souche, fam, seq in seqs["Nif_ESM"]:
        if fam in per_protein:
            per_protein[fam].append((souche, seq))
    for souche, _tag, seq in seqs["RecA_ESM"]:
        per_protein["RecA"].append((souche, seq))
    for p in PROTEINS:
        print(f"{p} : {len(per_protein[p])} séquences", flush=True)

    # matrices CKSAAP par protéine (moyenne des copies par souche)
    matrices: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for p in ["NifH", "NifD", "NifK"]:
        by_strain: dict[str, list[np.ndarray]] = {}
        for souche, seq in per_protein[p]:
            by_strain.setdefault(souche, []).append(cksaap_vec(seq))
        souches = sorted(by_strain)
        X = np.vstack([np.mean(by_strain[s], axis=0) for s in souches])
        y = np.array([label_of[s] for s in souches], dtype=int)
        matrices[p] = (souches, X, y)
    s, X, y, _g = load_control_cksaap("RecA")
    matrices["RecA"] = ([str(v) for v in s], X, y)
    s, X, y, _g = load_nif_cksaap()
    matrices["Nif_combine"] = ([str(v) for v in s], X, y)

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

    skipped: list[str] = []

    def evaluate(feature, souches, X, y, groups_all, thr, rows):
        for corpus, keep in [("full", None), ("strict_489", strict)]:
            if keep is None:
                idx = np.arange(len(souches))
            else:
                idx = np.array([i for i, s in enumerate(souches) if s in keep])
            Xs, ys, gs = X[idx], y[idx], groups_all[idx]
            df_g = pd.DataFrame({"g": gs, "y": ys}).groupby("g")["y"].agg(["count", "mean"])
            # Garde-fou : trop peu de composantes = découpage impossible
            # (ex. NifK à 30 % : une seule composante chaine toutes les souches).
            if len(df_g) < 5:
                skipped.append(f"{feature}|{corpus}|id{int(thr * 100)} : {len(df_g)} composante(s), split impossible")
                print(f"SKIP {skipped[-1]}", flush=True)
                continue
            mono_share = float(((df_g["mean"] == 0) | (df_g["mean"] == 1)).mean())
            largest = float(df_g["count"].max() / len(ys))
            for seed in SEEDS:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                tr, te = next(gss.split(Xs, ys, gs))
                m = fit_eval(Xs[tr], Xs[te], ys[tr], ys[te], seed)
                rows.append({
                    "feature": feature, "corpus": corpus, "min_seq_id": thr, "seed": seed,
                    "n": int(len(ys)), "n_components_corpus": int(len(df_g)),
                    "mono_class_component_share": mono_share,
                    "largest_component_share": largest,
                    "n_components_test": int(len(set(gs[te]))), "n_test": int(len(te)),
                    "n_test_aerobie": int((ys[te] == 1).sum()),
                    "n_test_anaerobie": int((ys[te] == 0).sum()),
                    "components_overlap": int(len(set(gs[tr]) & set(gs[te]))),
                    **m,
                })
            print(f"eval {feature}|{corpus}|id{int(thr * 100)} ok ({time.time() - t0:.0f}s)", flush=True)

    mmseqs = find_mmseqs()
    rows = []
    comp_by_protein: dict[tuple[str, float], dict[str, int]] = {}
    for thr in THRESHOLDS:
        for p in PROTEINS:
            seq_cluster = cluster_protein(mmseqs, p, per_protein[p], thr)
            souches_p = matrices[p][0]
            groups = components(seq_cluster, souches_p)
            comp_by_protein[(p, thr)] = {s: int(g) for s, g in zip(souches_p, groups)}
            evaluate(p, matrices[p][0], matrices[p][1], matrices[p][2], groups, thr, rows)
        # bloc combiné : union des appartenances H, D, K
        souches_c, Xc, yc = matrices["Nif_combine"]
        dsu = DSU()
        for p in ["NifH", "NifD", "NifK"]:
            by_comp: dict[int, list[str]] = {}
            for s, g in comp_by_protein[(p, thr)].items():
                by_comp.setdefault(g, []).append(s)
            for strains in by_comp.values():
                uniq = sorted(set(strains))
                for other in uniq[1:]:
                    dsu.union(uniq[0], other)
        ids, groups_c = {}, []
        for s in souches_c:
            r = dsu.find(s) if s in dsu.p else f"seule:{s}"
            if r not in ids:
                ids[r] = len(ids)
            groups_c.append(ids[r])
        evaluate("Nif_combine", souches_c, Xc, yc, np.array(groups_c), thr, rows)

    per = pd.DataFrame(rows)
    per.to_csv(OUT_METRICS / "phase12_mmseqs_protein_per_repeat.csv", index=False)
    summary = (
        per.groupby(["feature", "corpus", "min_seq_id"])
        .agg(n_repeats=("seed", "count"),
             accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
             auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
             auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
             recall_aerobie_mean=("recall_aerobie", "mean"),
             recall_anaerobie_mean=("recall_anaerobie", "mean"),
             n_components_mean=("n_components_corpus", "mean"),
             mono_share_mean=("mono_class_component_share", "mean"),
             largest_share_mean=("largest_component_share", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase12_mmseqs_protein_summary.csv", index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        labels, nif_m, nif_s, rec_m, rec_s = [], [], [], [], []
        for feat, short in [("NifH", "NifH"), ("NifD", "NifD"), ("NifK", "NifK"), ("Nif_combine", "Nif comb.")]:
            for thr in THRESHOLDS:
                sn = summary[(summary.feature == feat) & (summary.corpus == corpus) & (summary.min_seq_id == thr)]
                labels.append(f"{short}\n{int(thr * 100)} %")
                nif_m.append(float(sn.auc_mean.iloc[0]) if len(sn) else np.nan)
                nif_s.append(float(sn.auc_sd.iloc[0]) if len(sn) else 0.0)
        sr50 = summary[(summary.feature == "RecA") & (summary.corpus == corpus)]
        x = np.arange(len(labels))
        ax.bar(x, nif_m, yerr=nif_s, capsize=3, color="#BC5A28", label="Protéines Nif (split par leurs clusters)")
        for i, thr in enumerate([0.5, 0.3]):
            sub = sr50[sr50.min_seq_id == thr]
            if len(sub):
                ax.axhline(float(sub.auc_mean.iloc[0]), color="#66742E", ls="--" if thr == 0.5 else ":", lw=1.6,
                           label=f"RecA {int(thr * 100)} % : {float(sub.auc_mean.iloc[0]):.3f}")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits par cluster)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Phase 12 — Splits par clusters MMseqs par protéine (protocole corrigé)")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase12_mmseqs_by_protein.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    max_largest = float(summary["largest_share_mean"].max()) if len(summary) else 0.0

    def get(feature, corpus, thr):
        s = summary[(summary.feature == feature) & (summary.corpus == corpus) & (summary.min_seq_id == thr)]
        return float(s.auc_mean.iloc[0]) if len(s) else float("nan")

    lines = [
        "PHASE 12 — SPLITS PAR CLUSTERS MMseqs PAR PROTÉINE (protocole corrigé, 20 tirages)",
        "Clustering séparé par protéine descripteur ; souches reliées via les",
        "clusters de cette protéine uniquement (union-find).",
    ]
    if max_largest > 0.5:
        lines += ["", f"⚠️ ATTENTION : plus grande composante = {max_largest:.1%} des souches ;",
                  "les blocs concernés restent dégénérés et non citables."]
    lines += ["", "AUC moyenne (20 splits) :"]
    for feature in ["NifH", "NifD", "NifK", "Nif_combine", "RecA"]:
        for corpus in ["full", "strict_489"]:
            lines.append(
                f"  {feature:12s} {corpus:11s} id50={get(feature, corpus, 0.5):.3f}"
                f" | id30={get(feature, corpus, 0.3):.3f}"
            )
    if skipped:
        lines += ["", "BLOCS SANS SPLIT POSSIBLE (composantes insuffisantes) :"]
        lines += [f"  - {s}" for s in skipped]
    lines += [
        "",
        "LECTURE : comparer chaque protéine à son propre split par cluster,",
        "et Nif_combine à RecA. Si les AUC restent hautes alors qu'aucun",
        "cluster de séquences n'est partagé entre train et test, le signal",
        "ne repose pas sur des séquences quasi identiques ; si elles",
        "s'effondrent, la reconnaissance de séquences proches portait le score.",
        "La part de composantes mono-classe et la plus grande composante sont",
        "dans phase12_mmseqs_protein_summary.csv (garde-fou de dureté).",
        "",
        f"Durée totale : {time.time() - t0:.0f}s",
        f"Par répétition : {OUT_METRICS / 'phase12_mmseqs_protein_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase12_mmseqs_protein_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE12_MMSEQS_PROTEIN_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
