# -*- coding: utf-8 -*-
"""
Phase 6 — Baseline plus proche voisin phylogénétique GTDB.

Question (Annexe B du manuscrit) : le score des modèles de séquences
s'explique-t-il simplement par la parenté ? Pour chaque souche de test,
on prédit son étiquette oxygène à partir de ses plus proches parents
PRÉSENTS DANS LE TRAIN, la distance étant la distance phylogénétique
(longueurs de branches) dans l'arbre GTDB (bac120 / ar53, release latest
téléchargée le 2026-10-21).

Appariement souche -> GTDB :
- niveau "species" : la souche est rattachée à un génome GTDB de la même
  espèce (accession représentative déterministe : RS_ d'abord, puis ordre
  lexicographique) ;
- niveau "genus" (repli) : espèce absente de GTDB mais genre présent ->
  génome représentatif du genre GTDB ;
- sinon : non appariée (exclue de cette baseline, couverture rapportée).

Protocoles identiques à la Phase 5 : corpus Full / Strict, splits
aléatoire stratifié et par genre, 20 graines (42..61). Aucun
apprentissage, aucun SMOTE : les étiquettes du train sont seulement
recopiées / moyennées depuis les voisins.

Sorties :
- results/metrics/phase6_gtdb_mapping.csv
- results/metrics/phase6_gtdb_nn_per_repeat.csv
- results/metrics/phase6_gtdb_nn_summary.csv
- results/figures/phase6_gtdb_nn_vs_hgb.png
- results/metrics/PHASE6_GTDB_NN_DECISION.txt
"""
from __future__ import annotations

import gzip
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
GTDB_DIR = ROOT / "data" / "external" / "gtdb"
OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
P3 = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
PHASE5_SUMMARY = OUT_METRICS / "phase5_repeated_splits_summary.csv"

SEEDS = list(range(42, 62))
TREES = {
    "Bacteria": ("bac120_taxonomy.tsv.gz", "bac120.tree.gz"),
    "Archaea": ("ar53_taxonomy.tsv.gz", "ar53.tree.gz"),
}
EPS = 1e-9


# ---------------------------------------------------------------- GTDB data
def load_gtdb_taxonomy() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    """species -> accessions, genus -> accessions, accession -> domaine."""
    species_to_acc: dict[str, list[str]] = {}
    genus_to_acc: dict[str, list[str]] = {}
    acc_domain: dict[str, str] = {}
    for domain, (tax_file, _) in TREES.items():
        path = GTDB_DIR / tax_file
        if not path.exists():
            raise FileNotFoundError(f"GTDB manquant : {path}")
        with gzip.open(path, "rt") as fh:
            for line in fh:
                acc, tax = line.rstrip("\n").split("\t", 1)
                acc_domain[acc] = domain
                mg = re.search(r"g__([^;]+)", tax)
                ms = re.search(r"s__([^;]+)$", tax)
                if ms:
                    species_to_acc.setdefault(ms.group(1).strip(), []).append(acc)
                if mg:
                    genus_to_acc.setdefault(mg.group(1).strip(), []).append(acc)
    return species_to_acc, genus_to_acc, acc_domain


def representative(accessions: list[str]) -> str:
    return sorted(accessions, key=lambda a: (0 if a.startswith("RS_") else 1, a))[0]


def species_guess(souche: str) -> str:
    toks = [re.sub(r"[^A-Za-z0-9.-]", "", t) for t in str(souche).replace("(", " ").split("_")]
    toks = [t for t in toks if t]
    return " ".join(toks[:2]) if len(toks) >= 2 else str(souche)


# ------------------------------------------------------------- Newick parse
TOKEN_RE = re.compile(r"\(|\)|,|:|;|'[^']*'|[^(),:;\s]+")


class Tree:
    def __init__(self, newick: str):
        parent: list[int] = []
        branch: list[float] = []
        label: list[str] = []
        tokens = TOKEN_RE.findall(newick)
        pos = 0
        sys.setrecursionlimit(100000)

        def new_node(par: int) -> int:
            parent.append(par)
            branch.append(0.0)
            label.append("")
            return len(parent) - 1

        def parse_node(par: int) -> int:
            nonlocal pos
            node = new_node(par)
            if pos < len(tokens) and tokens[pos] == "(":
                pos += 1  # '('
                while True:
                    parse_node(node)
                    if pos < len(tokens) and tokens[pos] == ",":
                        pos += 1
                        continue
                    if pos < len(tokens) and tokens[pos] == ")":
                        pos += 1
                    break
                if pos < len(tokens) and tokens[pos] not in ("(", ")", ",", ":", ";"):
                    label[node] = tokens[pos].strip("'")
                    pos += 1
            else:
                if pos < len(tokens):
                    label[node] = tokens[pos].strip("'")
                    pos += 1
            if pos < len(tokens) and tokens[pos] == ":":
                pos += 1
                if pos < len(tokens):
                    try:
                        branch[node] = float(tokens[pos])
                    except ValueError:
                        branch[node] = 0.0
                    pos += 1
            return node

        parse_node(-1)
        self.parent = np.array(parent, dtype=np.int32)
        self.branch = np.array(branch, dtype=np.float64)
        self.label = label
        self.n = len(parent)
        # distance au root
        depth = np.zeros(self.n, dtype=np.float64)
        order = np.argsort(np.where(self.parent >= 0, self.parent, 0))
        # propagation simple : les parents sont toujours créés avant les enfants
        for node in range(1, self.n):
            p = int(self.parent[node])
            depth[node] = depth[p] + self.branch[node]
        self.depth = depth
        self.leaf_node = {lab: i for i, lab in enumerate(label) if lab}
        # ancêtres binaires
        log = max(1, int(np.ceil(np.log2(max(self.n, 2)))))
        up = np.zeros((log, self.n), dtype=np.int32)
        up[0] = np.where(self.parent >= 0, self.parent, 0)
        for k in range(1, log):
            up[k] = up[k - 1][up[k - 1]]
        self.up = up
        self.log = log
        # profondeur en nombre d'arêtes
        edge_depth = np.zeros(self.n, dtype=np.int32)
        for node in range(1, self.n):
            edge_depth[node] = edge_depth[int(self.parent[node])] + 1
        self.edge_depth = edge_depth

    def lca_many(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.copy()
        b = b.copy()
        da = self.edge_depth[a]
        db = self.edge_depth[b]
        diff = (da - db).astype(np.int64)
        # remonter a
        bit = 0
        d = np.abs(diff)
        aa = np.where(diff > 0, a, b)
        while (d > 0).any():
            step = (d & 1).astype(bool)
            aa[step] = self.up[bit][aa[step]]
            d >>= 1
            bit += 1
        a2 = np.where(diff > 0, aa, a)
        b2 = np.where(diff > 0, b, aa)
        same = a2 == b2
        for k in range(self.log - 1, -1, -1):
            ua = self.up[k][a2]
            ub = self.up[k][b2]
            diff_nodes = (~same) & (ua != ub)
            a2[diff_nodes] = ua[diff_nodes]
            b2[diff_nodes] = ub[diff_nodes]
            same = a2 == b2
        lca = np.where(same, a2, self.parent[np.maximum(a2, 0)])
        lca = np.where(same, a2, lca)
        # si non égaux après la boucle, le LCA est le parent commun
        not_same = a2 != b2
        lca[not_same] = self.up[0][a2[not_same]]
        return lca

    def distances(self, nodes_a: np.ndarray, nodes_b: np.ndarray) -> np.ndarray:
        """Distances cophenetiques entre deux ensembles de noeuds (même arbre)."""
        ia, ib = np.meshgrid(nodes_a, nodes_b, indexing="ij")
        ia = ia.ravel()
        ib = ib.ravel()
        lca = self.lca_many(ia, ib)
        d = self.depth[ia] + self.depth[ib] - 2.0 * self.depth[lca]
        return d.reshape(len(nodes_a), len(nodes_b))


def load_trees() -> dict[str, Tree]:
    trees = {}
    for domain, (_, tree_file) in TREES.items():
        path = GTDB_DIR / tree_file
        if not path.exists():
            raise FileNotFoundError(f"GTDB manquant : {path}")
        with gzip.open(path, "rt") as fh:
            trees[domain] = Tree(fh.read())
        print(f"arbre {domain}: {trees[domain].n} noeuds")
    return trees


# ------------------------------------------------------------------ metrics
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
        "n_test": int(len(yte)),
        "n_test_aerobie": int((yte == 1).sum()),
        "n_test_anaerobie": int((yte == 0).sum()),
    }


def knn_scores(dist_row: np.ndarray, y_train: np.ndarray, k: int) -> float:
    """Score aérobie depuis les voisins (distance phylogénétique croissante)."""
    order = np.argsort(dist_row, kind="stable")
    d_sorted = dist_row[order]
    y_sorted = y_train[order]
    finite = np.isfinite(d_sorted)
    if not finite.any():
        return np.nan
    d_sorted, y_sorted = d_sorted[finite], y_sorted[finite]
    if d_sorted[0] <= 1e-12:
        # même noeud représentatif : moyenne de tout le groupe à distance nulle
        return float(y_sorted[d_sorted <= 1e-12].mean())
    kk = min(k, len(d_sorted))
    w = 1.0 / (d_sorted[:kk] + EPS)
    return float(np.average(y_sorted[:kk], weights=w))


def main():
    t0 = time.time()
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    prov = pd.read_csv(P3)
    prov["genus"] = prov["souche"].astype(str).str.split("_").str[0]
    prov["species_guess"] = prov["souche"].map(species_guess)

    print("=== Chargement GTDB ===")
    species_to_acc, genus_to_acc, acc_domain = load_gtdb_taxonomy()
    trees = load_trees()

    # ---------------- appariement
    rows = []
    for _, r in prov.iterrows():
        sp = r["species_guess"]
        genus = str(r["souche"]).split("_")[0]
        genus_clean = re.sub(r"[^A-Za-z0-9.-]", "", genus)
        acc, level = None, "unmapped"
        sp_candidates = [
            a
            for a in species_to_acc.get(sp, [])
            if a in trees[acc_domain[a]].leaf_node
        ]
        genus_candidates = [
            a
            for a in genus_to_acc.get(genus_clean, [])
            if a in trees[acc_domain[a]].leaf_node
        ]
        if sp_candidates:
            acc, level = representative(sp_candidates), "species"
        elif genus_candidates:
            acc, level = representative(genus_candidates), "genus"
        rows.append(
            {
                "souche": r["souche"],
                "label": int(r["label"]),
                "genus_projet": genus,
                "species_guess": sp,
                "corpus_strict": r["corpus_strict"],
                "gtdb_accession": acc,
                "gtdb_domain": acc_domain.get(acc) if acc else None,
                "match_level": level,
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(OUT_METRICS / "phase6_gtdb_mapping.csv", index=False)
    cov = mapping["match_level"].value_counts().to_dict()
    print("appariement:", cov)

    # noeud d'arbre par souche
    node_of: dict[str, tuple[str, int]] = {}
    for _, r in mapping.iterrows():
        if pd.notna(r["gtdb_accession"]):
            tree = trees[r["gtdb_domain"]]
            node_of[r["souche"]] = (r["gtdb_domain"], int(tree.leaf_node[r["gtdb_accession"]]))

    souches_all = prov["souche"].astype(str).tolist()
    y_all = prov["label"].astype(int).values
    genus_all = prov["genus"].astype(str).values
    strict_mask = (prov["corpus_strict"].astype(str) == "oui").values

    # ---------------- distances entre souches appariées (par domaine)
    mapped_idx = [i for i, s in enumerate(souches_all) if s in node_of]
    dist = np.full((len(souches_all), len(souches_all)), np.nan, dtype=np.float64)
    for domain, tree in trees.items():
        idx = [i for i in mapped_idx if node_of[souches_all[i]][0] == domain]
        if not idx:
            continue
        nodes = np.array([node_of[souches_all[i]][1] for i in idx], dtype=np.int64)
        d = tree.distances(nodes, nodes)
        dist[np.ix_(idx, idx)] = d
        print(f"distances {domain}: {len(idx)} souches")

    # ---------------- évaluation répétée
    per_rows = []
    for corpus, cmask in [("full", np.ones(len(prov), dtype=bool)), ("strict_489", strict_mask)]:
        base_idx = np.array([i for i in range(len(prov)) if cmask[i]], dtype=int)
        mapped_base = np.array([i for i in base_idx if souches_all[i] in node_of], dtype=int)
        y_c = y_all[base_idx]
        for seed in SEEDS:
            # split aléatoire (sur tout le corpus, couverture mesurée ensuite)
            strat = y_c if min(np.bincount(y_c)) >= 2 else None
            tr_all, te_all = train_test_split(
                base_idx, test_size=0.2, random_state=seed, stratify=strat
            )
            # split genre
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr_g, te_g = next(gss.split(base_idx, y_c, genus_all[base_idx]))
            splits = {
                "random_80_20": (tr_all, te_all),
                "group_shuffle_genus": (base_idx[tr_g], base_idx[te_g]),
            }
            for protocol, (tr_idx, te_idx) in splits.items():
                tr = np.array([i for i in tr_idx if souches_all[i] in node_of], dtype=int)
                te = np.array([i for i in te_idx if souches_all[i] in node_of], dtype=int)
                if len(te) == 0 or len(np.unique(y_all[te])) < 2:
                    continue
                D = dist[np.ix_(te, tr)]
                ytr = y_all[tr]
                for method, k in [("knn1", 1), ("knn5_weighted", 5)]:
                    proba = np.array([knn_scores(D[j], ytr, k) for j in range(len(te))])
                    ok = np.isfinite(proba)
                    if ok.sum() < 5 or len(np.unique(y_all[te][ok])) < 2:
                        continue
                    m = metrics_of(y_all[te][ok], proba[ok])
                    per_rows.append(
                        {
                            "corpus": corpus,
                            "protocol": protocol,
                            "seed": seed,
                            "method": method,
                            "n_corpus_mapped": int(len(mapped_base)),
                            "n_test_total_split": int(len(te_idx)),
                            "n_test_mapped": int(len(te)),
                            "coverage_test": float(len(te) / max(len(te_idx), 1)),
                            **m,
                        }
                    )
        print(f"{corpus} terminé ({time.time() - t0:.0f}s)")

    per = pd.DataFrame(per_rows)
    per.to_csv(OUT_METRICS / "phase6_gtdb_nn_per_repeat.csv", index=False)
    summary = (
        per.groupby(["corpus", "protocol", "method"])
        .agg(
            n_repeats=("seed", "count"),
            coverage_test_mean=("coverage_test", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_sd=("mcc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            recall_aerobie_mean=("recall_aerobie", "mean"),
            recall_anaerobie_mean=("recall_anaerobie", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_METRICS / "phase6_gtdb_nn_summary.csv", index=False)

    # ---------------- figure : GTDB-NN (k=5) vs HGB Nif (Phase 5)
    hgb = pd.read_csv(PHASE5_SUMMARY)
    hgb = hgb[hgb["feature"] == "Nif_CKSAAP"][
        ["corpus", "protocol", "auc_mean", "auc_sd"]
    ].rename(columns={"auc_mean": "hgb_auc_mean", "auc_sd": "hgb_auc_sd"})
    hgb["protocol"] = hgb["protocol"].replace(
        {"random_80_20_SMOTE": "random_80_20", "group_shuffle_genus_SMOTE_train": "group_shuffle_genus"}
    )
    nn = summary[summary["method"] == "knn5_weighted"]
    merged = nn.merge(hgb, on=["corpus", "protocol"], how="left")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, corpus in zip(axes, ["full", "strict_489"]):
        sub = merged[merged["corpus"] == corpus].sort_values("protocol")
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["auc_mean"], width=0.4, yerr=sub["auc_sd"], capsize=3,
               color="#8B4513", label="GTDB plus proche voisin (k=5)")
        ax.bar(x + 0.2, sub["hgb_auc_mean"], width=0.4, yerr=sub["hgb_auc_sd"], capsize=3,
               color="#1f4e79", label="Nif CKSAAP + HGB (Phase 5)")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["aléatoire" if "random" in p else "genre" for p in sub["protocol"]])
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"Corpus {corpus}")
        ax.set_ylabel("AUC moyenne ± écart-type (20 splits)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 6 — Baseline phylogénétique GTDB contre modèle Nif")
    fig.tight_layout()
    fig_path = OUT_FIG / "phase6_gtdb_nn_vs_hgb.png"
    fig.savefig(fig_path, dpi=160)
    plt.close()

    # ---------------- décision
    def get(corpus, protocol, col="auc_mean", method="knn5_weighted"):
        s = summary[
            (summary["corpus"] == corpus)
            & (summary["protocol"] == protocol)
            & (summary["method"] == method)
        ]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    lines = [
        "PHASE 6 — BASELINE PLUS PROCHE VOISIN PHYLOGÉNÉTIQUE (GTDB)",
        "Arbres GTDB bac120/ar53 (release latest, téléchargés 2026-10-21).",
        f"Appariement : {cov} (species = même espèce GTDB ; genus = repli genre).",
        "",
        "AUC moyenne ± écart-type (20 splits, GTDB-NN k=5 pondéré distance) :",
    ]
    for corpus in ["full", "strict_489"]:
        for protocol in ["random_80_20", "group_shuffle_genus"]:
            s = summary[
                (summary["corpus"] == corpus)
                & (summary["protocol"] == protocol)
                & (summary["method"] == "knn5_weighted")
            ]
            if len(s):
                r = s.iloc[0]
                hg = merged[
                    (merged["corpus"] == corpus) & (merged["protocol"] == protocol)
                ]
                hg_txt = (
                    f" | Nif HGB Phase 5 = {float(hg['hgb_auc_mean'].iloc[0]):.3f}"
                    if len(hg) and pd.notna(hg["hgb_auc_mean"].iloc[0])
                    else ""
                )
                lines.append(
                    f"  {corpus:11s} {protocol:20s} AUC={r['auc_mean']:.3f} ± {r['auc_sd']:.3f}"
                    f" | Acc={r['accuracy_mean']:.3f} | MCC={r['mcc_mean']:.3f}"
                    f" | couverture test={r['coverage_test_mean']:.1%}{hg_txt}"
                )
    lines += [
        "",
        "LECTURE :",
        "- Si GTDB-NN approche Nif-HGB en aléatoire puis chute en genre, la parenté",
        "  suffit à produire l'essentiel du score historique.",
        "- GTDB-NN ne prédit que les souches appariées (couverture ci-dessus) ;",
        "  comparaison à HGB faite sur le corpus complet : écart de couverture à garder en tête.",
        "- Niveau species = même représentant GTDB pour toutes les souches d'une espèce",
        "  (distance nulle entre elles) ; niveau genus = représentant du genre.",
        "",
        f"Durée totale : {time.time() - t0:.0f}s",
        f"Mapping : {OUT_METRICS / 'phase6_gtdb_mapping.csv'}",
        f"Par répétition : {OUT_METRICS / 'phase6_gtdb_nn_per_repeat.csv'}",
        f"Synthèse : {OUT_METRICS / 'phase6_gtdb_nn_summary.csv'}",
        f"Figure : {fig_path}",
    ]
    (OUT_METRICS / "PHASE6_GTDB_NN_DECISION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
