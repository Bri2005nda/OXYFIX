# -*- coding: utf-8 -*-
"""
Phase 9 (préparation) — Famille et ordre de chaque souche.

Source : taxonomie GTDB des génomes représentatifs appariés en Phase 6
(results/metrics/phase6_gtdb_mapping.csv + fichiers bac120/ar53).
Pour les souches non appariées, famille/ordre sont hérités du consensus
du même genre de projet parmi les souches appariées (source marquée).
Sortie : data/raw/metadata/table_taxonomy_levels.csv
"""
from __future__ import annotations

import csv
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GTDB = ROOT / "data" / "external" / "gtdb"
MAPPING = ROOT / "results" / "metrics" / "phase6_gtdb_mapping.csv"
OUT = ROOT / "data" / "raw" / "metadata" / "table_taxonomy_levels.csv"


def parse_ranks(tax: str) -> dict[str, str]:
    out = {}
    for part in tax.split(";"):
        part = part.strip()
        if "__" in part:
            k, v = part.split("__", 1)
            out[k] = v
    return out


def main():
    mapping = list(csv.DictReader(MAPPING.open(encoding="utf-8")))
    wanted = {r["gtdb_accession"] for r in mapping if r.get("gtdb_accession")}
    acc_ranks: dict[str, dict[str, str]] = {}
    for fname in ["bac120_taxonomy.tsv.gz", "ar53_taxonomy.tsv.gz"]:
        with gzip.open(GTDB / fname, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                acc, tax = line.rstrip("\n").split("\t", 1)
                if acc in wanted and acc not in acc_ranks:
                    acc_ranks[acc] = parse_ranks(tax)
    print(f"rangs GTDB récupérés : {len(acc_ranks)}/{len(wanted)}")

    rows = []
    genus_family: dict[str, Counter] = defaultdict(Counter)
    genus_order: dict[str, Counter] = defaultdict(Counter)
    for r in mapping:
        acc = r.get("gtdb_accession") or ""
        ranks = acc_ranks.get(acc, {})
        genus = str(r["souche"]).split("_")[0]
        fam, order = ranks.get("f", ""), ranks.get("o", "")
        source = "gtdb_representant" if acc else ""
        rows.append({
            "souche": r["souche"], "label": r["label"],
            "corpus_strict": r["corpus_strict"], "genus_projet": genus,
            "gtdb_genus": ranks.get("g", ""), "family": fam, "order": order,
            "taxonomy_source": source,
        })
        if fam:
            genus_family[genus][fam] += 1
        if order:
            genus_order[genus][order] += 1

    n_fill = 0
    for row in rows:
        if not row["family"]:
            genus = row["genus_projet"]
            fam = genus_family[genus].most_common(1)
            order = genus_order[genus].most_common(1)
            if fam:
                row["family"] = fam[0][0]
                row["order"] = order[0][0] if order else ""
                row["taxonomy_source"] = "consensus_genre_projet"
                n_fill += 1
    print(f"souches complétées par consensus de genre : {n_fill}")

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_fam = sum(1 for r in rows if r["family"])
    n_ord = sum(1 for r in rows if r["order"])
    print(f"famille renseignée : {n_fam}/{len(rows)} | ordre : {n_ord}/{len(rows)}")
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
