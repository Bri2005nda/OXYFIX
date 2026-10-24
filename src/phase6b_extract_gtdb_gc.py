# -*- coding: utf-8 -*-
"""
Phase 6b (extraction) — GC % génomique GTDB des souches appariées.

Lit les métadonnées GTDB (bac120 + ar53) en flux et ne conserve que les
accessions représentatives utilisées par la baseline phylogénétique
(results/metrics/phase6_gtdb_mapping.csv). Le GC retenu est donc celui
du génome représentatif GTDB de l'espèce (ou, en repli, du genre) de la
souche — pas une mesure directe du génome de la souche elle-même.

Sortie : results/metrics/phase6b_gtdb_gc_by_strain.csv
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GTDB_DIR = ROOT / "data" / "external" / "gtdb"
MAPPING = ROOT / "results" / "metrics" / "phase6_gtdb_mapping.csv"
OUT = ROOT / "results" / "metrics" / "phase6b_gtdb_gc_by_strain.csv"


def main():
    mapping = list(csv.DictReader(MAPPING.open(encoding="utf-8")))
    wanted = {
        r["gtdb_accession"]: r
        for r in mapping
        if r.get("gtdb_accession")
    }
    print(f"accessions recherchées : {len(wanted)}")

    found: dict[str, dict] = {}
    for fname in ["bac120_metadata.tsv.gz", "ar53_metadata.tsv.gz"]:
        path = GTDB_DIR / fname
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader)
            i_acc = header.index("accession")
            i_gc = header.index("gc_percentage")
            i_taxid = header.index("ncbi_taxid")
            i_species_taxid = header.index("ncbi_species_taxid")
            for row in reader:
                acc = row[i_acc]
                if acc in wanted and acc not in found:
                    found[acc] = {
                        "gtdb_accession": acc,
                        "gc_percentage": row[i_gc],
                        "gtdb_ncbi_taxid": row[i_taxid],
                        "gtdb_ncbi_species_taxid": row[i_species_taxid],
                    }
        print(f"{fname}: {len(found)} trouvées")

    rows = []
    for r in mapping:
        acc = r.get("gtdb_accession") or ""
        info = found.get(acc, {})
        rows.append(
            {
                "souche": r["souche"],
                "label": r["label"],
                "corpus_strict": r["corpus_strict"],
                "match_level": r["match_level"],
                "gtdb_accession": acc,
                "gc_percentage": info.get("gc_percentage", ""),
                "gtdb_ncbi_taxid": info.get("gtdb_ncbi_taxid", ""),
                "gtdb_ncbi_species_taxid": info.get("gtdb_ncbi_species_taxid", ""),
            }
        )
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_gc = sum(1 for r in rows if r["gc_percentage"])
    print(f"souches avec GC : {n_gc}/{len(rows)}")
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
