# -*- coding: utf-8 -*-
"""
Phase 1 — Téléchargement UniProt des contrôles négatifs (RecA, GyrB, RpoB).
Une séquence par souche et par gène (meilleure entrée disponible).
N'écrit PAS dans table_provenance.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"
OUT_DIR = ROOT / "data" / "raw" / "controls_phase1"
META_CSV = OUT_DIR / "controls_fetch_table.csv"
FASTA_DIR = OUT_DIR / "fasta"
CACHE = OUT_DIR / "uniprot_cache"
LOG = OUT_DIR / "fetch_log.txt"

GENES = {
    "RecA": ["gene:recA", "protein_name:RecA"],
    "GyrB": ["gene:gyrB", "protein_name:DNA gyrase subunit B"],
    "RpoB": ["gene:rpoB", "protein_name:DNA-directed RNA polymerase subunit beta"],
}

SESSION = requests.Session()
SESSION.headers.update({"Accept": "text/plain", "User-Agent": "OXYFIX-Phase1/1.0"})
SLEEP = 0.25


def souche_to_species(souche: str) -> str:
    s = str(souche)
    s = re.split(r"_\(", s, maxsplit=1)[0]
    parts = s.split("_")
    if len(parts) < 2:
        return parts[0].replace("_", " ")
    genus, epithet = parts[0], parts[1]
    if re.fullmatch(r"(JCM|DSM|ATCC|NBRC|LMG|CIP|KCTC).*", epithet, flags=re.I):
        return genus
    return f"{genus} {epithet}"


def pick_fasta(text: str) -> tuple[str, str] | None:
    """Retourne (header, sequence) de la meilleure entrée (évite fragments courts)."""
    if not text or not text.strip().startswith(">"):
        return None
    entries = []
    cur_h, cur_s = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur_h is not None:
                seq = "".join(cur_s)
                entries.append((cur_h, seq))
            cur_h, cur_s = line[1:].strip(), []
        else:
            cur_s.append(line.strip())
    if cur_h is not None:
        entries.append((cur_h, "".join(cur_s)))
    if not entries:
        return None
    # préférer non-fragment et plus long
    def score(item):
        h, seq = item
        frag = 1 if "Fragment" in h else 0
        return (frag, -len(seq))

    entries.sort(key=score)
    h, seq = entries[0]
    if len(seq) < 50:
        return None
    return h, seq


def uniprot_search(query: str, size: int = 5) -> str:
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        f"query={quote(query)}&format=fasta&size={size}"
    )
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200:
                return r.text
            time.sleep(SLEEP * (attempt + 2))
        except Exception:
            time.sleep(SLEEP * (attempt + 2))
    return ""


def fetch_one(species: str, gene_key: str) -> dict:
    cache_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{species}_{gene_key}")[:150]
    cache_path = CACHE / f"{cache_key}.fasta"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        text = cache_path.read_text(encoding="utf-8", errors="ignore")
        picked = pick_fasta(text)
        if picked:
            h, seq = picked
            return {
                "status": "ok_cache",
                "header": h,
                "sequence": seq,
                "length": len(seq),
                "query": "cache",
            }

    queries = []
    for qpart in GENES[gene_key]:
        queries.append(f"({qpart}) AND (organism_name:{species})")
        # fallback genus only
    genus = species.split()[0]
    for qpart in GENES[gene_key]:
        queries.append(f"({qpart}) AND (taxonomy_name:{genus})")

    last_q = ""
    for q in queries:
        last_q = q
        text = uniprot_search(q, size=5)
        time.sleep(SLEEP)
        picked = pick_fasta(text)
        if picked:
            cache_path.write_text(text, encoding="utf-8")
            h, seq = picked
            level = "species" if f"organism_name:{species}" in q else "genus"
            return {
                "status": "ok",
                "header": h,
                "sequence": seq,
                "length": len(seq),
                "query": q,
                "match_level": level,
            }
    return {
        "status": "missing",
        "header": "",
        "sequence": "",
        "length": 0,
        "query": last_q,
        "match_level": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = toutes les souches")
    ap.add_argument("--genes", nargs="+", default=["RecA", "GyrB", "RpoB"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for g in args.genes:
        (FASTA_DIR / g).mkdir(parents=True, exist_ok=True)

    prov = pd.read_csv(PROV)
    if args.limit and args.limit > 0:
        prov = prov.head(args.limit)

    rows = []
    n = len(prov)
    for i, row in enumerate(prov.itertuples(index=False), 1):
        souche = row.souche
        label = int(row.label)
        label_texte = row.label_texte
        species = souche_to_species(souche)
        print(f"[{i}/{n}] {souche} -> {species}")
        for gene in args.genes:
            res = fetch_one(species, gene)
            fasta_path = ""
            if res["status"].startswith("ok") and res["sequence"]:
                fasta_path = str(FASTA_DIR / gene / f"{souche}.fasta")
                Path(fasta_path).write_text(
                    f">{souche}|{gene}|{res.get('match_level','')}|{res['header']}\n"
                    + "\n".join(res["sequence"][j : j + 60] for j in range(0, len(res["sequence"]), 60))
                    + "\n",
                    encoding="utf-8",
                )
            rows.append(
                {
                    "souche": souche,
                    "label": label,
                    "label_texte": label_texte,
                    "species_query": species,
                    "gene": gene,
                    "status": res["status"],
                    "match_level": res.get("match_level", ""),
                    "length": res["length"],
                    "header": res.get("header", "")[:200],
                    "fasta_path": fasta_path,
                    "query": res.get("query", ""),
                }
            )
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(META_CSV, index=False)
            print("  checkpoint", META_CSV)

    out = pd.DataFrame(rows)
    out.to_csv(META_CSV, index=False)
    summary = out.groupby(["gene", "status"]).size().unstack(fill_value=0)
    print(summary)
    LOG.write_text(summary.to_string() + "\n", encoding="utf-8")
    print("OK ->", META_CSV)


if __name__ == "__main__":
    main()
