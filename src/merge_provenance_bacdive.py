# -*- coding: utf-8 -*-
"""
Fusionne les résultats BacDive (statut=documente uniquement) dans table_provenance.csv.
Crée un backup avant écriture.
Ne change JAMAIS le label Alida.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"
OUT_DIR = ROOT / "results" / "provenance"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lots",
        nargs="+",
        required=True,
        help="Numéros de lots à fusionner, ex: 0 1 2",
    )
    args = ap.parse_args()

    frames = []
    for lot in args.lots:
        p = OUT_DIR / f"bacdive_lot{lot}.csv"
        if not p.exists():
            raise SystemExit(f"Manquant: {p}")
        frames.append(pd.read_csv(p))
    lots = pd.concat(frames, ignore_index=True)

    # uniquement documente avec un id
    doc = lots[lots["statut_propose"].astype(str) == "documente"].copy()
    doc = doc[doc["bacdive_id_propose"].notna() & (doc["bacdive_id_propose"].astype(str).str.strip() != "")]
    # si doublons souche, garder la dernière
    doc = doc.drop_duplicates(subset=["souche"], keep="last")

    prov = pd.read_csv(PROV)
    # colonnes texte pour éviter conflits float64 / str
    for col in ("bacdive_id", "ncbi_taxid", "preuve_label", "statut_provenance"):
        if col in prov.columns:
            prov[col] = prov[col].astype("object")

    # corriger typo documente4
    prov["statut_provenance"] = prov["statut_provenance"].astype(str).replace({"documente4": "documente"})

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PROV.with_name(f"table_provenance_backup_{stamp}.csv")
    # recharger original non touché pour backup propre
    pd.read_csv(PROV).to_csv(backup, index=False)

    updated = 0
    for _, row in doc.iterrows():
        souche = row["souche"]
        idx = prov.index[prov["souche"] == souche]
        if len(idx) == 0:
            continue
        i = idx[0]
        prov.at[i, "bacdive_id"] = str(int(float(row["bacdive_id_propose"])))
        tax = row.get("ncbi_taxid_propose")
        if pd.notna(tax) and str(tax).strip() not in ("", "nan"):
            tax_s = str(tax).strip()
            if tax_s.replace(".", "", 1).isdigit():
                prov.at[i, "ncbi_taxid"] = str(int(float(tax_s)))
            else:
                import re

                m = re.search(r"\b(\d{3,})\b", tax_s)
                if m:
                    prov.at[i, "ncbi_taxid"] = m.group(1)
        preuve = row.get("preuve_proposee")
        if pd.notna(preuve) and str(preuve).strip():
            prov.at[i, "preuve_label"] = str(preuve)
        prov.at[i, "statut_provenance"] = "documente"
        updated += 1

    prov.to_csv(PROV, index=False)

    print("Backup ->", backup)
    print("Fusion documente:", updated, "souches")
    print("Statuts après:")
    print(prov["statut_provenance"].value_counts().to_string())
    print("OK ->", PROV)


if __name__ == "__main__":
    main()
