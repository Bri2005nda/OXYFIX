# -*- coding: utf-8 -*-
"""
Lot 0 — test BacDive (10 souches).
Écrit UNIQUEMENT dans results/provenance/ (pas dans table_provenance.csv).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"
OUT_DIR = ROOT / "results" / "provenance"
CACHE = OUT_DIR / "bacdive_cache"
OUT_CSV = OUT_DIR / "bacdive_lot0.csv"
OUT_SUMMARY = OUT_DIR / "bacdive_lot0_summary.txt"

API = "https://api.bacdive.dsmz.de/v2"
SLEEP = 0.35
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "OXYFIX-URIFIA-Lot0/1.0"})


def souche_to_species(souche: str) -> tuple[str, str | None]:
    """Acidihalobacter_aeolianus -> (Acidihalobacter, aeolianus). Ignore suffixes type JCM_21531."""
    parts = str(souche).split("_")
    if len(parts) < 2:
        return parts[0], None
    genus = parts[0]
    epithet = parts[1]
    # si epithet ressemble à un code collection, pas une espèce
    if re.fullmatch(r"(JCM|DSM|ATCC|NBRC|LMG|CIP|KCTC)\d*", epithet, flags=re.I):
        return genus, None
    return genus, epithet


def classify_oxygen(oxy: str | None, label_texte: str) -> str:
    if not oxy:
        return "partiel"
    o = oxy.lower().strip()
    lab = label_texte.lower().strip()

    aero_ok = ("aerobe" in o or "aerobic" in o) and "anaerobe" not in o.replace("facultative anaerobe", "")
    # BacDive: aerobe, facultative anaerobe, microaerophile, anaerobe, obligate anaerobe...
    if "microaerophil" in o or "facultative" in o:
        return "a_verifier"

    if lab.startswith("aerobie"):
        if o in ("aerobe", "obligate aerobe") or (o.endswith("aerobe") and "anaerobe" not in o):
            return "documente"
        if "anaerobe" in o and "facultative" not in o:
            return "a_verifier"  # conflit
        return "a_verifier"

    if lab.startswith("anaerobie"):
        if "anaerobe" in o and "facultative" not in o:
            return "documente"
        if o in ("aerobe", "obligate aerobe"):
            return "a_verifier"
        return "a_verifier"

    return "a_verifier"


def extract_oxygen(record: dict) -> str | None:
    phys = record.get("Physiology and metabolism") or {}
    ot = phys.get("oxygen tolerance")
    if isinstance(ot, dict):
        return ot.get("oxygen tolerance") or ot.get("oxygen_tol")
    if isinstance(ot, list) and ot:
        first = ot[0]
        if isinstance(first, dict):
            return first.get("oxygen tolerance") or first.get("oxygen_tol")
        return str(first)
    if isinstance(ot, str):
        return ot
    # fallback keywords
    gen = record.get("General") or {}
    kws = gen.get("keywords") or []
    for k in kws:
        kl = str(k).lower()
        if kl in ("aerobe", "anaerobe", "microaerophile", "facultative anaerobe", "obligate anaerobe"):
            return kl
    return None


def extract_taxid(record: dict) -> str:
    gen = record.get("General") or {}
    tax = gen.get("NCBI tax id")
    if isinstance(tax, dict):
        return str(tax.get("NCBI tax id") or "")
    if tax is not None:
        return str(tax)
    return ""


def api_get(url: str) -> dict | None:
    try:
        r = SESSION.get(url, timeout=45)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def taxon_ids(genus: str, epithet: str | None) -> list[int]:
    if epithet:
        url = f"{API}/taxon/{genus}/{epithet}"
    else:
        url = f"{API}/taxon/{genus}"
    data = api_get(url)
    time.sleep(SLEEP)
    if not data or "_error" in data:
        return []
    results = data.get("results") or []
    # results may be list of ints or list of dicts
    ids = []
    for x in results:
        if isinstance(x, int):
            ids.append(x)
        elif isinstance(x, dict) and "BacDive-ID" in x:
            ids.append(int(x["BacDive-ID"]))
        elif isinstance(x, dict) and "id" in x:
            ids.append(int(x["id"]))
    # pagination first page only for Lot0 (cap ambiguity)
    return ids


def fetch_ids(ids: list[int]) -> dict:
    if not ids:
        return {}
    # limit 100
    joined = ";".join(str(i) for i in ids[:20])  # Lot0: max 20
    url = f"{API}/fetch/{joined}"
    data = api_get(url)
    time.sleep(SLEEP)
    if not data or "_error" in data:
        return {"_error": data}
    return data.get("results") or {}


def pick_lot0(df: pd.DataFrame) -> pd.DataFrame:
    documented = df[df["statut_provenance"].astype(str).str.startswith("documente")]
    partial = df[df["statut_provenance"].astype(str) == "partiel"].copy()
    partial["genus"] = partial["souche"].astype(str).str.split("_").str[0]
    # 8 nouvelles : prioriser gros genres, 1–2 par genre
    priority = [
        "Clostridium",
        "Paenibacillus",
        "Paraburkholderia",
        "Sinorhizobium",
        "Rhizobium",
        "Bradyrhizobium",
        "Mesorhizobium",
        "Klebsiella",
        "Azospirillum",
    ]
    chosen = []
    for g in priority:
        sub = partial[partial["genus"] == g]
        if len(sub) == 0:
            continue
        chosen.append(sub.iloc[0])
        if len(chosen) >= 8:
            break
    # si moins de 8, compléter
    if len(chosen) < 8:
        rest = partial[~partial["souche"].isin([r["souche"] for r in chosen])]
        for _, row in rest.head(8 - len(chosen)).iterrows():
            chosen.append(row)
    new_df = pd.DataFrame(chosen)
    out = pd.concat([documented, new_df], ignore_index=True)
    return out.head(10)


def process_row(row: pd.Series) -> dict:
    souche = row["souche"]
    label_texte = str(row["label_texte"])
    genus, epithet = souche_to_species(souche)
    species_query = f"{genus} {epithet}" if epithet else genus

    # si déjà un bacdive_id, fetch direct
    existing_id = row.get("bacdive_id")
    ids: list[int] = []
    if pd.notna(existing_id) and str(existing_id).strip() not in ("", "nan"):
        try:
            ids = [int(float(existing_id))]
        except Exception:
            ids = []

    if not ids:
        ids = taxon_ids(genus, epithet)

    cache_path = CACHE / f"{souche}.json"
    result = {
        "souche": souche,
        "label_texte": label_texte,
        "species_query": species_query,
        "n_bacdive_hits": len(ids),
        "bacdive_id_propose": "",
        "ncbi_taxid_propose": "",
        "oxygen_tol": "",
        "statut_propose": "partiel",
        "preuve_proposee": "",
        "commentaire": "",
        "label_alida_inchange": True,
    }

    if not ids:
        result["commentaire"] = "Aucun hit taxon BacDive"
        result["statut_propose"] = "partiel"
        return result

    if len(ids) > 1:
        # trop ambigu pour auto-documenter : on fetch le 1er pour info, statut a_verifier
        records = fetch_ids(ids[:5])
        cache_path.write_text(json.dumps({"ids": ids, "sample": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        # essayer de trouver une tolérance unique
        oxys = []
        for bid, rec in (records.items() if isinstance(records, dict) else []):
            if bid.startswith("_"):
                continue
            oxy = extract_oxygen(rec) if isinstance(rec, dict) else None
            oxys.append((bid, oxy, extract_taxid(rec) if isinstance(rec, dict) else ""))
        unique_oxy = {o for _, o, _ in oxys if o}
        result["n_bacdive_hits"] = len(ids)
        result["commentaire"] = f"Ambigu: {len(ids)} souches BacDive"
        if len(unique_oxy) == 1:
            bid, oxy, tax = oxys[0]
            # still a_verifier because multiple strains
            result["bacdive_id_propose"] = str(bid)
            result["oxygen_tol"] = oxy or ""
            result["ncbi_taxid_propose"] = tax
            result["statut_propose"] = "a_verifier"
            result["preuve_proposee"] = f"BacDive oxygen tolerance={oxy} (ID {bid}; {len(ids)} hits espèce)"
        else:
            result["statut_propose"] = "a_verifier"
            result["preuve_proposee"] = f"BacDive multi-hits ({len(ids)}) oxygen_tol diverge ou absente"
        return result

    # exactement 1 id
    bid = ids[0]
    records = fetch_ids([bid])
    cache_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    rec = records.get(str(bid)) or records.get(bid) or {}
    if isinstance(records, dict) and "_error" in records:
        result["commentaire"] = f"Erreur fetch: {records['_error']}"
        result["statut_propose"] = "partiel"
        return result
    if not isinstance(rec, dict) or not rec:
        # parfois results est {id: {...}}
        if isinstance(records, dict) and len(records) == 1:
            rec = next(iter(records.values()))
        else:
            result["commentaire"] = "Fetch vide"
            result["statut_propose"] = "partiel"
            return result

    oxy = extract_oxygen(rec)
    tax = extract_taxid(rec)
    statut = classify_oxygen(oxy, label_texte)
    result["bacdive_id_propose"] = str(bid)
    result["ncbi_taxid_propose"] = tax
    result["oxygen_tol"] = oxy or ""
    result["statut_propose"] = statut
    if oxy:
        result["preuve_proposee"] = f"BacDive oxygen tolerance={oxy} (ID {bid})"
    else:
        result["preuve_proposee"] = f"BacDive ID {bid} trouvé, oxygen tolerance absente"
        result["statut_propose"] = "partiel"
        result["commentaire"] = "Pas de champ oxygen tolerance"
    if statut == "a_verifier" and oxy:
        result["commentaire"] = f"À vérifier: label Alida={label_texte} vs BacDive={oxy}"
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROV)
    lot = pick_lot0(df)
    print("Lot 0 souches:")
    for s in lot["souche"]:
        print(" -", s)

    rows = []
    for _, row in lot.iterrows():
        print("...", row["souche"])
        rows.append(process_row(row))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # résumé
    lines = []
    lines.append(f"Lot 0 — {len(out)} souches")
    lines.append(f"Fichier: {OUT_CSV}")
    lines.append(f"table_provenance.csv NON modifié")
    lines.append("")
    lines.append(out["statut_propose"].value_counts().to_string())
    lines.append("")
    lines.append(out.to_string(index=False))
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:20]))
    print("OK ->", OUT_CSV)


if __name__ == "__main__":
    main()
