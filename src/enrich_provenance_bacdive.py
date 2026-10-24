# -*- coding: utf-8 -*-
"""
Enrichissement BacDive par lot (genre).
Écrit UNIQUEMENT dans results/provenance/ — ne modifie PAS table_provenance.csv.

Usage:
  python src/enrich_provenance_bacdive.py --lot 1 --genus Clostridium
"""
from __future__ import annotations

import argparse
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

API = "https://api.bacdive.dsmz.de/v2"
SLEEP = 0.35
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "OXYFIX-URIFIA/1.1"})


COLL_RE = re.compile(
    r"\b(DSM|ATCC|JCM|NBRC|LMG|CIP|KCTC|NCIMB|NRRL|VKM|IAM)\s*[-_]?(\d+[A-Za-z0-9-]*)\b",
    re.I,
)


def extract_culture_codes(souche: str) -> list[str]:
    """Retourne des clés type 'DSM 12555' pour l'endpoint culturecollectionno."""
    text = str(souche).replace("_", " ")
    found = []
    for m in COLL_RE.finditer(text):
        found.append(f"{m.group(1).upper()} {m.group(2)}")
    # dédup en gardant l'ordre
    seen = set()
    out = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def souche_to_species(souche: str) -> tuple[str, str | None]:
    """
    Clostridium_aceticum -> (Clostridium, aceticum)
    Clostridium_acidisoli_DSM_12555 -> (Clostridium, acidisoli)
    Clostridium_acetobutylicum_(strain_ATCC_824_...) -> (Clostridium, acetobutylicum)
    """
    s = str(souche)
    # couper avant parenthèse de strain
    s = re.split(r"_\(", s, maxsplit=1)[0]
    parts = s.split("_")
    if not parts:
        return s, None
    genus = parts[0]
    if len(parts) < 2:
        return genus, None
    epithet = parts[1]
    # codes collection seuls
    if re.fullmatch(r"(JCM|DSM|ATCC|NBRC|LMG|CIP|KCTC)[-_]?\d*", epithet, flags=re.I):
        return genus, None
    if re.fullmatch(r"\d+", epithet):
        return genus, None
    return genus, epithet


def classify_oxygen(oxy: str | None, label_texte: str) -> str:
    if not oxy:
        return "partiel"
    o = oxy.lower().strip()
    lab = label_texte.lower().strip()

    if "microaerophil" in o or "facultative" in o:
        return "a_verifier"

    if lab.startswith("aerobie"):
        if o in ("aerobe", "obligate aerobe") or (
            "aerobe" in o and "anaerobe" not in o
        ):
            return "documente"
        if "anaerobe" in o:
            return "a_verifier"
        return "a_verifier"

    if lab.startswith("anaerobie"):
        if "anaerobe" in o and "facultative" not in o:
            return "documente"
        if o in ("aerobe", "obligate aerobe") or (
            "aerobe" in o and "anaerobe" not in o
        ):
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
    gen = record.get("General") or {}
    for k in gen.get("keywords") or []:
        kl = str(k).lower()
        if kl in (
            "aerobe",
            "anaerobe",
            "microaerophile",
            "facultative anaerobe",
            "facultative aerobe",
            "obligate anaerobe",
            "obligate aerobe",
        ):
            return kl
    return None


def extract_taxid(record: dict) -> str:
    """Préfère matching level == species, sinon premier id numérique."""
    gen = record.get("General") or {}
    tax = gen.get("NCBI tax id")
    if isinstance(tax, list):
        for pref in ("species", "strain", "genus"):
            for item in tax:
                if isinstance(item, dict) and str(item.get("Matching level", "")).lower() == pref:
                    tid = item.get("NCBI tax id")
                    if tid is not None:
                        return str(tid)
        for item in tax:
            if isinstance(item, dict) and item.get("NCBI tax id") is not None:
                return str(item["NCBI tax id"])
            if isinstance(item, (int, float)):
                return str(int(item))
        return ""
    if isinstance(tax, dict):
        tid = tax.get("NCBI tax id")
        return str(tid) if tid is not None else ""
    if tax is not None and str(tax).strip() not in ("", "nan"):
        return str(tax)
    return ""


def api_get(url: str, retries: int = 3):
    last_err = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(SLEEP * (attempt + 2))
    return {"_error": str(last_err)}


def taxon_ids(genus: str, epithet: str | None) -> list[int]:
    url = f"{API}/taxon/{genus}/{epithet}" if epithet else f"{API}/taxon/{genus}"
    data = api_get(url)
    time.sleep(SLEEP)
    if not data or "_error" in data:
        return []
    ids = []
    for x in data.get("results") or []:
        if isinstance(x, int):
            ids.append(x)
        elif isinstance(x, dict):
            for key in ("BacDive-ID", "id", "bacdive_id"):
                if key in x:
                    ids.append(int(x[key]))
                    break
    return ids


def culture_ids(codes: list[str]) -> list[int]:
    """Résout via /v2/culturecollectionno/DSM%2012555."""
    from urllib.parse import quote

    ids: list[int] = []
    for code in codes:
        url = f"{API}/culturecollectionno/{quote(code)}"
        data = api_get(url)
        time.sleep(SLEEP)
        if not data or "_error" in data:
            continue
        for x in data.get("results") or []:
            if isinstance(x, int):
                ids.append(x)
            elif isinstance(x, dict):
                for key in ("BacDive-ID", "id"):
                    if key in x:
                        ids.append(int(x[key]))
                        break
    # unique
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def fetch_ids(ids: list[int]) -> dict:
    if not ids:
        return {}
    joined = ";".join(str(i) for i in ids[:20])
    data = api_get(f"{API}/fetch/{joined}")
    time.sleep(SLEEP)
    if not data or "_error" in data:
        return {"_error": data}
    return data.get("results") or {}


def match_by_culture_in_records(codes: list[str], records: dict) -> list[tuple[str, dict]]:
    """Filtre les records dont DSM-Number / JSON contient un code collection."""
    if not codes or not isinstance(records, dict):
        return []
    needles = []
    for c in codes:
        parts = c.split()
        if len(parts) == 2:
            needles.append((parts[0].upper(), parts[1], c))
    matched = []
    for bid, rec in records.items():
        if str(bid).startswith("_") or not isinstance(rec, dict):
            continue
        gen = rec.get("General") or {}
        dsm = str(gen.get("DSM-Number") or "")
        blob = json.dumps(rec).upper()
        ok = False
        for coll, num, _full in needles:
            if coll == "DSM" and dsm and dsm == num:
                ok = True
                break
            if f"{coll} {num}" in blob or f"{coll}_{num}" in blob or f"{coll}-{num}" in blob:
                ok = True
                break
        if ok:
            matched.append((str(bid), rec))
    return matched


def fill_from_record(result: dict, bid: str, rec: dict, label_texte: str, note: str = "") -> dict:
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
    if note:
        result["commentaire"] = (result.get("commentaire") or "")
        result["commentaire"] = (result["commentaire"] + "; " if result["commentaire"] else "") + note
    if statut == "a_verifier" and oxy:
        result["commentaire"] = f"À vérifier: label Alida={label_texte} vs BacDive={oxy}" + (
            f" ({note})" if note else ""
        )
    return result


def process_row(row: pd.Series) -> dict:
    souche = row["souche"]
    label_texte = str(row["label_texte"])
    genus, epithet = souche_to_species(souche)
    species_query = f"{genus} {epithet}" if epithet else genus
    codes = extract_culture_codes(souche)

    ids: list[int] = []
    resolved_via = ""
    existing_id = row.get("bacdive_id")
    if pd.notna(existing_id) and str(existing_id).strip() not in ("", "nan"):
        try:
            ids = [int(float(existing_id))]
            resolved_via = "bacdive_id_existant"
        except Exception:
            ids = []

    # 1) priorité : numéros de collection dans le nom
    if not ids and codes:
        ids = culture_ids(codes)
        if ids:
            resolved_via = f"culturecollection:{','.join(codes[:3])}"

    # 2) sinon taxon espèce
    if not ids:
        ids = taxon_ids(genus, epithet)
        if ids:
            resolved_via = "taxon"

    cache_path = CACHE / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', souche)[:120]}.json"
    result = {
        "souche": souche,
        "label_texte": label_texte,
        "species_query": species_query,
        "culture_codes": ";".join(codes),
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
        result["commentaire"] = "Aucun hit taxon/culture BacDive"
        return result

    if len(ids) > 1:
        # si on a des codes, tenter de désambiguïser sur un échantillon élargi
        records = fetch_ids(ids[:20])
        cache_path.write_text(
            json.dumps({"ids": ids, "resolved_via": resolved_via, "sample": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if codes:
            matched = match_by_culture_in_records(codes, records if isinstance(records, dict) else {})
            if len(matched) == 1:
                bid, rec = matched[0]
                result["n_bacdive_hits"] = len(ids)
                return fill_from_record(
                    result, bid, rec, label_texte, note=f"désambiguïsé via {codes[0]}"
                )

        oxys = []
        if isinstance(records, dict):
            for bid, rec in records.items():
                if str(bid).startswith("_") or not isinstance(rec, dict):
                    continue
                oxys.append((str(bid), extract_oxygen(rec), extract_taxid(rec)))
        result["commentaire"] = f"Ambigu: {len(ids)} souches BacDive"
        result["statut_propose"] = "a_verifier"
        unique_oxy = {o for _, o, _ in oxys if o}
        if len(oxys) >= 1 and len(unique_oxy) == 1:
            bid, oxy, tax = oxys[0]
            result["bacdive_id_propose"] = bid
            result["oxygen_tol"] = oxy or ""
            result["ncbi_taxid_propose"] = tax
            result["preuve_proposee"] = (
                f"BacDive oxygen tolerance={oxy} (ID {bid}; {len(ids)} hits espèce)"
            )
        else:
            result["preuve_proposee"] = (
                f"BacDive multi-hits ({len(ids)}) oxygen_tol diverge ou absente"
            )
        return result

    bid = ids[0]
    records = fetch_ids([bid])
    cache_path.write_text(
        json.dumps({"resolved_via": resolved_via, "data": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if isinstance(records, dict) and "_error" in records:
        result["commentaire"] = f"Erreur fetch: {records['_error']}"
        return result

    rec = {}
    if isinstance(records, dict):
        rec = records.get(str(bid)) or records.get(bid) or {}
        if not rec and len(records) == 1:
            rec = next(iter(records.values()))
    if not isinstance(rec, dict) or not rec:
        result["commentaire"] = "Fetch vide"
        return result

    return fill_from_record(result, str(bid), rec, label_texte, note=resolved_via or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lot", required=True, help="Numéro de lot (ex: 1)")
    ap.add_argument(
        "--genus",
        nargs="*",
        default=None,
        help="Genre(s) à traiter. Ignoré si --all-partial.",
    )
    ap.add_argument(
        "--all-partial",
        action="store_true",
        help="Traiter toutes les souches encore statut=partiel.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"bacdive_lot{args.lot}.csv"
    out_summary = OUT_DIR / f"bacdive_lot{args.lot}_summary.txt"

    df = pd.read_csv(PROV)
    if args.all_partial:
        lot = df[df["statut_provenance"].astype(str) == "partiel"].copy()
        genera = ["ALL_PARTIAL"]
        print(f"Lot {args.lot} — TOUTES les partiel: {len(lot)} souches")
    else:
        if not args.genus:
            raise SystemExit("Fournir --genus ... ou --all-partial")
        genera = list(args.genus)
        mask = df["souche"].astype(str).str.split("_").str[0].isin(genera)
        lot = df[mask].copy()
        print(f"Lot {args.lot} — genres {genera}: {len(lot)} souches")
    print("table_provenance.csv NON modifié")

    rows = []
    for i, (_, row) in enumerate(lot.iterrows(), 1):
        print(f"[{i}/{len(lot)}] {row['souche']}")
        rows.append(process_row(row))
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  checkpoint -> {out_csv} ({i} lignes)")

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    lines = [
        f"Lot {args.lot} — genres {genera} — {len(out)} souches",
        f"Fichier: {out_csv}",
        "table_provenance.csv NON modifié",
        "",
        out["statut_propose"].value_counts().to_string(),
        "",
        out[
            [
                "souche",
                "label_texte",
                "n_bacdive_hits",
                "bacdive_id_propose",
                "oxygen_tol",
                "statut_propose",
                "commentaire",
            ]
        ].to_string(index=False),
    ]
    out_summary.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + out["statut_propose"].value_counts().to_string())
    print("OK ->", out_csv)


if __name__ == "__main__":
    main()
