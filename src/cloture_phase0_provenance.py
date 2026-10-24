# -*- coding: utf-8 -*-
"""Clôture Phase 0 : fusion lot, manquant, note + réconciliation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"
OUT = ROOT / "results" / "provenance"
NOTE = OUT / "NOTE_PHASE0_CLOTURE.txt"


def merge_lots(lots: list[str]):
    from merge_provenance_bacdive import main as merge_main
    import sys

    sys.argv = ["merge_provenance_bacdive.py", "--lots", *lots]
    merge_main()


def apply_a_verifier_and_manquant():
    prov = pd.read_csv(PROV)
    for c in ("bacdive_id", "ncbi_taxid", "preuve_label", "statut_provenance"):
        prov[c] = prov[c].astype("object")

    frames = []
    for p in sorted(OUT.glob("bacdive_lot*.csv")):
        if "summary" in p.name:
            continue
        frames.append(pd.read_csv(p))
    lots = pd.concat(frames, ignore_index=True).drop_duplicates("souche", keep="last")

    # a_verifier depuis lots
    for _, row in lots[lots["statut_propose"].astype(str) == "a_verifier"].iterrows():
        idx = prov.index[prov["souche"] == row["souche"]]
        if len(idx) == 0:
            continue
        i = idx[0]
        if str(prov.at[i, "statut_provenance"]) != "partiel":
            continue
        bid = row.get("bacdive_id_propose")
        if pd.notna(bid) and str(bid).strip() not in ("", "nan"):
            try:
                prov.at[i, "bacdive_id"] = str(int(float(bid)))
            except Exception:
                prov.at[i, "bacdive_id"] = str(bid)
        if pd.notna(row.get("preuve_proposee")) and str(row["preuve_proposee"]).strip():
            prov.at[i, "preuve_label"] = str(row["preuve_proposee"])
        prov.at[i, "statut_provenance"] = "a_verifier"

    # partiel restants : classer manquant si aucun hit / pas d'oxygen (après retries)
    last = lots.set_index("souche", drop=False)
    n_manquant = 0
    for i in prov.index[prov["statut_provenance"].astype(str) == "partiel"]:
        souche = prov.at[i, "souche"]
        if souche not in last.index:
            prov.at[i, "statut_provenance"] = "manquant"
            prov.at[i, "preuve_label"] = "non_trouve_BacDive_apres_requetes"
            n_manquant += 1
            continue
        row = last.loc[souche]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        com = str(row.get("commentaire") or "")
        oxy = str(row.get("oxygen_tol") or "").strip()
        hits = int(row.get("n_bacdive_hits") or 0)
        # erreurs réseau : laisser partiel pour retry futur
        if "Connection aborted" in com or "RemoteDisconnected" in com or "Erreur fetch" in com:
            continue
        if hits == 0 or "Aucun hit" in com:
            prov.at[i, "statut_provenance"] = "manquant"
            prov.at[i, "preuve_label"] = "non_trouve_BacDive_apres_requetes"
            n_manquant += 1
        elif (not oxy or oxy == "nan") and (
            "Pas de champ oxygen" in com or "oxygen tolerance absente" in com.lower()
        ):
            prov.at[i, "statut_provenance"] = "manquant"
            if pd.notna(row.get("bacdive_id_propose")) and str(row["bacdive_id_propose"]).strip() not in ("", "nan"):
                try:
                    prov.at[i, "bacdive_id"] = str(int(float(row["bacdive_id_propose"])))
                except Exception:
                    pass
            prov.at[i, "preuve_label"] = "BacDive_trouve_mais_oxygen_tolerance_absente"
            n_manquant += 1

    prov.to_csv(PROV, index=False)
    return prov, n_manquant


def reconcile():
    lines = []
    meta = ROOT / "data" / "raw" / "metadata"
    fasta_a = list((ROOT / "data/raw/fasta/Fixateurs_aerobie/Fixateurs_aerobie").glob("*.fasta"))
    fasta_n = list((ROOT / "data/raw/fasta/Fixateurs_anaerobie/Fixateurs_anaerobie").glob("*.fasta"))
    labels = pd.read_csv(meta / "table_labels.csv")
    prov = pd.read_csv(meta / "table_provenance.csv")
    lines.append("=== RÉCONCILIATION COMPTAGES (Phase 0) ===")
    lines.append(f"FASTA aéro: {len(fasta_a)} | anaéro: {len(fasta_n)} | total: {len(fasta_a)+len(fasta_n)}")
    lines.append(
        f"table_labels: {len(labels)} | aéro(1): {(labels.label==1).sum()} | anaéro(0): {(labels.label==0).sum()}"
    )
    lines.append(f"table_provenance: {len(prov)}")
    lines.append(prov["statut_provenance"].value_counts().to_string())
    lines.append(f"Cible projet: 787 / 356 aéro / 431 anaéro")
    lines.append(
        f"Écart labels vs cible: total {len(labels)-787}, aéro {(labels.label==1).sum()-356}, anaéro {(labels.label==0).sum()-431}"
    )
    return "\n".join(lines)


def write_note(prov: pd.DataFrame, n_manquant: int):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    vc = prov["statut_provenance"].value_counts()
    recon = reconcile()
    text = f"""NOTE DE CLÔTURE — PHASE 0 (OXYFIX)
Date: {stamp}
Candidate: LEMOGUE JIOFACK BRINDA
Cadre: plan détaillé OXYFIX — Phase 0 (reproduction + provenance)

1. OBJECTIF PHASE 0
Relancer le pipeline hérité, constituer la table de provenance, réconcilier les comptages.
Les labels Alida (aérobie/anaérobie) n'ont PAS été modifiés automatiquement.

2. ÉTAT DE LA TABLE DE PROVENANCE (787 souches)
{vc.to_string()}

Définitions:
- documente : BacDive trouvé, oxygen tolerance cohérente avec le label Alida.
- a_verifier : BacDive ambigu (plusieurs souches), oxygen tolerance floue
  (facultative / microaerophile), ou conflit léger à trancher manuellement / Phase 3.
- manquant : absente de BacDive ou sans champ oxygen tolerance après requêtes.
- partiel : encore en attente (ex. erreur réseau à rejouer).

3. TRAVAIL RÉALISÉ
- Scripts: src/enrich_provenance_bacdive.py, src/merge_provenance_bacdive.py
- Lots BacDive 0 à 11 + cache results/provenance/bacdive_cache/
- Backups: data/raw/metadata/table_provenance_backup_*.csv
- Nouveaux classés manquant cette clôture: {n_manquant}

4. RÈGLE POUR LES  a_verifier (premium)
Ne pas forcer documente. À traiter en Phase 3 (recuration des labels) ou revue manuelle
par genre prioritaire. En attendant: exclus d'un sous-ensemble « gold » si besoin.

5. RÉCONCILIATION DES COMPTES
{recon}

6. LIVRABLES PHASE 0
- Dépôt structuré data/ src/ results/
- table_provenance.csv à jour
- résultats lots BacDive + cette note
- essais préliminaires déjà réalisés: reproduction ~0.91; fusion ~0.918;
  chute sous split genre ~0.65 (hors Phase 0 stricte, mais utiles pour l'orientation)

7. CRITÈRE DE SORTIE PHASE 0
Phase 0 considérée CLOTURABLE pour enchaîner Phase 1 lorsque:
- provenance traitée pour l'essentiel (documente + a_verifier + manquant),
- partiel résiduel documenté,
- note de réconciliation disponible (ce fichier).
Les a_verifier ne bloquent pas Phase 1; ils alimentent Phase 3.

8. PROCHAINE ÉTAPE (après validation encadreur)
Phase 1 — contrôles négatifs (RecA, GyrB, protéine ribosomique).
"""
    NOTE.write_text(text, encoding="utf-8")
    (OUT / "reconciliation_counts.txt").write_text(recon, encoding="utf-8")
    print(text)
    print("OK ->", NOTE)


def main():
    # fusion documente lots 0-11 si lot11 existe
    lot_nums = []
    for i in range(0, 12):
        if (OUT / f"bacdive_lot{i}.csv").exists():
            lot_nums.append(str(i))
    print("Merge lots:", lot_nums)
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, str(ROOT / "src" / "merge_provenance_bacdive.py"), "--lots", *lot_nums],
        cwd=str(ROOT),
    )
    prov, n_manquant = apply_a_verifier_and_manquant()
    write_note(prov, n_manquant)


if __name__ == "__main__":
    main()
