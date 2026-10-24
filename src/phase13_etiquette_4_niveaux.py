# -*- coding: utf-8 -*-
"""
Phase 13 — Étiquette à quatre niveaux depuis BacDive (version rapide).

Transforme la chaîne d'oxygène BacDive déjà récupérée (Phase 11) en
quatre niveaux de relation à l'oxygène :
  1 = aérobie (aérobe strict)
  2 = facultatif (anaérobie facultatif / aérobe facultatif)
  3 = microaérophile
  4 = anaérobie (anaérobe strict)
  NR = non renseigné (aucune chaîne retrouvée)

NOMMAGE HONNÊTE : ces niveaux décrivent la relation de l'organisme à
l'oxygène telle que consignée par BacDive. Ils ne constituent pas
encore l'étiquette de CONDITION DE FIXATION avec preuve (réduction
d'acétylène, croissance diazotrophe sous pO2 définie) demandée par
l'Annexe B : la colonne preuve_fixation est laissée explicitement
« non consignée » plutôt que remplie par conjecture.

Sorties :
- results/provenance/table_etiquette_4_niveaux.csv
- results/metrics/PHASE13_ETIQUETTE4_DECISION.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "provenance" / "table_provenance_complete.csv"
OUT = ROOT / "results" / "provenance" / "table_etiquette_4_niveaux.csv"
DECISION = ROOT / "results" / "metrics" / "PHASE13_ETIQUETTE4_DECISION.txt"

LIBELLES = {
    "1": "aérobie (aérobe strict)",
    "2": "facultatif",
    "3": "microaérophile",
    "4": "anaérobie (anaérobe strict)",
    "NR": "non renseigné (aucune chaîne BacDive retrouvée)",
}


def niveau(chaine: str) -> str:
    t = str(chaine).strip().lower()
    if not t or t == "nan":
        return "NR"
    if "microaerophil" in t or "microaérophil" in t:
        return "3"
    if "facult" in t:
        return "2"
    if "anaerob" in t or "anaérob" in t:
        return "4"
    if "aerob" in t or "aérob" in t:
        return "1"
    return "NR"


def main():
    df = pd.read_csv(SRC)
    niv = [niveau(c) for c in df["chaine_oxygene_bacdive"]]
    out = pd.DataFrame({
        "souche": df["souche"],
        "etiquette_binaire_heritee": df["etiquette_heritee"],
        "chaine_oxygene_bacdive": df["chaine_oxygene_bacdive"],
        "niveau_4_relation_oxygene": niv,
        "niveau_4_libelle": [LIBELLES[n] for n in niv],
        "base_niveau": "BacDive — champ « tolérance à l'oxygène » (chaîne ci-dessus)",
        "preuve_fixation_4_niveaux": "non consignée — à construire (preuve de fixation sous pO2 définie, au moins pour Strict)",
        "corpus_strict": df["corpus_strict"],
        "corpus_etendu": df["corpus_etendu"],
        "decision_provenance": df["decision"],
    })
    out.to_csv(OUT, index=False)

    lines = [
        "PHASE 13 — ÉTIQUETTE À QUATRE NIVEAUX DEPUIS BacDive (version rapide)",
        "",
        "Effectifs par niveau (relation à l'oxygène, BacDive) :",
    ]
    for corpus, sub in [("full (787)", out), ("strict_489", out[out["corpus_strict"] == "oui"])]:
        counts = sub["niveau_4_relation_oxygene"].value_counts().to_dict()
        detail = " · ".join(f"niv. {k} ({LIBELLES[k].split(' (')[0]}) = {counts.get(k, 0)}" for k in ["1", "2", "3", "4", "NR"])
        lines.append(f"  {corpus:12s} n={len(sub)} : {detail}")
    lines += [
        "",
        "CHAÎNES BacDive rencontrées :",
    ]
    for chaine, n in df["chaine_oxygene_bacdive"].fillna("").value_counts().items():
        if str(chaine).strip():
            lines.append(f"  « {chaine} » : {n} souches -> niveau {niveau(str(chaine))}")
    lines += [
        "",
        "LIMITES (à écrire dans le manuscrit) :",
        "- Ces niveaux décrivent la relation de l'organisme à l'oxygène,",
        "  pas encore la condition d'oxygène de la fixation elle-même.",
        "  Pour les rhizobiums, Azospirillum et cyanobactéries hétérocystées,",
        "  les deux diffèrent : la colonne preuve_fixation_4_niveaux reste",
        "  « non consignée » plutôt que remplie par conjecture.",
        "- Version complète (Annexe B) = preuve de fixation souche par",
        "  souche (réduction d'acétylène, croissance diazotrophe sous air),",
        "  au moins pour les 489 souches Strict : chantier de recherche",
        "  bibliographique, non automatisable proprement.",
        "",
        f"Table : {OUT}",
    ]
    DECISION.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
