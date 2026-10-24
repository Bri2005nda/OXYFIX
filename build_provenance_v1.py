# build_provenance_v1.py — Tâche 4 (version 1 : base depuis table_labels)
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
src = root / "data/raw/metadata/table_labels.csv"
out = root / "data/raw/metadata/table_provenance.csv"

df = pd.read_csv(src)

prov = pd.DataFrame({
    "souche": df["souche"],
    "label": df["label"],
    "label_texte": df["label_texte"],
    "groupe": df["groupe"],
    "fichier_fasta": df["fichier_fasta"],
    "source_sequence": df["source_sequence"],
    "source_label": df["source_label"],
    # À compléter ensuite (BacDive / NCBI)
    "bacdive_id": "",
    "ncbi_taxid": "",
    "preuve_label": "herite_dossier_Alida_(a_confirmer_BacDive)",
    "statut_provenance": "partiel",  # partiel | documente | a_verifier | manquant
})

prov.to_csv(out, index=False)
print("OK →", out)
print("Lignes :", len(prov))
print(prov.head(3).to_string(index=False))