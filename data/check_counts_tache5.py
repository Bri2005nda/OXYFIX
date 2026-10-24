# check_counts_tache5.py — Tâche 5 : réconciliation des comptages
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent

fasta_aero = list((root / "data/raw/fasta/Fixateurs_aerobie/Fixateurs_aerobie").glob("*.fasta"))
fasta_ana  = list((root / "data/raw/fasta/Fixateurs_anaerobie/Fixateurs_anaerobie").glob("*.fasta"))

liste = pd.read_csv(root / "data/raw/metadata/liste_souches.csv")
labels = pd.read_csv(root / "data/raw/metadata/table_labels.csv")
prov = pd.read_csv(root / "data/raw/metadata/table_provenance.csv")

ck_aero = pd.read_csv(root / "data/raw/Resultats/Fixateurs_aerobie/CKSAAP type 1.csv")
ck_ana  = pd.read_csv(root / "data/raw/Resultats/Fixateurs_anaerobie/CKSAAP type 1.csv")

print("=== FASTA ===")
print("aéro:", len(fasta_aero), "| anaéro:", len(fasta_ana), "| total:", len(fasta_aero)+len(fasta_ana))

print("\n=== CSV metadata ===")
print("liste_souches:", len(liste), "| aéro:", (liste.groupe=="aerobie").sum(), "| anaéro:", (liste.groupe=="anaerobie").sum())
print("table_labels:", len(labels), "| label1:", (labels.label==1).sum(), "| label0:", (labels.label==0).sum())
print("table_provenance:", len(prov), "| documente:", (prov.statut_provenance=="documente").sum(), "| partiel:", (prov.statut_provenance=="partiel").sum())

print("\n=== CKSAAP ===")
print("CSV aéro lignes:", len(ck_aero), "| CSV anaéro lignes:", len(ck_ana))

print("\n=== Doublons souches? ===")
print("liste doublons:", liste["souche"].duplicated().sum())
print("labels doublons:", labels["souche"].duplicated().sum())

print("\n=== Écarts liste vs labels ===")
s_liste = set(liste["souche"])
s_lab = set(labels["souche"])
print("dans liste pas labels:", len(s_liste - s_lab))
print("dans labels pas liste:", len(s_lab - s_liste))

print("\nCIBLE: tout à 787 / 356 / 431")