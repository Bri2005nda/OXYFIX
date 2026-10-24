# -*- coding: utf-8 -*-
"""
Phase 3 — Raffinement des labels / construction de corpus propres

Regles (labels Alida JAMAIS modifies automatiquement):
- documente          -> CORPUS_STRICT (garder)
- facultatif /
  microaerophile /
  oxygen ambigu /
  manquant /
  conflit BacDive<>Alida -> EXCLURE du corpus propre
- multi-hits avec oxygen
  coherent avec label Alida -> CORPUS_ETENDU (candidat, confiance moyenne)

Sorties:
- data/raw/metadata/table_provenance_phase3.csv
- results/metrics/phase3_corpus_counts.csv
- results/metrics/phase3_eval_nif.csv
- results/figures/phase3_corpus_and_eval.png
- results/metrics/PHASE3_DECISION.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_premium_complete import genus_of, load_nif_cksaap  # noqa: E402

PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance.csv"
OUT_PROV = ROOT / "data" / "raw" / "metadata" / "table_provenance_phase3.csv"
OUT_METRICS = ROOT / "results" / "metrics"
OUT_FIG = ROOT / "results" / "figures"
SEED = 42


def categorize_preuve(preuve: str, statut: str, label_texte: str) -> tuple[str, str, str]:
    """
    Retourne (categorie, action, raison)
    action in {garder_strict, candidat_etendu, exclure}
    """
    statut = str(statut)
    preuve = str(preuve or "")
    pl = preuve.lower()
    lab = str(label_texte).lower()

    if statut == "documente":
        return "documente", "garder_strict", "BacDive coherent avec label Alida"

    if statut == "manquant":
        return "manquant", "exclure", "Pas de preuve oxygen tolerance exploitable"

    # --- a_verifier ---
    if "facultative" in pl:
        return "facultatif", "exclure", "BacDive facultatif: hors binaire aerobie/anaerobie"
    if "microaerophile" in pl:
        return "microaerophile", "exclure", "BacDive microaerophile: hors binaire"

    # conflit clair BacDive vs Alida (sans multi-hits dans le motif principal)
    if "multi-hits" not in pl and "hits esp" not in pl:
        if "oxygen tolerance=anaerobe" in pl and lab == "aerobie":
            return "conflit_bacdive_anaerobe", "exclure", "BacDive anaerobe vs label Alida aerobie"
        if "oxygen tolerance=aerobe" in pl and "anaerobe" not in pl and lab == "anaerobie":
            return "conflit_bacdive_aerobe", "exclure", "BacDive aerobe vs label Alida anaerobie"
        if "oxygen tolerance=none" in pl or "oxygen_tol diverge" in pl:
            return "oxygen_absent_ambigu", "exclure", "Oxygen tolerance absente ou divergente"

    # multi-hits
    if "multi-hits" in pl or "hits esp" in pl:
        if "oxygen_tol diverge" in pl or "oxygen tolerance=none" in pl:
            return "multi_ambigu", "exclure", "Multi-hits BacDive sans oxygen coherent"
        if "facultative" in pl:
            return "facultatif", "exclure", "Multi-hits + facultatif"
        if "microaerophile" in pl:
            return "microaerophile", "exclure", "Multi-hits + microaerophile"
        # coherent avec Alida ?
        if "anaerobe" in pl and "facultative" not in pl and lab == "anaerobie":
            return "multi_coherent_anaerobie", "candidat_etendu", (
                "Multi-hits mais oxygen=anaerobe coherent avec Alida"
            )
        if (
            "aerobe" in pl
            and "anaerobe" not in pl
            and "facultative" not in pl
            and lab == "aerobie"
        ):
            return "multi_coherent_aerobie", "candidat_etendu", (
                "Multi-hits mais oxygen=aerobe coherent avec Alida"
            )
        return "multi_autre", "exclure", "Multi-hits non classable proprement"

    # cas restants a_verifier
    if "oxygen tolerance=anaerobe" in pl and lab == "aerobie":
        return "conflit_bacdive_anaerobe", "exclure", "BacDive anaerobe vs Alida aerobie"
    if "oxygen tolerance=aerobe" in pl and lab == "anaerobie":
        return "conflit_bacdive_aerobe", "exclure", "BacDive aerobe vs Alida anaerobie"

    return "a_verifier_autre", "exclure", "Cas a_verifier non promu automatiquement"


def build_phase3_table(prov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in prov.iterrows():
        cat, action, raison = categorize_preuve(
            r.get("preuve_label"), r.get("statut_provenance"), r.get("label_texte")
        )
        in_strict = action == "garder_strict"
        in_etendu = action in ("garder_strict", "candidat_etendu")
        rows.append(
            {
                **r.to_dict(),
                "phase3_categorie": cat,
                "phase3_action": action,
                "phase3_raison": raison,
                "corpus_strict": "oui" if in_strict else "non",
                "corpus_etendu": "oui" if in_etendu else "non",
                # labels Alida inchanges
                "label_alida_inchange": True,
            }
        )
    return pd.DataFrame(rows)


def eval_on_subset(souches_all, X_all, y_all, groups_all, keep_set, corpus_name: str):
    mask = np.array([s in keep_set for s in souches_all])
    if mask.sum() < 40:
        return []
    X, y, groups = X_all[mask], y_all[mask], groups_all[mask]
    if len(np.unique(y)) < 2:
        return []
    out = []
    # random
    strat = y if min(np.bincount(y.astype(int))) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strat
    )
    try:
        Xtr_s, ytr_s = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    except ValueError:
        Xtr_s, ytr_s = Xtr, ytr
    model = HistGradientBoostingClassifier(random_state=SEED, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out.append(
        {
            "corpus": corpus_name,
            "n": int(len(y)),
            "n_aerobie": int((y == 1).sum()),
            "n_anaerobie": int((y == 0).sum()),
            "protocol": "random_80_20_SMOTE",
            "accuracy": float(accuracy_score(yte, pred)),
            "auc": float(roc_auc_score(yte, proba)),
            "mcc": float(matthews_corrcoef(yte, pred)),
        }
    )
    # genus
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    try:
        Xtr_s, ytr_s = SMOTE(random_state=SEED).fit_resample(X[tr], y[tr])
    except ValueError:
        Xtr_s, ytr_s = X[tr], y[tr]
    if len(np.unique(y[te])) < 2:
        return out
    model = HistGradientBoostingClassifier(random_state=SEED, max_depth=6)
    model.fit(Xtr_s, ytr_s)
    proba = model.predict_proba(X[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    out.append(
        {
            "corpus": corpus_name,
            "n": int(len(y)),
            "n_aerobie": int((y == 1).sum()),
            "n_anaerobie": int((y == 0).sum()),
            "protocol": "group_shuffle_genus_SMOTE_train",
            "accuracy": float(accuracy_score(y[te], pred)),
            "auc": float(roc_auc_score(y[te], proba)),
            "mcc": float(matthews_corrcoef(y[te], pred)),
            "genres_overlap": int(len(set(groups[tr]) & set(groups[te]))),
        }
    )
    return out


def main():
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    prov = pd.read_csv(PROV)
    p3 = build_phase3_table(prov)
    p3.to_csv(OUT_PROV, index=False)

    # counts
    counts = (
        p3.groupby(["phase3_categorie", "phase3_action", "label_texte"])
        .size()
        .reset_index(name="n")
        .sort_values(["phase3_action", "n"], ascending=[True, False])
    )
    counts_path = OUT_METRICS / "phase3_corpus_counts.csv"
    counts.to_csv(counts_path, index=False)

    n_strict = int((p3["corpus_strict"] == "oui").sum())
    n_etendu = int((p3["corpus_etendu"] == "oui").sum())
    n_excl = int((p3["phase3_action"] == "exclure").sum())
    n_cand = int((p3["phase3_action"] == "candidat_etendu").sum())

    print("=== PHASE 3 CORPUS ===")
    print(f"strict={n_strict}  etendu={n_etendu}  exclus={n_excl}  candidats={n_cand}")
    print(p3["phase3_action"].value_counts())
    print(p3["phase3_categorie"].value_counts())

    # eval
    souches, X, y, groups = load_nif_cksaap()
    set_full = set(souches)
    set_strict = set(p3.loc[p3["corpus_strict"] == "oui", "souche"].astype(str))
    set_etendu = set(p3.loc[p3["corpus_etendu"] == "oui", "souche"].astype(str))

    rows = []
    rows += eval_on_subset(souches, X, y, groups, set_full, "full_787")
    rows += eval_on_subset(souches, X, y, groups, set_strict, "strict_documente")
    rows += eval_on_subset(souches, X, y, groups, set_etendu, "etendu_documente_plus_coherent")
    df_eval = pd.DataFrame(rows)
    eval_path = OUT_METRICS / "phase3_eval_nif.csv"
    df_eval.to_csv(eval_path, index=False)
    print(df_eval.to_string(index=False))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # A: composition
    ax = axes[0]
    actions = ["garder_strict", "candidat_etendu", "exclure"]
    labels_fr = ["Strict\n(documente)", "Candidats\netendu", "Exclus"]
    vals = [int((p3["phase3_action"] == a).sum()) for a in actions]
    colors = ["#1a7f37", "#1f4e79", "#a04040"]
    ax.bar(labels_fr, vals, color=colors, edgecolor="black", lw=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 5, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Nombre de souches")
    ax.set_title("A) Decisions Phase 3 (labels Alida inchanges)")
    ax.set_ylim(0, max(vals) * 1.15)

    # B: AUC random vs genus
    ax = axes[1]
    corpora = ["full_787", "strict_documente", "etendu_documente_plus_coherent"]
    display = ["Full 787", "Strict", "Etendu"]
    x = np.arange(len(corpora))
    w = 0.35
    auc_r, auc_g = [], []
    for c in corpora:
        rr = df_eval[(df_eval["corpus"] == c) & (df_eval["protocol"] == "random_80_20_SMOTE")]
        gg = df_eval[
            (df_eval["corpus"] == c)
            & (df_eval["protocol"] == "group_shuffle_genus_SMOTE_train")
        ]
        auc_r.append(float(rr["auc"].iloc[0]) if len(rr) else np.nan)
        auc_g.append(float(gg["auc"].iloc[0]) if len(gg) else np.nan)
    ax.bar(x - w / 2, auc_r, w, label="Aleatoire", color="#1f4e79")
    ax.bar(x + w / 2, auc_g, w, label="Split genre", color="#8b6914")
    ax.axhline(0.90, color="red", ls="--", lw=1)
    ax.axhline(0.70, color="orange", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC Nif CKSAAP (HGB)")
    ax.set_title("B) Performance selon proprete du corpus")
    ax.legend(fontsize=8)
    for i in range(len(corpora)):
        if not np.isnan(auc_r[i]):
            ax.text(i - w / 2, auc_r[i] + 0.01, f"{auc_r[i]:.2f}", ha="center", fontsize=7)
        if not np.isnan(auc_g[i]):
            ax.text(i + w / 2, auc_g[i] + 0.01, f"{auc_g[i]:.2f}", ha="center", fontsize=7)

    fig.suptitle(
        "Phase 3 — Corpus propres et reevaluation Nif",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = OUT_FIG / "phase3_corpus_and_eval.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # decision text
    def get_auc(corpus, protocol):
        r = df_eval[(df_eval["corpus"] == corpus) & (df_eval["protocol"] == protocol)]
        return float(r["auc"].iloc[0]) if len(r) else float("nan")

    def get_n(corpus):
        r = df_eval[df_eval["corpus"] == corpus]
        return int(r["n"].iloc[0]) if len(r) else 0

    lines = [
        "PHASE 3 — RAFFINEMENT LABELS / CORPUS PROPRES",
        "Principe: labels Alida NON modifies automatiquement.",
        "",
        "REGLES:",
        "  garder_strict   = statut documente (BacDive coherent)",
        "  candidat_etendu = multi-hits BacDive mais oxygen coherent avec Alida",
        "  exclure         = facultatif, microaerophile, manquant, ambigu, conflit",
        "",
        "EFFECTIFS:",
        f"  Full 787              : {get_n('full_787')}",
        f"  Strict (documente)    : {n_strict}",
        f"  Etendu (strict+cand.) : {n_etendu}",
        f"  Exclus                : {n_excl}",
        "",
        "CATEGORIES (top):",
    ]
    for cat, n in p3["phase3_categorie"].value_counts().items():
        lines.append(f"  {cat:28s} {int(n)}")

    lines += [
        "",
        "EVALUATION Nif CKSAAP (AUC HGB):",
        f"  Full    aleatoire={get_auc('full_787','random_80_20_SMOTE'):.3f} | "
        f"genre={get_auc('full_787','group_shuffle_genus_SMOTE_train'):.3f}",
        f"  Strict  aleatoire={get_auc('strict_documente','random_80_20_SMOTE'):.3f} | "
        f"genre={get_auc('strict_documente','group_shuffle_genus_SMOTE_train'):.3f}",
        f"  Etendu  aleatoire={get_auc('etendu_documente_plus_coherent','random_80_20_SMOTE'):.3f} | "
        f"genre={get_auc('etendu_documente_plus_coherent','group_shuffle_genus_SMOTE_train'):.3f}",
        "",
        "LECTURE:",
        "- Phase 3 separe ce qui est DEFENDABLE (preuve BacDive) du reste.",
        "- Le corpus STRICT est la base recommandee pour l'article / suite.",
        "- Le corpus ETENDU teste un compromis taille vs confiance.",
        "- Si les conclusions Phase1/2 tiennent sur STRICT, elles sont plus robustes.",
        "- Les exclus (facultatif/microaero/manquant/conflit) restent documentes",
        "  mais hors analyses principales.",
        "",
        "ORIENTATION:",
        "- Ne pas re-labeler automatiquement en aerobie/anaerobie les facultatifs.",
        "- Pour la suite (Phase 4 / article): analyses principales sur STRICT;",
        "  sensibilite eventuelle sur ETENDU.",
        "",
        f"Table : {OUT_PROV}",
        f"Counts: {counts_path}",
        f"Eval  : {eval_path}",
        f"Figure: {fig_path}",
    ]
    dec = OUT_METRICS / "PHASE3_DECISION.txt"
    dec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nOK -> {dec}")


if __name__ == "__main__":
    main()
