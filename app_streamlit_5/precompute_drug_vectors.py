"""
precompute_drug_vectors.py
===========================
Calcule UNE FOIS, par substance :
  1. Les vecteurs de profil d'effets (mode proportionnel et mode TF-IDF) à
     partir de sentences.csv, projetés en 2D via UMAP, puis clusterisés via
     HDBSCAN.
  2. La projection 2D (UMAP) des embeddings sémantiques moyens fournis dans
     data/mean_embeddings_by_substance.csv (moyenne des embeddings de phrases
     décrivant les effets subjectifs de la substance), également clusterisée
     via HDBSCAN.

Ce sont DEUX TYPES DE DONNÉES bien distincts (profil d'effets catégoriel à
faible dimension vs embeddings sémantiques denses à 768 dimensions) : leurs
paramètres UMAP et HDBSCAN sont donc volontairement séparés en deux blocs de
constantes (EFFECTS_* / SEMANTIC_*) plutôt que partagés, voir plus bas.

Le résultat est FIGÉ dans les fichiers de données :

  - data/sentences.csv : ajoute les colonnes "drug", "chem_family" et
    "pharm_family" (persistées).

  - data/reports.json : ajoute une clé de premier niveau "_drug_vectors"
    contenant, pour chaque substance, son nombre de phrases, ses 3 effets
    dominants, ses coordonnées UMAP ("umap_prop" / "umap_tfidf" / "umap_semantic")
    et le cluster HDBSCAN correspondant ("cluster_prop" / "cluster_tfidf" /
    "cluster_semantic" — entier, -1 = bruit / non clusterisé).

À relancer uniquement si les données sources changent. L'appli Streamlit, elle,
ne fait plus AUCUN calcul UMAP/HDBSCAN : elle lit directement les coordonnées
et labels de cluster déjà stockés.

Dépendances (seulement pour CE script, pas pour l'appli) :
    pip install umap-learn hdbscan scikit-learn pandas numpy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import umap
import hdbscan

DATA_DIR = Path(__file__).parent / "data"
SENTENCES_PATH = DATA_DIR / "sentences.csv"
REPORTS_PATH = DATA_DIR / "reports.json"
EMBEDDINGS_PATH = DATA_DIR / "mean_embeddings_by_substance.csv"

# Alias pour corriger les écarts de casse/orthographe connus entre le fichier
# d'embeddings et les noms de substances utilisés dans sentences.csv/reports.json.
EMBEDDING_NAME_ALIASES = {
    "Nitrous oxide": "Nitrous Oxide",
    "Psilocybin Mushroom": "Psilocybin Mushrooms",
}

# Seuil minimum de phrases (top1_label != "None") pour qu'une substance soit
# incluse dans les cartes. En-dessous, le vecteur est trop bruité pour être
# significatif. Les substances exclues restent dans sentences.csv / reports.json
# normalement — elles sont juste absentes de "_drug_vectors".
MIN_PHRASES = 25
SEED = 42

# ==============================================================================
# TYPE DE DONNÉES 1 — PROFIL D'EFFETS (proportionnel & TF-IDF)
# Vecteurs catégoriels, faible dimension (~23 effets), dérivés du comptage des
# labels top-1 par phrase. Structure plutôt discrète (des substances peuvent
# partager exactement les mêmes effets dominants).
# ==============================================================================
UMAP_PARAMS_EFFECTS = dict(
    n_neighbors=3,
    min_dist=0.10,
    metric="cosine",
    random_state=SEED,
    n_components=2,
)
HDBSCAN_PARAMS_EFFECTS = dict(
    min_cluster_size=5,
    min_samples=2,
    cluster_selection_epsilon=0.0,
    metric="euclidean",       # appliqué sur les coordonnées UMAP 2D, pas sur les vecteurs d'origine
)

# ==============================================================================
# TYPE DE DONNÉES 2 — EMBEDDINGS SÉMANTIQUES MOYENS (768 dimensions)
# Vecteurs denses issus d'un modèle de langage (moyenne des embeddings de
# phrases par substance). Structure plus continue/lissée par la moyenne.
# ==============================================================================
UMAP_PARAMS_SEMANTIC = dict(
    n_neighbors=7,
    min_dist=0.05,
    metric="cosine",
    random_state=SEED,
    n_components=2,
)
HDBSCAN_PARAMS_SEMANTIC = dict(
    min_cluster_size=5,       # vecteurs plus "lisses" (moyennés) : on exige un groupe un peu plus net
    min_samples=3,
    cluster_selection_epsilon=0.0,
    metric="euclidean",        # appliqué sur les coordonnées UMAP 2D, pas sur les 768 dimensions
)


def load_raw():
    sentences = pd.read_csv(SENTENCES_PATH)
    with open(REPORTS_PATH, encoding="utf-8") as f:
        reports = json.load(f)
    # repartir d'une clé propre si le script a déjà tourné une fois
    reports.pop("_drug_vectors", None)
    return sentences, reports


def attach_drug_and_family(sentences: pd.DataFrame, reports: dict) -> pd.DataFrame:
    """Ajoute 'drug', 'chem_family' et 'pharm_family' à sentences (depuis
    reports.json et la taxonomie codée en dur dans drug_taxonomy.py)."""
    from drug_taxonomy import chem_family, pharm_family
    sentences = sentences.copy()
    sentences["drug"] = sentences["titre"].map(lambda t: reports.get(t, {}).get("drug", "Inconnu"))
    sentences["chem_family"] = sentences["drug"].map(chem_family)
    sentences["pharm_family"] = sentences["drug"].map(pharm_family)
    return sentences


def build_count_matrix(sentences: pd.DataFrame) -> pd.DataFrame:
    """Matrice substances × effets : nb de phrases où l'effet est en top-1 (None exclu)."""
    work = sentences[sentences["top1_label"].notna() & (sentences["top1_label"] != "None")]
    return pd.crosstab(work["drug"], work["top1_label"])


def vectorize(counts: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "tfidf":
        transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
        return transformer.fit_transform(counts.values).toarray()
    row_sums = counts.sum(axis=1).replace(0, 1)
    return counts.div(row_sums, axis=0).values


def load_semantic_embeddings(eligible_drugs) -> pd.DataFrame | None:
    """Charge data/mean_embeddings_by_substance.csv, corrige les alias connus,
    et restreint aux substances retenues pour la carte d'effets (eligible_drugs)."""
    if not EMBEDDINGS_PATH.exists():
        print(f"Pas de fichier {EMBEDDINGS_PATH.name} trouvé — carte sémantique ignorée.")
        return None
    emb = pd.read_csv(EMBEDDINGS_PATH)
    emb["substance"] = emb["substance"].replace(EMBEDDING_NAME_ALIASES)
    emb = emb.drop_duplicates("substance", keep="first").set_index("substance")
    emb_cols = [c for c in emb.columns if c.startswith("embedding_")]
    emb = emb[emb_cols]

    common = [d for d in eligible_drugs if d in emb.index]
    missing = [d for d in eligible_drugs if d not in emb.index]
    if missing:
        print(f"{len(missing)} substance(s) sans embedding sémantique (ignorées pour cette carte) : "
              f"{missing}")
    return emb.loc[common] if common else None


def run_umap(vectors: np.ndarray, params: dict) -> np.ndarray:
    """params doit être UMAP_PARAMS_EFFECTS ou UMAP_PARAMS_SEMANTIC selon le
    type de données — voir les deux blocs de constantes en tête de fichier."""
    params = dict(params)
    params["n_neighbors"] = max(2, min(params["n_neighbors"], vectors.shape[0] - 1))
    reducer = umap.UMAP(**params)
    return reducer.fit_transform(vectors)


def run_hdbscan(coords_2d: np.ndarray, params: dict) -> np.ndarray:
    """Clustering APRÈS UMAP : on clusterise les coordonnées 2D déjà projetées
    (pas les vecteurs d'origine), pour rester cohérent avec ce qui est affiché.
    params doit être HDBSCAN_PARAMS_EFFECTS ou HDBSCAN_PARAMS_SEMANTIC."""
    params = dict(params)
    params["min_cluster_size"] = max(2, min(params["min_cluster_size"], coords_2d.shape[0] - 1))
    clusterer = hdbscan.HDBSCAN(**params)
    return clusterer.fit_predict(coords_2d)


def cluster_summary(labels: np.ndarray) -> str:
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    return f"{n_clusters} clusters, {n_noise}/{len(labels)} points en bruit"


def top_effects(sentences: pd.DataFrame, drug: str, n: int = 3) -> list:
    sub = sentences[(sentences["drug"] == drug) & (sentences["top1_label"] != "None")]
    return sub["top1_label"].value_counts().head(n).index.tolist()


def main():
    sentences, reports = load_raw()
    sentences = attach_drug_and_family(sentences, reports)

    counts = build_count_matrix(sentences)
    sizes = counts.sum(axis=1)
    keep = sizes[sizes >= MIN_PHRASES].index
    counts_kept = counts.loc[keep]

    print(f"{len(counts)} substances au total, {len(counts_kept)} retenues "
          f"(seuil >= {MIN_PHRASES} phrases à effet non-None).")

    drug_vectors = {}

    # --- Type 1 : profils d'effets (proportionnel + TF-IDF) ---
    if len(counts_kept) >= 4:
        coords_prop = run_umap(vectorize(counts_kept, "prop"), UMAP_PARAMS_EFFECTS)
        coords_tfidf = run_umap(vectorize(counts_kept, "tfidf"), UMAP_PARAMS_EFFECTS)
        clusters_prop = run_hdbscan(coords_prop, HDBSCAN_PARAMS_EFFECTS)
        clusters_tfidf = run_hdbscan(coords_tfidf, HDBSCAN_PARAMS_EFFECTS)

        print(f"  Clusters (proportionnel) : {cluster_summary(clusters_prop)}")
        print(f"  Clusters (TF-IDF)        : {cluster_summary(clusters_tfidf)}")

        for i, drug in enumerate(counts_kept.index):
            drug_vectors[drug] = {
                "n_phrases": int(sizes[drug]),
                "top_effets": top_effects(sentences, drug),
                "umap_prop": [round(float(coords_prop[i, 0]), 4), round(float(coords_prop[i, 1]), 4)],
                "umap_tfidf": [round(float(coords_tfidf[i, 0]), 4), round(float(coords_tfidf[i, 1]), 4)],
                "cluster_prop": int(clusters_prop[i]),
                "cluster_tfidf": int(clusters_tfidf[i]),
            }
    else:
        print("Pas assez de substances au-dessus du seuil pour calculer une carte UMAP.")

    # --- Type 2 : carte sémantique (embeddings moyens de phrases) ---
    semantic_emb = load_semantic_embeddings(counts_kept.index)
    if semantic_emb is not None and len(semantic_emb) >= 4:
        coords_semantic = run_umap(semantic_emb.values, UMAP_PARAMS_SEMANTIC)
        clusters_semantic = run_hdbscan(coords_semantic, HDBSCAN_PARAMS_SEMANTIC)
        print(f"  Clusters (sémantique)    : {cluster_summary(clusters_semantic)}")

        for i, drug in enumerate(semantic_emb.index):
            drug_vectors.setdefault(drug, {
                "n_phrases": int(sizes.get(drug, 0)),
                "top_effets": top_effects(sentences, drug),
            })
            drug_vectors[drug]["umap_semantic"] = [
                round(float(coords_semantic[i, 0]), 4), round(float(coords_semantic[i, 1]), 4),
            ]
            drug_vectors[drug]["cluster_semantic"] = int(clusters_semantic[i])
        print(f"Carte sémantique calculée pour {len(semantic_emb)} substances "
              f"(dimension des embeddings : {semantic_emb.shape[1]}).")
    else:
        print("Pas assez de substances avec embedding sémantique pour calculer cette carte.")

    # --- écriture sentences.csv (colonnes drug / chem_family / pharm_family persistées) ---
    sentences.to_csv(SENTENCES_PATH, index=False)
    print(f"sentences.csv réécrit avec les colonnes 'drug', 'chem_family' et 'pharm_family' "
          f"({len(sentences)} lignes).")

    # --- écriture reports.json (clé _drug_vectors ajoutée, reste inchangé) ---
    n_semantic = sum(1 for v in drug_vectors.values() if "umap_semantic" in v)
    reports["_drug_vectors"] = {
        "_meta": {
            "min_phrases": MIN_PHRASES,
            "seed": SEED,
            "umap_params_effects": UMAP_PARAMS_EFFECTS,
            "hdbscan_params_effects": HDBSCAN_PARAMS_EFFECTS,
            "umap_params_semantic": UMAP_PARAMS_SEMANTIC,
            "hdbscan_params_semantic": HDBSCAN_PARAMS_SEMANTIC,
            "n_drugs": len(drug_vectors),
            "n_drugs_semantic": n_semantic,
        },
        "drugs": drug_vectors,
    }
    with open(REPORTS_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=1)
    print(f"reports.json réécrit avec la clé '_drug_vectors' "
          f"({len(drug_vectors)} substances, dont {n_semantic} avec carte sémantique).")


if __name__ == "__main__":
    main()
