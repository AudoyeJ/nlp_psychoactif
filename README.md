# Récits des états modifiés de conscience

Application **Streamlit** de data visualisation explorant, par traitement automatique des
langues (NLP), les effets phénoménologiques rapportés dans des récits d'usagers de
substances psychoactives ("trip reports").

Chaque phrase d'un récit a été passée au crible d'un classifieur qui lui attribue ses 3
effets les plus probables (ex. *internal hallucination*, *anxiety*, *time distortion…*).
L'application explore comment ces effets se répartissent, évoluent au fil d'un trip
(montée → pic → descente), et diffèrent selon la substance, sa famille chimique/
pharmacologique, ou la source du récit.

---

## Aperçu des fonctionnalités

| Onglet | Contenu |
|---|---|
| **◆ Vue d'ensemble** | KPIs globaux, effets les plus fréquents, répartition par phase du trip, bande-témoin du rythme moyen d'un trip. |
| **◆ Cartographie** | Deux cartes UMAP des substances (voir détail plus bas), avec aires de cluster HDBSCAN superposables. |
| **◆ Effets & phases** | Distribution des effets par phase (montée/pic/descente), comparaisons. |
| **◆ Substances** | Classement des substances par volume de récits/phrases, répartition par famille chimique. |
| **◆ Explorateur de récits** | Lecture phrase par phrase d'un récit choisi (ou aléatoire), avec bande colorée par effet dominant, texte intégral et conclusion de l'auteur. |
| **◆ Recherche** | Filtrage des phrases par effet, phase, substance, score de confiance et mot-clé ; export CSV. |
| **◆ Sources** | Comparaison effectindex.com vs psychonautwiki.org (volume, profil d'effets, confiance moyenne du classifieur). |

### La Cartographie, en détail

Chaque substance retenue (≥ 25 phrases à effet non-`None`) est représentée par un point
dans un espace 2D obtenu par **UMAP**, selon deux logiques de vectorisation distinctes :

1. **Clusterization via effet subjectif** — le vecteur d'une substance est sa distribution
   d'effets dominants (top-1) sur l'ensemble de ses phrases, en mode *proportionnel*
   (fréquence relative) ou *TF-IDF* (les effets partagés par presque toutes les substances
   sont atténués, les effets distinctifs ressortent).
2. **Clusterization via vecteur sémantique** — le vecteur est la moyenne des *embeddings*
   (768 dimensions) de toutes les phrases décrivant la substance : une comparaison fondée
   sur le langage réellement employé, pas sur un comptage de labels.

Dans les deux cas, on peut colorer les points par **classe chimique** ou **classe
pharmacologique**, et superposer des **ellipses de confiance HDBSCAN** ("Aires de
cluster", réglables en écarts-types σ) pour visualiser les regroupements automatiques
détectés sur chaque carte.

Toutes ces coordonnées (UMAP) et regroupements (HDBSCAN) sont **précalculés une fois pour
toutes** — l'application ne fait aucun calcul de ce type au runtime, elle ne fait que lire
et filtrer (voir [Architecture](#architecture--précalcul-vs-runtime)).

---

## Architecture — précalcul vs runtime

Le projet est volontairement scindé en deux temps, pour que l'application reste légère et
rapide à charger :

```
┌──────────────────────────────┐         ┌────────────────────────────────┐
│  precompute_drug_vectors.py  │  ───▶   │     data/reports.json          │
│  (hors-ligne, à relancer     │         │     + data/sentences.csv       │
│   après toute mise à jour    │         │     (coordonnées UMAP et       │
│   des données sources)       │         │      clusters HDBSCAN figés)   │
└──────────────────────────────┘         └────────────────────────────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │      app.py      │
                                                │  (Streamlit, lit │
                                                │   uniquement)    │
                                                └──────────────────┘
```

- **`precompute_drug_vectors.py`** charge `sentences.csv` + `reports.json`, calcule les
  vecteurs d'effets (proportionnel/TF-IDF) et les embeddings sémantiques moyens
  (`mean_embeddings_by_substance.csv`), les projette en 2D via **UMAP**, les regroupe via
  **HDBSCAN**, puis réécrit le résultat dans `sentences.csv` (colonnes `drug`,
  `chem_family`, `pharm_family`) et `reports.json` (clé `_drug_vectors`).
- **`app.py`** ne dépend que de Streamlit/pandas/plotly/numpy : aucune dépendance ML
  lourde (UMAP, HDBSCAN, scikit-learn) n'est nécessaire pour faire tourner l'application
  elle-même.

Les paramètres UMAP et HDBSCAN sont **volontairement séparés en deux blocs** dans
`precompute_drug_vectors.py`, car ce sont deux types de données différents (profil
d'effets catégoriel à faible dimension vs embeddings sémantiques denses à 768
dimensions) :

```python
UMAP_PARAMS_EFFECTS     # + HDBSCAN_PARAMS_EFFECTS   → profils d'effets (prop & TF-IDF)
UMAP_PARAMS_SEMANTIC    # + HDBSCAN_PARAMS_SEMANTIC  → embeddings sémantiques moyens
```

Ajuster ces constantes puis relancer le script est le seul moyen de changer le rendu des
cartes — l'application ne propose aucun réglage UMAP/HDBSCAN en interface, par choix.

---

## Structure du projet

```
.
├── app.py                          # Application Streamlit (runtime)
├── precompute_drug_vectors.py      # Script de précalcul (UMAP + HDBSCAN), à lancer hors-ligne
├── drug_taxonomy.py                # Double classification des substances (chimique / pharmacologique)
├── requirements.txt                # Dépendances de l'application (légères)
├── requirements-precompute.txt     # Dépendances additionnelles du script de précalcul
└── data/
    ├── sentences.csv                       # 1 ligne = 1 phrase annotée (effets top-1/2/3 + scores)
    ├── reports.json                        # 1 entrée = 1 récit (métadonnées + clé _drug_vectors)
    └── mean_embeddings_by_substance.csv    # Embeddings moyens (768 dim) par substance
```

### `data/sentences.csv`

| Colonne | Description |
|---|---|
| `titre` | Identifiant du récit (clé vers `reports.json`) |
| `idx` | Position de la phrase dans le récit |
| `sous_cle` | Phase du trip : `onset` / `peak` / `offset` |
| `sentence` | Texte de la phrase |
| `top1_label`, `top2_label`, `top3_label` | Les 3 effets les plus probables prédits par le classifieur |
| `top1_score`, `top2_score`, `top3_score` | Scores de confiance associés |
| `drug` | Substance du récit (persistée par le précalcul) |
| `chem_family`, `pharm_family` | Classification chimique/pharmacologique (persistée par le précalcul) |

### `data/reports.json`

Dictionnaire `{titre_du_récit: {drug, domain, url, conclusion, logs, …}}`, plus une clé de
premier niveau `_drug_vectors` ajoutée par le précalcul :

```json
"_drug_vectors": {
  "_meta": { "min_phrases": 25, "umap_params_effects": {...}, "hdbscan_params_effects": {...} },
  "drugs": {
    "LSD": {
      "n_phrases": 546,
      "top_effets": ["delusion", "internal hallucination", "transformation"],
      "umap_prop": [0.0, 0.0], "umap_tfidf": [0.0, 0.0], "umap_semantic": [0.0, 0.0],
      "cluster_prop": 1, "cluster_tfidf": 1, "cluster_semantic": 0
    }
  }
}
```

(`cluster_* == -1` signifie "bruit", c'est-à-dire non rattaché à un cluster par HDBSCAN.)

### `data/mean_embeddings_by_substance.csv`

Une ligne par substance : colonne `substance` + 768 colonnes `embedding_0`…`embedding_767`
(embedding moyen des phrases décrivant ses effets). Les substances absentes de ce fichier,
ou sous le seuil `MIN_PHRASES`, n'apparaissent pas sur la carte sémantique.

---

## Installation

```bash
git clone <url-du-repo>
cd <repo>
python -m venv .venv && source .venv/bin/activate   # ou l'équivalent Windows
pip install -r requirements.txt
streamlit run app.py
```

Les données de `data/` sont déjà précalculées (`_drug_vectors` présent dans
`reports.json`) : aucune étape supplémentaire n'est nécessaire pour lancer l'application
telle quelle.

## Relancer le précalcul

Nécessaire uniquement si tu modifies les données sources (`sentences.csv`,
`reports.json` avant précalcul, ou `mean_embeddings_by_substance.csv`) ou les paramètres
UMAP/HDBSCAN :

```bash
pip install -r requirements-precompute.txt
python precompute_drug_vectors.py
```

Le script réécrit `data/sentences.csv` et `data/reports.json` sur place.

## Personnaliser la taxonomie

`drug_taxonomy.py` contient deux dictionnaires codés en dur (`CHEMICAL_FAMILY_MAP` et
`PHARMACOLOGICAL_FAMILY_MAP`) faisant correspondre une substance à une famille, plus leurs
palettes de couleurs associées. Toute substance absente de ces dictionnaires tombe par
défaut dans `"Autres / divers"`. **Ce regroupement est indicatif et à but de
visualisation — ce n'est pas une référence pharmacologique.**

---

## Dépendances

**Application (`requirements.txt`)** — légères, suffisent à faire tourner `app.py` :

| Paquet | Usage |
|---|---|
| `streamlit` | Interface web |
| `pandas` | Manipulation des données |
| `plotly` | Graphiques interactifs |
| `numpy` | Calcul des ellipses de cluster (algèbre linéaire) |

**Précalcul uniquement (`requirements-precompute.txt`)** — installés en plus, jamais
requis par l'application elle-même :

| Paquet | Usage |
|---|---|
| `umap-learn` | Projection 2D des vecteurs d'effets et des embeddings sémantiques |
| `hdbscan` | Clustering automatique après UMAP |
| `scikit-learn` | Pondération TF-IDF du profil d'effets |

---

## Limites connues

- Le seuil `MIN_PHRASES = 25` exclut des deux cartes les substances trop peu représentées
  (vecteur trop bruité) — elles restent visibles dans les autres onglets.
- Le matching entre `mean_embeddings_by_substance.csv` et les noms de substances internes
  est sensible à la casse/orthographe ; une table d'alias (`EMBEDDING_NAME_ALIASES`) corrige
  les écarts connus, à étendre si de nouvelles substances sont ajoutées.
- La classification chimique/pharmacologique (`drug_taxonomy.py`) est une simplification
  manuelle, pas une nomenclature officielle.
- Les clusters HDBSCAN sont calculés sur les coordonnées UMAP déjà réduites (et non sur les
  vecteurs d'origine à plus haute dimension), par choix de cohérence avec ce qui est affiché.

## Données & licence

Les récits sont issus de [Effect Index](https://www.effectindex.com/) et
[PsychonautWiki](https://psychonautwiki.org/). Le contenu de cette application est mis à
disposition selon les termes de la licence
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
