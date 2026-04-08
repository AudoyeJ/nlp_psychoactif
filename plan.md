# 🧭 Feuille de route Data Science : Effets subjectifs des drogues

## Objectif : Les expériences subjectives liées aux drogues peuven etre representées dans un espace sémantique, et que leur structure reflete des effets psychologiques cohérents.

## 1. Préparation & compréhension des données
**Objectif :** Nettoyer et structurer le corpus pour traitement.

- **Tâches :**
  - Collecter tous les rapports d’usager en anglais.
  - Segmenter chaque rapport en phrases.
  - Créer un petit échantillon annoté de phrases hors sujet (0/1) — 500 phrases.
  - Vérifier la distribution des drogues dans l’échantillon.

- **Outils :**
  - Python : `nltk` / `spacy`
  - Pandas

- **Livrables :**
  - Corpus segmenté en phrases avec colonnes : `texte`, `drogue`, `hors_sujet`.

---

## 2. Filtrage des phrases hors sujet
**Objectif :** Éliminer le bruit pour ne garder que les phrases pertinentes.

- **Tâches :**
  - Utiliser Qwen pour étiqueter l’échantillon.
  - Entraîner un classifieur traditionnel (SVM, Logistic Regression, Random Forest).
  - Appliquer le classifieur sur tout le corpus.
  - Vérification manuelle sur un petit échantillon.

- **Outils :**
  - `scikit-learn` pour SVM / RandomForest
  - `transformers` pour features textuelles

- **Livrables :**
  - Corpus filtré prêt pour embeddings.

---

## 3. Embeddings sémantiques
**Objectif :** Représenter chaque phrase dans un espace vectoriel sémantique.

- **Tâches :**
  - Choisir un modèle Transformer : `all-MiniLM-L6-v2`, `MPNet`, etc.
  - Calculer embeddings pour chaque phrase pertinente.

- **Outils :**
  - `sentence-transformers`
  - GPU recommandé si corpus > 10k phrases

- **Livrables :**
  - Matrice embeddings (`num_phrases x embedding_dim`)

---

## 4. Réduction de dimension
**Objectif :** Passer à un espace 2D/3D interprétable et exploitable pour clustering.

- **Tâches :**
  - UMAP pour réduire dimensions (5–10 pour clustering, 2D pour visualisation)
  - Sauvegarder embeddings réduits

- **Outils :**
  - `umap-learn`

- **Livrables :**
  - Embeddings réduits
  - Graphiques UMAP

---

## 5. Clustering global
**Objectif :** Identifier des groupes de phrases correspondant à des effets subjectifs communs.

- **Tâches :**
  - HDBSCAN sur embeddings réduits
  - Analyse des clusters : taille, densité, thèmes dominants
  - Associer chaque phrase à son cluster

- **Outils :**
  - `hdbscan`
  - `matplotlib` / `plotly`

- **Livrables :**
  - Cluster global de phrases
  - Histogramme des clusters par drogue
  - CSV : phrase, drogue, cluster global

---

## 6. Clustering interne par drogue
**Objectif :** Capturer les variations internes à chaque drogue.

- **Tâches :**
  - Pour chaque drogue, clustering interne (HDBSCAN ou KMeans)
  - Obtenir plusieurs centroides / clusters par drogue
  - Calculer distribution et densité interne

- **Livrables :**
  - Centroides par drogue
  - Histogramme interne
  - DataFrame : drogue, cluster interne, centroïde

---

## 7. Analyse inter-drogues
**Objectif :** Comprendre similarités entre drogues via leurs effets subjectifs.

- **Tâches :**
  - Comparer centroides avec distance cosinus ou Euclidienne
  - Matrice de similarité drogue x drogue
  - Visualisation : heatmap ou graph réseau
  - Option : UMAP sur tous les centroides

- **Livrables :**
  - Matrice de similarité
  - Graphe ou carte des drogues
  - Clusters de drogues selon effets similaires

---

## 8. Documentation & communication
- Rapport avec méthodologie et visualisations
- Dashboard interactif (Streamlit / Plotly Dash)
- Explication des clusters avec exemples de phrases

---

**Conseil méthodologique :**
- Commencer par un sous-ensemble de corpus pour valider le pipeline
- Modulariser : filtrage → embeddings → UMAP → clustering
- Utiliser embeddings + clustering interne avant comparaison inter-drogues