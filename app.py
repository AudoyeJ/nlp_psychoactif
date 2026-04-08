import streamlit as st
import numpy as np
import json
from umap import UMAP
import hdbscan
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🔬 UMAP + HDBSCAN Explorer (2D / 3D)")

# =========================
# Chargement des données
# =========================
@st.cache_data
def load_data():
    with open("mean_embeddings_substance_index.json", encoding="utf-8") as f:
        substance_index = json.load(f)
    mean_embeddings = np.load("mean_embeddings_with_index.npy")
    
    # Ordonner par index
    sorted_items = sorted(substance_index.items(), key=lambda item: item[1])
    valid_substances = [item[0] for item in sorted_items]
    ordered_indices = [item[1] for item in sorted_items]
    mean_embeddings_ordered = mean_embeddings[ordered_indices, :]
    
    return valid_substances, mean_embeddings_ordered

valid_substances, mean_embeddings = load_data()

# =========================
# Sidebar paramètres
# =========================
st.sidebar.header("⚙️ Paramètres")

# Mode 2D/3D
mode = st.sidebar.radio("Mode UMAP", ["2D", "3D"])

# UMAP
n_neighbors = st.sidebar.slider("UMAP n_neighbors", 2, 50, 6)
min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 1.0, 0.3)
metric_umap = st.sidebar.selectbox("UMAP metric", ["cosine", "euclidean"])

# HDBSCAN
min_cluster_size = st.sidebar.slider("HDBSCAN min_cluster_size", 2, 10, 2)
min_samples = st.sidebar.slider("HDBSCAN min_samples", 1, 10, 2)

# Taille des points
point_size = st.sidebar.slider("Taille des points", 2, 15, 6)

# =========================
# UMAP
# =========================
n_components = 3 if mode == "3D" else 2
umap_model = UMAP(
    n_components=n_components,
    random_state=42,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric=metric_umap
)
vectors_umap = umap_model.fit_transform(mean_embeddings)

# =========================
# HDBSCAN (sur embeddings originaux)
# =========================
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric='euclidean'
)
clusters = clusterer.fit_predict(mean_embeddings)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
st.write(f"### 🔢 Nombre de clusters : {n_clusters}")

# =========================
# Préparer DataFrame pour Plotly
# =========================
df = pd.DataFrame({
    "substance": valid_substances,
    "cluster": clusters.astype(str)
})
df["cluster"] = df["cluster"].replace("-1", "Bruit")  # renommer le bruit
if mode == "2D":
    df["x"], df["y"] = vectors_umap[:, 0], vectors_umap[:, 1]
else:
    df["x"], df["y"], df["z"] = vectors_umap[:, 0], vectors_umap[:, 1], vectors_umap[:, 2]

# =========================
# Plot interactif
# =========================
if mode == "2D":
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_name="substance",
        title="UMAP 2D + HDBSCAN",
        width=1000,
        height=700
    )
    fig.update_traces(marker=dict(size=point_size))
else:
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="cluster",
        hover_name="substance",
        title="UMAP 3D + HDBSCAN",
        width=1000,
        height=700
    )
    fig.update_traces(marker=dict(size=point_size))

st.plotly_chart(fig, use_container_width=True)

# =========================
# Contenu des clusters
# =========================
st.write("### 📦 Contenu des clusters")
for cluster_id in sorted(df["cluster"].unique()):
    if cluster_id == "Bruit":
        continue
    members = df[df["cluster"] == cluster_id]["substance"].tolist()
    st.write(f"**Cluster {cluster_id} ({len(members)} éléments):** {', '.join(members)}")

# =========================
# Télécharger CSV
# =========================
csv = df[["substance", "cluster"]].to_csv(index=False)
st.download_button(
    label="📥 Télécharger clusters",
    data=csv,
    file_name="clusters.csv",
    mime="text/csv"
)