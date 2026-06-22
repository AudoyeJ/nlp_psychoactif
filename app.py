"""
ÉTAT MODIFIÉ — Cartographie des effets psychédéliques par phrase
Application Streamlit de data visualisation.

Données : sentence_effect_final.json (NLP top-3 effets prédits par phrase,
extrait de récits de trip sur effectindex.com et psychonautwiki.org).
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

# ----------------------------------------------------------------------------
# Palette — les effets sont regroupés par famille phénoménologique :
# violet/magenta = perception visuelle, ambre = cognition, rose/vert = émotion,
# bleu/teal = corps. "None" = gris neutre (aucun effet net détecté).
# ----------------------------------------------------------------------------
FAMILIES = {
    "Perception": ["internal hallucination", "external hallucination", "geometry",
                    "colour enhancement", "transformation", "drifting"],
    "Cognition": ["delusion", "memory suppression", "time distortion", "cognitive euphoria"],
    "Émotion": ["emotion enhancement", "anxiety", "anxiety suppression",
                "unity and interconnectedness"],
    "Corps": ["stimulation", "sedation", "physical euphoria", "nausea",
              "tactile enhancement", "spontaneous tactile sensation",
              "motor control loss", "auditory hallucination"],
}

EFFECT_COLORS = {
    "None": "#4A4458",
    # Perception — violets / magentas
    "internal hallucination": "#A855F7",
    "external hallucination": "#D946EF",
    "geometry": "#C084FC",
    "colour enhancement": "#E879F9",
    "transformation": "#9333EA",
    "drifting": "#7C3AED",
    # Cognition — ambres / ors
    "delusion": "#F59E0B",
    "memory suppression": "#D97706",
    "time distortion": "#FBBF24",
    "cognitive euphoria": "#FCD34D",
    # Émotion — rose / rouge / vert
    "emotion enhancement": "#FB7185",
    "anxiety": "#EF4444",
    "anxiety suppression": "#34D399",
    "unity and interconnectedness": "#10B981",
    # Corps — bleus / teals
    "stimulation": "#22D3EE",
    "sedation": "#0E7490",
    "physical euphoria": "#2DD4BF",
    "nausea": "#65A30D",
    "tactile enhancement": "#38BDF8",
    "spontaneous tactile sensation": "#7DD3FC",
    "motor control loss": "#0891B2",
    "auditory hallucination": "#67E8F9",
}

ORDERED_LABELS = [lab for fam in FAMILIES.values() for lab in fam]

# ----------------------------------------------------------------------------
# Classification chimique indicative des substances — voir drug_taxonomy.py
# (module partagé avec precompute_drug_vectors.py pour éviter toute divergence).
# ----------------------------------------------------------------------------
from drug_taxonomy import (
    CHEMICAL_FAMILY_MAP, CHEM_COLORS, chem_family, get_chem_color,
    PHARMACOLOGICAL_FAMILY_MAP, PHARM_COLORS, pharm_family, get_pharm_color,
)

BG = "#0D0B14" #fdf6e3
PANEL = "#171320"
GRID = "#2D2640"
TEXT = "#F1EDFA"
MUTED = "#9089A8"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)

# ----------------------------------------------------------------------------
# Chargement des données
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des phrases annotées…")
def load_sentences() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sentences.csv")
    df["sous_cle"] = pd.Categorical(df["sous_cle"], categories=["onset", "peak", "offset"], ordered=True)
    return df


@st.cache_data(show_spinner="Chargement des récits…")
def load_reports() -> tuple:
    """Retourne (reports, drug_vectors). drug_vectors == {} si le précalcul
    (precompute_drug_vectors.py) n'a pas encore été exécuté sur ces données."""
    with open(DATA_DIR / "reports.json", encoding="utf-8") as f:
        data = json.load(f)
    drug_vectors_block = data.pop("_drug_vectors", None) or {}
    drug_vectors = drug_vectors_block.get("drugs", {})
    return data, drug_vectors


def color_for(label: str) -> str:
    return EFFECT_COLORS.get(label, "#6B6480")


_CLUSTER_PALETTE = [
    "#A855F7", "#22D3EE", "#F59E0B", "#FB7185", "#34D399", "#EF4444",
    "#60A5FA", "#65A30D", "#E879F9", "#FBBF24", "#10B981", "#F472B6",
]
_NOISE_COLOR = "#3F3A4D"  # gris sombre, distinct des couleurs de cluster


def cluster_palette_and_labels(cluster_ids: pd.Series):
    """Couleur + libellé stables pour une série de labels HDBSCAN (-1 = bruit).
    Clusters triés par taille décroissante pour donner les couleurs les plus
    distinctes aux groupes les plus importants."""
    counts = cluster_ids[cluster_ids != -1].value_counts()
    ordered = counts.index.tolist()
    colors = {cid: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i, cid in enumerate(ordered)}
    colors[-1] = _NOISE_COLOR
    labels = {cid: f"Cluster {i}" for i, cid in enumerate(ordered)}
    labels[-1] = "Bruit (non clusterisé)"
    return colors, labels


# ----------------------------------------------------------------------------
# Fonctions utilitaires pour les ellipses de cluster
# ----------------------------------------------------------------------------

def _compute_ellipse_params(points_x, points_y, n_std=2.0):
    """Calcule les paramètres d'une ellipse de confiance à *n_std* écarts-types
    autour du centre de gravité d'un nuage de points 2D.

    Retourne un dict {cx, cy, w, h, angle} ou None si < 2 points.
    """
    n = len(points_x)
    if n < 2:
        return None

    cx, cy = float(np.mean(points_x)), float(np.mean(points_y))

    if n == 2:
        dx = float(points_x.iloc[1]) - float(points_x.iloc[0])
        dy = float(points_y.iloc[1]) - float(points_y.iloc[0])
        dist = np.hypot(dx, dy)
        return dict(cx=cx, cy=cy, w=dist + 0.4, h=dist + 0.4, angle=0.0)

    cov = np.cov(
        np.asarray(points_x, dtype=float),
        np.asarray(points_y, dtype=float),
    )
    cov += np.eye(2) * 1e-8  # régularisation pour éviter matrice singulière

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    w = 2.0 * n_std * np.sqrt(eigenvalues[0])
    h = 2.0 * n_std * np.sqrt(eigenvalues[1])

    return dict(cx=cx, cy=cy, w=w, h=h, angle=angle)


def _ellipse_trace(cx, cy, w, h, angle, color, label="", n_points=0,
                   fill_opacity=0.10, line_opacity=0.35):
    """Retourne un trace Plotly (Scatter rempli) dessinant une ellipse."""
    t = np.linspace(0, 2 * np.pi, 80)
    x0 = w / 2 * np.cos(t)
    y0 = h / 2 * np.sin(t)
    rad = np.radians(angle)
    x = x0 * np.cos(rad) - y0 * np.sin(rad) + cx
    y = x0 * np.sin(rad) + y0 * np.cos(rad) + cy
    # Fermer le polygone
    x, y = np.append(x, x[0]), np.append(y, y[0])

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    return go.Scatter(
        x=x, y=y,
        mode="lines",
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},{fill_opacity})",
        line=dict(color=f"rgba({r},{g},{b},{line_opacity})", width=1.5),
        hovertemplate=f"<b>{label}</b><br>{n_points} substance(s)<extra></extra>",
        showlegend=False,
        name=label,
    )


PHASE_LABELS_FR = {"onset": "montée", "peak": "pic", "offset": "descente"}


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3 {{
        font-family: 'Fraunces', serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}
    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: {MUTED};
        margin-bottom: -0.4rem;
    }}
    .hero-title {{
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 500;
        font-size: 2.6rem;
        line-height: 1.05;
        background: linear-gradient(90deg, #E9D5FF 0%, #A855F7 45%, #67E8F9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .hero-sub {{
        color: {MUTED};
        max-width: 700px;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }}
    .kpi-card {{
        background: {PANEL};
        border: 1px solid {GRID};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    .kpi-num {{
        font-family: 'Fraunces', serif;
        font-size: 1.9rem;
        color: {TEXT};
        line-height: 1;
    }}
    .kpi-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {MUTED};
    }}
    .pill {{
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid {GRID};
        color: {MUTED};
        margin-right: 6px;
    }}
    .legend-fam {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {MUTED};
        margin-right: 18px;
    }}
    .legend-dot {{
        display:inline-block; width:9px; height:9px; border-radius:50%;
        margin-right:5px; vertical-align:middle;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Fraunces', serif;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    blockquote {{
        border-left: 2px solid #A855F7;
        padding-left: 1rem;
        color: {MUTED};
        font-style: italic;
    }}
    </style>
    """, unsafe_allow_html=True)


def family_legend():
    fam_color_sample = {"Perception": "#A855F7", "Cognition": "#FBBF24",
                         "Émotion": "#FB7185", "Corps": "#22D3EE"}
    html = '<div style="margin-top:6px;">'
    for fam, col in fam_color_sample.items():
        html += f'<span class="legend-fam"><span class="legend-dot" style="background:{col};"></span>{fam}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def strip_figure(sub: pd.DataFrame, height: int = 110, show_axis: bool = False) -> go.Figure:
    """Bande continue colorée = une phrase = une barre, couleur = effet dominant."""
    sub = sub.sort_values("idx")
    colors = [color_for(l) for l in sub["top1_label"]]
    hover = [
        f"<b>{PHASE_LABELS_FR.get(p,p)}</b> · {lab} ({sc:.0%})<br>{s[:140]}"
        for p, lab, sc, s in zip(sub["sous_cle"], sub["top1_label"], sub["top1_score"], sub["sentence"])
    ]
    fig = go.Figure(go.Bar(
        x=sub["idx"], y=[1] * len(sub), marker_color=colors, width=1.02,
        hovertext=hover, hoverinfo="text",
    ))
    # bandes de phase
    for phase, color in zip(["onset", "peak", "offset"], ["#A855F7", "#67E8F9", "#34D399"]):
        rows = sub[sub["sous_cle"] == phase]
        if len(rows):
            fig.add_vrect(x0=rows["idx"].min() - 0.5, x1=rows["idx"].max() + 0.5,
                           line_width=0, fillcolor=color, opacity=0.06, layer="below")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig.update_layout(height=height, bargap=0, showlegend=False,
                       margin=dict(l=4, r=4, t=4, b=4 if not show_axis else 24))
    fig.update_xaxes(visible=show_axis, showgrid=False)
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


# ============================================================================
st.set_page_config(page_title="NLP psychoactifs · Cartographie des effets",
                    page_icon="◈", layout="wide")
inject_css()

sentences = load_sentences()
reports, drug_vectors = load_reports()
if "drug" not in sentences.columns:
    # rétro-compatibilité : sentences.csv n'a pas encore été enrichi par
    # precompute_drug_vectors.py — on retombe sur le merge à la volée.
    sentences["drug"] = sentences["titre"].map(lambda t: reports.get(t, {}).get("drug", "Inconnu"))
if "chem_family" not in sentences.columns:
    sentences["chem_family"] = sentences["drug"].map(chem_family)

st.markdown('<div class="eyebrow">NLP · récits de trip · effectindex & psychonautwiki</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-title">Récits des états modifiés de conscience</div>', unsafe_allow_html=True)
st.markdown(
    '''
    <div class="hero-sub">
    Cette application présente le résultat d'un travail exploratoire en traitement automatique des langues (NLP) pour cartographier les effets phénoménologiques provoqués par des substances psychoactives, à partir de récits d'usagers.

    Les récits sont issus des sites <a href="https://www.effectindex.com/" target="_blank">Effect Index</a> & <a href="https://psychonautwiki.org/" target="_blank">PsychonautWiki</a>.
    </div>
    ''',
    unsafe_allow_html=True
)

st.write("")

tab_overview, tab_carto, tab_phases, tab_substances, tab_explore, tab_search, tab_sources = st.tabs(
    ["◆ Vue d'ensemble", "◆ Cartographie", "◆ Effets & phases", "◆ Substances",
     "◆ Explorateur de récits", "◆ Recherche", "◆ Sources"]
)

# ----------------------------------------------------------------------------
# TAB 1 — VUE D'ENSEMBLE
# ----------------------------------------------------------------------------
with tab_overview:
    n_reports = len(reports)
    n_sentences = len(sentences)
    n_labels = sentences["top1_label"].nunique()
    n_drugs = sentences["drug"].nunique()
    avg_score = sentences["top1_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in zip(
        [c1, c2, c3, c4],
        [f"{n_reports}", f"{n_sentences:,}".replace(",", " "), f"{n_labels}",
         f"{n_drugs}"],
        ["Récits analysés", "Phrases annotées", "Effets distincts",
         "Substances distinctes"]
    ):
        col.markdown(f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
                      f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    st.write("")

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Effets les plus fréquents")
        counts = sentences["top1_label"].value_counts()
        counts = counts.head(15).sort_values()
        fig = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker_color=[color_for(l) for l in counts.index],
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(title="Phrases où l'effet est prédit en 1ère position")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("##### Sources")
        dom_counts = sentences["titre"].map(lambda t: reports.get(t, {}).get("domain", "?"))
        dc = dom_counts.value_counts()
        fig3 = go.Figure(go.Bar(
            x=dc.values, y=[d.replace("www.", "") for d in dc.index], orientation="h",
            marker_color=["#A855F7", "#67E8F9"],
        ))
        fig3.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
        fig3.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# TAB 2 — CARTOGRAPHIE (sous-onglets)
# ----------------------------------------------------------------------------
with tab_carto:
    tab_umap, tab_semantic = st.tabs(
        ["Clusterization via effet subjectif", "Clusterization via vecteur sémantique"]
    )

    # -- Sous-onglet : Clusterization via effet subjectif --
    with tab_umap:
        if not drug_vectors:
            st.warning(
                "Les coordonnées UMAP ne sont pas encore présentes dans `reports.json`.\n\n"
                "Lance une fois le script de précalcul (nécessite `umap-learn` et `scikit-learn`, "
                "uniquement pour ce script — pas pour l'appli) :\n"
                "```\npython precompute_drug_vectors.py\n```\n"
                "Il enrichit `data/sentences.csv` (colonnes `drug`, `chem_family`, `pharm_family`) et "
                "`data/reports.json` "
                "(clé `_drug_vectors`) une bonne fois pour toutes."
            )
        else:
            st.markdown("##### Carte UMAP des substances selon distribution des effets par substance : proximité = profil d'effets similaire")
            st.caption(
                "Chaque substance est représentée par un vecteur construit à partir de la distribution "
                "de ses effets dominants (top-1) sur l'ensemble de ses phrases, puis projetée en 2D par "
                "UMAP (distance cosinus). Deux substances proches sur la carte ont un profil d'effets "
                "phénoménologiques statistiquement proche dans ce corpus."
            )

            cu1, cu2, cu3 = st.columns([1.5, 1.1, 1])
            with cu1:
                mode_label = st.radio(
                    "Mode de vectorisation",
                    ["Proportionnel (fréquence relative)", "TF-IDF (pondéré par spécificité)"],
                    help=(
                        "Proportionnel : part de chaque effet parmi les phrases de la substance — "
                        "deux substances rapportant beaucoup d'effets très communs (ex. anxiété) "
                        "se retrouvent proches même si le reste de leur profil diffère.\n\n"
                        "TF-IDF : même comptage de départ, mais repondéré pour atténuer les effets "
                        "présents chez presque toutes les substances et faire ressortir ceux qui "
                        "distinguent réellement une substance des autres."
                    ),
                )
                coord_key = "umap_tfidf" if mode_label.startswith("TF-IDF") else "umap_prop"
                cluster_key = "cluster_tfidf" if mode_label.startswith("TF-IDF") else "cluster_prop"
            with cu2:
                class_label = st.radio(
                    "Colorer par",
                    ["Classe chimique", "Classe pharmacologique"],
                    help=(
                        "Classe chimique : famille moléculaire (ex. lysergamides, tryptamines…).\n\n"
                        "Classe pharmacologique : effet/mécanisme principal (ex. hallucinogène, "
                        "dissociatif…)."
                    ),
                )
            with cu3:
                all_sizes = [v["n_phrases"] for v in drug_vectors.values()]
                min_sentences = st.slider(
                    "Min. N phrases (affichage)", min(all_sizes),
                    max(all_sizes), min(all_sizes), step=5,
                    help=(
                        "Filtre d'affichage uniquement : les coordonnées ont déjà été calculées pour "
                        "toutes les substances retenues lors du précalcul, ce curseur ne fait que "
                        "masquer les points les moins représentés, sans rien recalculer."
                    ),
                )

            # --- Contrôles des aires de cluster ---
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                show_clusters = st.checkbox(
                    "Aires de cluster", value=True, key="umap_show_clusters",
                    help="Affiche des ellipses de confiance autour des groupes HDBSCAN "
                         "(calculés lors du précalcul). Les points en bruit n'ont pas d'ellipse.",
                )
            with cc2:
                n_std = st.slider(
                    "Extension des ellipses (σ)", 1.0, 3.5, 2.0, 0.25,
                    key="umap_n_std",
                    help="Nombre d'écarts-types pour la taille de l'ellipse. "
                         "2 σ ≈ 95 % des points d'une distribution normale bidimensionnelle.",
                )

            rows = [
                {
                    "drug": drug,
                    "n_phrases": v["n_phrases"],
                    "top_effets": ", ".join(v["top_effets"]),
                    "x": v[coord_key][0],
                    "y": v[coord_key][1],
                    "cluster_id": v.get(cluster_key, -1),
                }
                for drug, v in drug_vectors.items()
                if v["n_phrases"] >= min_sentences
            ]
            emb = pd.DataFrame(rows)

            if len(emb) < 4:
                st.info("Pas assez de substances au-dessus de ce seuil — baisse le minimum de phrases.")
            else:
                # Palette pour les ellipses (toujours calculée, même si masquées)
                cluster_colors_map, cluster_labels_map = cluster_palette_and_labels(emb["cluster_id"])

                if class_label == "Classe pharmacologique":
                    emb["famille"] = emb["drug"].map(pharm_family)
                    palette = PHARM_COLORS
                else:
                    emb["famille"] = emb["drug"].map(chem_family)
                    palette = CHEM_COLORS

                max_n = max(emb["n_phrases"].max(), 1)
                fig = go.Figure()

                # --- Ellipses de cluster (tracées EN PREMIER = arrière-plan) ---
                if show_clusters:
                    for cid in sorted(emb["cluster_id"].unique()):
                        if cid == -1:
                            continue
                        sub = emb[emb["cluster_id"] == cid]
                        params = _compute_ellipse_params(sub["x"], sub["y"], n_std=n_std)
                        if params is None:
                            continue
                        fig.add_trace(_ellipse_trace(
                            params["cx"], params["cy"],
                            params["w"], params["h"], params["angle"],
                            color=cluster_colors_map[cid],
                            label=cluster_labels_map[cid],
                            n_points=len(sub),
                        ))

                # --- Points (tracés APRÈS = premier plan) ---
                for fam, color in palette.items():
                    sub = emb[emb["famille"] == fam]
                    if not len(sub):
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub["x"], y=sub["y"], mode="markers+text",
                        text=sub["drug"], textposition="top center",
                        textfont=dict(size=9, color=MUTED),
                        name=fam,
                        marker=dict(
                            size=8 + 14 * (sub["n_phrases"] / max_n) ** 0.5,
                            color=color, line=dict(width=1, color=BG),
                        ),
                        customdata=sub[["n_phrases", "top_effets"]],
                        hovertemplate=(
                            "<b>%{text}</b> · " + fam +
                            "<br>%{customdata[0]} phrases"
                            "<br>effets dominants : %{customdata[1]}<extra></extra>"
                        ),
                    ))
                fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
                fig.update_layout(height=640, legend=dict(orientation="h", y=-0.12),
                                   margin=dict(l=10, r=10, t=10, b=10))
                fig.update_xaxes(title="UMAP 1", showticklabels=False, zeroline=False)
                fig.update_yaxes(title="UMAP 2", showticklabels=False, zeroline=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                st.caption(
                    f"{len(emb)} substances affichées · mode = {mode_label.split(' (')[0].lower()} · "
                    f"couleur = {class_label.lower()} · taille des points ∝ nombre de phrases."
                    + (f" · {show_clusters and len(set(emb['cluster_id']) - {-1})} ellipses de cluster "
                       f"({n_std}σ)" if show_clusters else "")
                )

                with st.expander("📐 Coordonnées (table)"):
                    st.dataframe(
                        emb.rename(columns={"famille": class_label})[
                            ["drug", class_label, "n_phrases", "top_effets", "x", "y"]
                        ].sort_values("n_phrases", ascending=False),
                        use_container_width=True, hide_index=True,
                    )

    # -- Sous-onglet : Clusterization via vecteur sémantique --
    with tab_semantic:
        semantic_drugs = {d: v for d, v in drug_vectors.items() if "umap_semantic" in v}

        if not semantic_drugs:
            st.warning(
                "Aucune coordonnée sémantique trouvée dans `reports.json`.\n\n"
                "Vérifie que `data/mean_embeddings_by_substance.csv` est présent, puis relance :\n"
                "```\npython precompute_drug_vectors.py\n```"
            )
        else:
            st.markdown("##### Carte UMAP des substances selon vecteurs moyens par substance : proximité = profil d'effets similaire")
            st.caption(
                "Chaque substance est représentée par la **moyenne des embeddings** de toutes "
                "les phrases de récits décrivant ses effets phénoménologiques — un vecteur dense issu d'un "
                "modèle de langage, pas un comptage de labels. Deux substances proches sur cette carte "
                "sont décrites avec un langage et des nuances sémantiquement proches dans ce corpus. "
        
            )

            cs1, cs2 = st.columns([1.3, 1])
            with cs1:
                class_label_sem = st.radio(
                    "Colorer par", ["Classe chimique", "Classe pharmacologique"],
                    key="semantic_class_label",
                    help=(
                        "Classe chimique : famille moléculaire (ex. lysergamides, tryptamines…).\n\n"
                        "Classe pharmacologique : effet/mécanisme principal (ex. hallucinogène, "
                        "dissociatif…)."
                    ),
                )
            with cs2:
                sizes_sem = [v["n_phrases"] for v in semantic_drugs.values()]
                min_sentences_sem = st.slider(
                    "Min. N phrases (affichage)", min(sizes_sem), max(sizes_sem), min(sizes_sem),
                    step=5, key="semantic_min_sentences",
                    help="Filtre d'affichage uniquement, aucun recalcul.",
                )

            # --- Contrôles des aires de cluster ---
            cc1s, cc2s = st.columns([1, 2])
            with cc1s:
                show_clusters_sem = st.checkbox(
                    "Aires de cluster", value=True, key="sem_show_clusters",
                    help="Affiche des ellipses de confiance autour des groupes HDBSCAN "
                         "(calculés lors du précalcul). Les points en bruit n'ont pas d'ellipse.",
                )
            with cc2s:
                n_std_sem = st.slider(
                    "Extension des ellipses (σ)", 1.0, 3.5, 2.0, 0.25,
                    key="sem_n_std",
                    help="Nombre d'écarts-types pour la taille de l'ellipse. "
                         "2 σ ≈ 95 % des points d'une distribution normale bidimensionnelle.",
                )

            rows_sem = [
                {
                    "drug": drug,
                    "n_phrases": v["n_phrases"],
                    "top_effets": ", ".join(v.get("top_effets", [])),
                    "x": v["umap_semantic"][0],
                    "y": v["umap_semantic"][1],
                    "cluster_id": v.get("cluster_semantic", -1),
                }
                for drug, v in semantic_drugs.items()
                if v["n_phrases"] >= min_sentences_sem
            ]
            emb_sem = pd.DataFrame(rows_sem)

            if len(emb_sem) < 4:
                st.info("Pas assez de substances au-dessus de ce seuil — baisse le minimum de phrases.")
            else:
                # Palette pour les ellipses
                cluster_colors_sem, cluster_labels_sem = cluster_palette_and_labels(emb_sem["cluster_id"])

                if class_label_sem == "Classe pharmacologique":
                    emb_sem["famille"] = emb_sem["drug"].map(pharm_family)
                    palette_sem = PHARM_COLORS
                else:
                    emb_sem["famille"] = emb_sem["drug"].map(chem_family)
                    palette_sem = CHEM_COLORS

                max_n_sem = max(emb_sem["n_phrases"].max(), 1)
                fig_sem = go.Figure()

                # --- Ellipses de cluster (arrière-plan) ---
                if show_clusters_sem:
                    for cid in sorted(emb_sem["cluster_id"].unique()):
                        if cid == -1:
                            continue
                        sub = emb_sem[emb_sem["cluster_id"] == cid]
                        params = _compute_ellipse_params(sub["x"], sub["y"], n_std=n_std_sem)
                        if params is None:
                            continue
                        fig_sem.add_trace(_ellipse_trace(
                            params["cx"], params["cy"],
                            params["w"], params["h"], params["angle"],
                            color=cluster_colors_sem[cid],
                            label=cluster_labels_sem[cid],
                            n_points=len(sub),
                        ))

                # --- Points (premier plan) ---
                for fam, color in palette_sem.items():
                    sub = emb_sem[emb_sem["famille"] == fam]
                    if not len(sub):
                        continue
                    fig_sem.add_trace(go.Scatter(
                        x=sub["x"], y=sub["y"], mode="markers+text",
                        text=sub["drug"], textposition="top center",
                        textfont=dict(size=9, color=MUTED),
                        name=fam,
                        marker=dict(
                            size=8 + 14 * (sub["n_phrases"] / max_n_sem) ** 0.5,
                            color=color, line=dict(width=1, color=BG),
                        ),
                        customdata=sub[["n_phrases", "top_effets"]],
                        hovertemplate=(
                            "<b>%{text}</b> · " + fam +
                            "<br>%{customdata[0]} phrases"
                            "<br>effets dominants : %{customdata[1]}<extra></extra>"
                        ),
                    ))
                fig_sem.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
                fig_sem.update_layout(height=640, legend=dict(orientation="h", y=-0.12),
                                       margin=dict(l=10, r=10, t=10, b=10))
                fig_sem.update_xaxes(title="UMAP 1", showticklabels=False, zeroline=False)
                fig_sem.update_yaxes(title="UMAP 2", showticklabels=False, zeroline=False)
                st.plotly_chart(fig_sem, use_container_width=True, config={"displayModeBar": False})

                st.caption(
                    f"{len(emb_sem)} substances affichées · couleur = {class_label_sem.lower()} · "
                    f"taille des points ∝ nombre de phrases."
                    + (f" · {show_clusters_sem and len(set(emb_sem['cluster_id']) - {-1})} ellipses de "
                       f"cluster ({n_std_sem}σ)" if show_clusters_sem else "")
                )

                with st.expander("📐 Coordonnées (table)"):
                    st.dataframe(
                        emb_sem.rename(columns={"famille": class_label_sem})[
                            ["drug", class_label_sem, "n_phrases", "top_effets", "x", "y"]
                        ].sort_values("n_phrases", ascending=False),
                        use_container_width=True, hide_index=True,
                    )

# ----------------------------------------------------------------------------
# TAB 3 — EFFETS & PHASES
# ----------------------------------------------------------------------------
with tab_phases:
    st.markdown("##### Comment le profil d'effets se déforme entre montée, pic et descente")
    colA, colB = st.columns([1, 1])
    with colA:
        normalize = st.radio("Affichage", ["% au sein de chaque phase", "Nombre de phrases"],
                              horizontal=True)
    with colB:
        incl_none = st.checkbox("Inclure « None »", value=False, key="ph_none")

    work = sentences.copy()
    if not incl_none:
        work = work[work["top1_label"] != "None"]

    pivot = pd.crosstab(work["sous_cle"], work["top1_label"])
    pivot = pivot.reindex(["onset", "peak", "offset"])
    if normalize.startswith("%"):
        pivot_disp = pivot.div(pivot.sum(axis=1), axis=0) * 100
        x_title = "Part des phrases (%)"
    else:
        pivot_disp = pivot
        x_title = "Nombre de phrases"

    cols_present = [l for l in ORDERED_LABELS if l in pivot_disp.columns]
    if incl_none and "None" in pivot_disp.columns:
        cols_present = ["None"] + cols_present

    fig = go.Figure()
    for lab in cols_present:
        fig.add_trace(go.Bar(
            y=[PHASE_LABELS_FR[p] for p in pivot_disp.index],
            x=pivot_disp[lab], name=lab, orientation="h",
            marker_color=color_for(lab),
            hovertemplate=f"<b>{lab}</b><br>%{{x:.1f}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", **PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig.update_layout(height=320, legend=dict(orientation="h", y=-0.25),
                       margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(title=x_title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    family_legend()

    st.write("")
    st.markdown("##### Détail — table effet × phase (%)")
    heat = pd.crosstab(work["top1_label"], work["sous_cle"], normalize="columns") * 100
    heat = heat.reindex(["onset", "peak", "offset"], axis=1)
    heat = heat.loc[[l for l in ORDERED_LABELS if l in heat.index]]
    fig_h = px.imshow(
        heat, color_continuous_scale=["#171320", "#7C3AED", "#E879F9"],
        labels=dict(color="% de la phase"),
        aspect="auto",
    )
    fig_h.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig_h.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))
    fig_h.update_xaxes(title=None, side="top",
                       tickvals=[0, 1, 2], ticktext=["montée", "pic", "descente"])
    fig_h.update_yaxes(title=None)
    st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# TAB 4 — SUBSTANCES
# ----------------------------------------------------------------------------
with tab_substances:
    st.markdown("##### Les substances les plus représentées dans le corpus")
    metric_choice = st.radio("Classer par", ["Nombre de phrases", "Nombre de récits"],
                              horizontal=True, key="sub_metric")
    top_n = st.slider("Top N substances", 5, 40, 20, key="sub_topn")

    if metric_choice == "Nombre de phrases":
        rank = sentences["drug"].value_counts()
    else:
        rank = sentences.drop_duplicates("titre")["drug"].value_counts()
    rank = rank.head(top_n).sort_values()
    bar_colors = [get_chem_color(chem_family(d)) for d in rank.index]
    fig = go.Figure(go.Bar(x=rank.values, y=rank.index, orientation="h", marker_color=bar_colors))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig.update_layout(height=max(320, top_n * 22), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    fam_html = '<div style="margin-top:-6px;">'
    for fam, col in CHEM_COLORS.items():
        fam_html += f'<span class="legend-fam"><span class="legend-dot" style="background:{col};"></span>{fam}</span>'
    fam_html += '</div>'
    st.markdown(fam_html, unsafe_allow_html=True)
    st.caption("Classification chimique indicative (regroupement approximatif à but visuel, "
               "pas une référence pharmacologique).")

    st.write("")
    st.markdown("##### Répartition par famille chimique")
    fam_counts = sentences.drop_duplicates("titre")["chem_family"].value_counts()
    fig_fam = go.Figure(go.Pie(
        labels=fam_counts.index, values=fam_counts.values, hole=0.5,
        marker_colors=[get_chem_color(f) for f in fam_counts.index],
        textfont=dict(family="IBM Plex Mono, monospace", color=TEXT),
    ))
    fig_fam.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig_fam.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_fam, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.markdown("##### Comparer le profil d'effets de plusieurs substances")
    top_drugs_default = sentences["drug"].value_counts().head(20).index.tolist()
    chosen = st.multiselect(
        "Substances à comparer (2 à 6 recommandé)", top_drugs_default,
        default=top_drugs_default[:3]
    )
    if len(chosen) >= 1:
        work = sentences[sentences["drug"].isin(chosen) & (sentences["top1_label"] != "None")]
        cross = pd.crosstab(work["top1_label"], work["drug"], normalize="columns") * 100
        top_labels = work["top1_label"].value_counts().head(12).index
        cross = cross.loc[[l for l in top_labels if l in cross.index]]
        fig2 = go.Figure()
        palette_cycle = ["#A855F7", "#67E8F9", "#FBBF24", "#FB7185", "#34D399", "#F97316"]
        for i, d in enumerate(chosen):
            if d in cross.columns:
                fig2.add_trace(go.Bar(
                    y=cross.index, x=cross[d], name=d, orientation="h",
                    marker_color=palette_cycle[i % len(palette_cycle)],
                ))
        fig2.update_layout(barmode="group", **PLOTLY_TEMPLATE["layout"].to_plotly_json())
        fig2.update_layout(height=460, legend=dict(orientation="h", y=-0.15),
                            margin=dict(l=10, r=10, t=10, b=10))
        fig2.update_xaxes(title="% des phrases (top-1) de la substance, hors « None »")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Choisis au moins une substance pour afficher la comparaison.")

    st.write("")
    st.markdown("##### Annuaire des substances")
    dir_df = (
        sentences.drop_duplicates("titre")
        .groupby("drug")
        .agg(récits=("titre", "nunique"))
        .join(sentences.groupby("drug").size().rename("phrases"))
        .reset_index()
        .rename(columns={"drug": "substance"})
        .sort_values("phrases", ascending=False)
    )
    dir_df["famille"] = dir_df["substance"].map(chem_family)
    st.dataframe(dir_df, use_container_width=True, height=320, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 5 — EXPLORATEUR DE RÉCITS
# ----------------------------------------------------------------------------
with tab_explore:
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        drug_options = ["Toutes"] + sorted({r["drug"] for r in reports.values()})
        drug_filter = st.selectbox("Filtrer par substance", drug_options, key="explore_drug_filter")
    titres_all = sorted(reports.keys())
    if drug_filter != "Toutes":
        titres = sorted([t for t in titres_all if reports[t]["drug"] == drug_filter])
    else:
        titres = titres_all

    c1, c2 = st.columns([3, 1])
    with c1:
        if "explore_titre" not in st.session_state or st.session_state["explore_titre"] not in titres:
            st.session_state["explore_titre"] = titres[0]
        titre = st.selectbox("Choisir un récit", titres, key="explore_titre")
    with c2:
        st.write("")
        if st.button("🎲 Récit au hasard"):
            import random
            titre = random.choice(titres)
            st.session_state["explore_titre"] = titre
            st.rerun()

    meta = reports[titre]
    sub = sentences[sentences["titre"] == titre].sort_values("idx")

    domain_clean = meta["domain"].replace("www.", "")
    st.markdown(
        f'<span class="pill">{domain_clean}</span>'
        f'<span class="pill">{meta["drug"]}</span>'
        f'<span class="pill">{len(sub)} phrases</span>'
        f'<a href="{meta["url"]}" target="_blank" style="color:#67E8F9; font-family: IBM Plex Mono, monospace; '
        f'font-size:0.75rem; text-decoration:none;">↗ voir la source originale</a>',
        unsafe_allow_html=True)

    st.write("")
    st.markdown("###### Lecture du trip — chaque barre = une phrase, couleur = effet dominant")
    st.plotly_chart(strip_figure(sub, height=110, show_axis=True), use_container_width=True,
                     config={"displayModeBar": False})
    family_legend()

    phase_filter = st.multiselect("Filtrer la table par phase", ["onset", "peak", "offset"],
                                   default=["onset", "peak", "offset"],
                                   format_func=lambda p: PHASE_LABELS_FR[p])
    table = sub[sub["sous_cle"].isin(phase_filter)][
        ["idx", "sous_cle", "sentence", "top1_label", "top1_score", "top2_label", "top3_label"]
    ].rename(columns={"idx": "#", "sous_cle": "phase", "sentence": "phrase",
                       "top1_label": "effet #1", "top1_score": "score #1",
                       "top2_label": "effet #2", "top3_label": "effet #3"})
    table["phase"] = table["phase"].map(PHASE_LABELS_FR)
    st.dataframe(table, use_container_width=True, height=300, hide_index=True,
                 column_config={"score #1": st.column_config.ProgressColumn(
                     "score #1", min_value=0, max_value=1, format="%.0f%%")})

    cexp1, cexp2 = st.columns(2)
    with cexp1:
        with st.expander("📄 Texte intégral du récit"):
            st.write(meta["logs"])
    with cexp2:
        with st.expander("🧭 Conclusion de l'auteur"):
            st.write(meta["conclusion"] or "_Pas de conclusion fournie pour ce récit._")

# ----------------------------------------------------------------------------
# TAB 6 — RECHERCHE
# ----------------------------------------------------------------------------
with tab_search:
    st.markdown("##### Rechercher des phrases par effet, phase, substance ou mot-clé")
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        all_labels = sorted(sentences["top1_label"].dropna().unique())
        sel_labels = st.multiselect("Effet (top-1)", all_labels, default=[])
    with f2:
        sel_phases = st.multiselect("Phase", ["onset", "peak", "offset"],
                                     format_func=lambda p: PHASE_LABELS_FR[p])
    with f3:
        all_drugs = sorted(sentences["drug"].dropna().unique())
        sel_drugs = st.multiselect("Substance", all_drugs, default=[])
    with f4:
        min_score = st.slider("Score min.", 0.0, 1.0, 0.3, 0.05)

    keyword = st.text_input("Mot-clé dans la phrase (optionnel)")

    res = sentences.copy()
    if sel_labels:
        res = res[res["top1_label"].isin(sel_labels)]
    if sel_phases:
        res = res[res["sous_cle"].isin(sel_phases)]
    if sel_drugs:
        res = res[res["drug"].isin(sel_drugs)]
    res = res[res["top1_score"] >= min_score]
    if keyword.strip():
        res = res[res["sentence"].str.contains(keyword, case=False, na=False)]

    st.caption(f"**{len(res)}** phrases correspondantes sur {len(sentences)}.")
    res_disp = res[["titre", "drug", "sous_cle", "sentence", "top1_label", "top1_score"]].copy()
    res_disp["url"] = res_disp["titre"].map(lambda t: reports.get(t, {}).get("url", ""))
    res_disp["sous_cle"] = res_disp["sous_cle"].map(PHASE_LABELS_FR)
    res_disp = res_disp.rename(columns={"titre": "récit", "drug": "substance", "sous_cle": "phase",
                                         "sentence": "phrase",
                                         "top1_label": "effet", "top1_score": "score"})
    st.dataframe(res_disp.sort_values("score", ascending=False), use_container_width=True,
                 height=420, hide_index=True,
                 column_config={
                     "score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.0f%%"),
                     "url": st.column_config.LinkColumn("source"),
                 })
    st.download_button("⬇ Télécharger ces résultats (CSV)",
                        res_disp.to_csv(index=False).encode("utf-8"),
                        file_name="phrases_filtrees.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# TAB 7 — SOURCES
# ----------------------------------------------------------------------------
with tab_sources:
    st.markdown("##### effectindex.com vs psychonautwiki.org")
    sentences_dom = sentences.copy()
    sentences_dom["domain"] = sentences_dom["titre"].map(
        lambda t: reports.get(t, {}).get("domain", "?").replace("www.", ""))

    doms = sentences_dom["domain"].unique().tolist()
    cols = st.columns(len(doms))
    for col, d in zip(cols, doms):
        sub_d = sentences_dom[sentences_dom["domain"] == d]
        n_rep = sub_d["titre"].nunique()
        col.markdown(f'<div class="kpi-card"><div class="kpi-num">{n_rep}</div>'
                      f'<div class="kpi-label">récits · {d}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Profil d'effets comparé (top-12, hors « None »)")
    work = sentences_dom[sentences_dom["top1_label"] != "None"]
    cross = pd.crosstab(work["top1_label"], work["domain"], normalize="columns") * 100
    top_labels = work["top1_label"].value_counts().head(12).index
    cross = cross.loc[[l for l in top_labels if l in cross.index]]

    fig = go.Figure()
    palette_dom = {doms[0]: "#A855F7", doms[1] if len(doms) > 1 else "x": "#67E8F9"}
    for d in doms:
        if d in cross.columns:
            fig.add_trace(go.Bar(
                y=cross.index, x=cross[d], name=d, orientation="h",
                marker_color=palette_dom.get(d, "#67E8F9"),
            ))
    fig.update_layout(barmode="group", **PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig.update_layout(height=460, legend=dict(orientation="h", y=-0.15),
                       margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(title="% des phrases (top-1) de la source")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.markdown("##### Confiance moyenne du classifieur par source")
    score_dom = sentences_dom.groupby("domain")["top1_score"].mean().sort_values()
    fig2 = go.Figure(go.Bar(
        x=score_dom.values, y=score_dom.index, orientation="h",
        marker_color=["#A855F7", "#67E8F9"][:len(score_dom)],
        text=[f"{v:.0%}" for v in score_dom.values], textposition="outside",
    ))
    fig2.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    fig2.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
    fig2.update_xaxes(range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

st.write("")
st.markdown(
    f'<div style="text-align:center; color:{MUTED}; font-family:\'IBM Plex Mono\',monospace; '
    f'font-size:0.7rem; margin-top:2rem;"> '
    f'classification NLP de trip reports de psychoactifs, sources <a href="https://effectindex.com" target="_blank" style="color:{MUTED}; text-decoration:underline;">effectindex.com</a> & <a href="https://psychonautwiki.org" target="_blank" style="color:{MUTED}; text-decoration:underline;">psychonautwiki.org</a></div>',
    unsafe_allow_html=True)

st.markdown(
    f'<div style="text-align:center; margin-top:1.2rem; padding-bottom:1.5rem;">'
    f'<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">'
    f'<img alt="Licence Creative Commons" style="border-width:0; opacity:0.7;" '
    f'src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br>'
    f'<span style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:{MUTED};">'
    f'Cette œuvre est mise à disposition selon les termes de la '
    f'<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" '
    f'style="color:{MUTED}; text-decoration:underline;">Licence Attribution-NonCommercial-ShareAlike 4.0 International</a>.'
    f'</span></div>',
    unsafe_allow_html=True)
