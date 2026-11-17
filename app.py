"""Interface Streamlit : visualisation du dataset + prédiction NORMAL / PNEUMONIA.

Lancer en local :  streamlit run app.py
"""

import random
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from history import clear_history, get_history, save_prediction
from report import build_pdf

st.set_page_config(page_title="Souffle", layout="wide")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VALID_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")
CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_COLORS = {"NORMAL": "#2f6f5e", "PNEUMONIA": "#b4412f"}  # mêmes couleurs que le PDF

# Ordre de priorité : le modèle allégé (utilisé en ligne), puis les modèles du notebook.
MODEL_CANDIDATES = [
    "model/pneumonia_model.keras",
    "best_model_vgg19_ft.h5",
    "best_model_vgg19.h5",
]

SAMPLES_DIR = Path("samples")  # quelques radios incluses pour la démo en ligne

# Statistiques du dataset, utilisées quand le dataset complet (2,4 Go) n'est pas présent.
SAVED_STATS = {
    "train": {"NORMAL": 1341, "PNEUMONIA": 3875},
    "val": {"NORMAL": 8, "PNEUMONIA": 8},
    "test": {"NORMAL": 234, "PNEUMONIA": 390},
}

# Résultats obtenus sur le jeu de test (624 images) dans le notebook.
TEST_METRICS = [
    ("Accuracy", "92,5 %"),
    ("Recall", "98,5 %"),
    ("Precision", "90,4 %"),
    ("F1-score", "94,2 %"),
    ("AUC", "0,975"),
]

# ---------------------------------------------------------------------------
# Feuille de style
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink: #1b2430;
        --muted: #5b6675;
        --line: #dfe3e8;
        --surface: #ffffff;
        --accent: #1f4e79;
        --normal: #2f6f5e;
        --normal-bg: #eaf3f0;
        --pneumo: #b4412f;
        --pneumo-bg: #f9ecea;
    }
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px;}

    /* En-tête */
    .topbar {display: flex; align-items: center; justify-content: space-between; gap: 1rem;
             padding-bottom: 1rem; border-bottom: 1px solid var(--line); margin-bottom: 0.4rem;}
    .brand {display: flex; align-items: center; gap: 0.8rem;}
    .brand-mark {width: 38px; height: 38px; border-radius: 6px; background: var(--accent); color: #fff;
                 display: grid; place-items: center; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em;}
    .brand-name {font-weight: 600; font-size: 1.15rem; color: var(--ink); line-height: 1.2;}
    .brand-sub {font-size: 0.85rem; color: var(--muted);}
    .tag {font-size: 0.75rem; color: var(--muted); border: 1px solid var(--line); border-radius: 999px;
          padding: 0.2rem 0.7rem; white-space: nowrap;}

    /* Titres de section */
    .section-label {font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
                    color: var(--muted); margin: 1.4rem 0 0.6rem;}

    /* Carte de résultat */
    .result {background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 1.2rem 1.3rem;}
    .result-head {display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap;}
    .verdict {font-size: 1.6rem; font-weight: 700; letter-spacing: 0.01em;}
    .verdict.normal {color: var(--normal);} .verdict.pneumo {color: var(--pneumo);}
    .badge {font-size: 0.8rem; font-weight: 500; padding: 0.25rem 0.6rem; border-radius: 4px;}
    .badge.normal {background: var(--normal-bg); color: var(--normal);}
    .badge.pneumo {background: var(--pneumo-bg); color: var(--pneumo);}
    .result-text {color: var(--muted); font-size: 0.92rem; margin: 0.4rem 0 1.1rem;}

    /* Jauge de probabilité avec repère du seuil */
    .gauge-label {display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted); margin-bottom: 0.35rem;}
    .gauge {position: relative; height: 10px; background: #edf0f3; border-radius: 5px;}
    .gauge-fill {position: absolute; inset: 0 auto 0 0; border-radius: 5px;}
    .gauge-mark {position: absolute; top: -5px; width: 2px; height: 20px; background: var(--ink);}
    .gauge-scale {display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); margin-top: 0.4rem;
                  font-family: "IBM Plex Mono", monospace;}

    /* Lignes de détails */
    .facts {margin-top: 1.1rem; border-top: 1px solid var(--line);}
    .fact {display: flex; justify-content: space-between; gap: 1rem; padding: 0.55rem 0; border-bottom: 1px solid var(--line);
           font-size: 0.9rem;}
    .fact span:first-child {color: var(--muted);}
    .fact span:last-child {font-family: "IBM Plex Mono", monospace; color: var(--ink); text-align: right; word-break: break-all;}

    /* Chiffres clés */
    .stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.6rem;}
    .stat {background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 0.8rem 1rem;}
    .stat-value {font-size: 1.45rem; font-weight: 600; color: var(--ink); font-family: "IBM Plex Mono", monospace;}
    .stat-name {font-size: 0.8rem; color: var(--muted);}

    .empty {border: 1px dashed var(--line); border-radius: 8px; padding: 2.2rem 1rem; text-align: center;
            color: var(--muted); font-size: 0.92rem; background: var(--surface);}
    .note {font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem;}
    .footer {margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: 0.78rem;
             color: var(--muted); display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;}

    /* Téléphone */
    @media (max-width: 640px) {
        .block-container {padding: 1rem 0.9rem 2.5rem;}
        .brand-sub, .tag {display: none;}
        .verdict {font-size: 1.35rem;}
        .stat-value {font-size: 1.2rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def dataset_base_dir():
    """Retourne le dossier du dataset complet, ou None s'il n'est pas présent."""
    for candidate in [Path("chest_xray/chest_xray"), Path("chest_xray")]:
        if (candidate / "train").exists() and (candidate / "test").exists():
            return candidate
    return None


def list_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]


@st.cache_data
def compute_stats(base_dir: str):
    base = Path(base_dir)
    return {
        split: {cls: len(list_images(base / split / cls)) for cls in CLASSES}
        for split in ["train", "val", "test"]
    }


@st.cache_resource
def load_model():
    """Charge le premier modèle disponible (une seule fois grâce au cache)."""
    for model_path in MODEL_CANDIDATES:
        if Path(model_path).exists():
            return tf.keras.models.load_model(model_path, compile=False), model_path
    return None, None


def preprocess_image(img: Image.Image):
    # Même traitement qu'à l'entraînement : RGB, 224x224, pixels entre 0 et 1.
    img = img.convert("RGB").resize((224, 224), Image.NEAREST)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img: Image.Image, threshold: float = 0.5):
    prob = float(model.predict(preprocess_image(img), verbose=0)[0][0])
    label = "PNEUMONIA" if prob > threshold else "NORMAL"
    confidence = prob if prob > threshold else 1 - prob
    return label, confidence, prob


def looks_like_xray(img: Image.Image) -> bool:
    """Vérification simple : une radio est en niveaux de gris.
    Si l'image est très colorée, ce n'est probablement pas une radio."""
    arr = np.array(img.convert("RGB").resize((128, 128))).astype("float32")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (np.abs(r - g) + np.abs(g - b)).mean() < 20


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def section(title: str):
    html(f'<div class="section-label">{title}</div>')


def stats_row(items):
    cells = "".join(
        f'<div class="stat"><div class="stat-value">{value}</div><div class="stat-name">{name}</div></div>'
        for name, value in items
    )
    html(f'<div class="stats">{cells}</div>')


def result_card(label, confidence, prob, threshold, filename, model_name):
    css = "pneumo" if label == "PNEUMONIA" else "normal"
    text = (
        "Le modèle détecte des opacités compatibles avec une pneumonie."
        if label == "PNEUMONIA"
        else "Le modèle ne détecte pas de signe de pneumonie."
    )
    html(f"""
    <div class="result">
      <div class="result-head">
        <div class="verdict {css}">{label}</div>
        <span class="badge {css}">Confiance {confidence * 100:.1f} %</span>
      </div>
      <div class="result-text">{text}</div>
      <div class="gauge-label"><span>Probabilité de pneumonie</span><span>{prob:.3f}</span></div>
      <div class="gauge">
        <div class="gauge-fill" style="width:{prob * 100:.1f}%; background:{CLASS_COLORS[label]};"></div>
        <div class="gauge-mark" style="left:calc({threshold * 100:.1f}% - 1px);" title="Seuil"></div>
      </div>
      <div class="gauge-scale"><span>0</span><span>seuil {threshold:.2f}</span><span>1</span></div>
      <div class="facts">
        <div class="fact"><span>Fichier</span><span>{filename}</span></div>
        <div class="fact"><span>Modèle</span><span>VGG19 fine-tuné</span></div>
        <div class="fact"><span>Entrée</span><span>224 × 224 px</span></div>
      </div>
    </div>
    """)


# ---------------------------------------------------------------------------
# En-tête + navigation
# ---------------------------------------------------------------------------
html("""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">S</div>
    <div>
      <div class="brand-name">Souffle</div>
      <div class="brand-sub">Lecture assistée des radiographies pulmonaires</div>
    </div>
  </div>
  <span class="tag">Prototype académique</span>
</div>
""")

page = st.segmented_control(
    "Navigation",
    ["Analyse", "Vue d'ensemble", "Dataset", "Historique"],
    default="Analyse",
    label_visibility="collapsed",
) or "Analyse"

base_dir = dataset_base_dir()
model, model_name = load_model()


# ---------------------------------------------------------------------------
# Page : Analyse
# ---------------------------------------------------------------------------
if page == "Analyse":
    if model is None:
        st.error("Aucun modèle trouvé. Entraînez le modèle dans le notebook.")
        st.stop()

    col_input, col_result = st.columns([5, 6], gap="large")

    with col_input:
        section("Image")
        source = st.segmented_control(
            "Source", ["Importer", "Photo", "Exemple"], default="Importer", label_visibility="collapsed"
        ) or "Importer"

        img, filename = None, None
        if source == "Importer":
            uploaded = st.file_uploader(
                "Radiographie thoracique", type=[e.strip(".") for e in VALID_EXTS], label_visibility="collapsed"
            )
            if uploaded:
                img, filename = Image.open(uploaded), uploaded.name
        elif source == "Photo":
            shot = st.camera_input("Photographier la radiographie", label_visibility="collapsed")
            html('<div class="note">Photographiez la radio bien à plat, sans reflet, cadrée sur le thorax.</div>')
            if shot:
                img, filename = Image.open(shot), "photo.jpg"
        else:
            samples = sorted(list_images(SAMPLES_DIR / "NORMAL") + list_images(SAMPLES_DIR / "PNEUMONIA"))
            if samples:
                choice = st.selectbox(
                    "Radiographie d'exemple",
                    samples,
                    format_func=lambda p: f"{p.parent.name.capitalize()}  ·  {p.name}",
                )
                img, filename = Image.open(choice), choice.name

        threshold = st.slider(
            "Seuil de décision",
            0.1, 0.9, 0.5, 0.05,
            help="Au-dessus du seuil, l'image est classée PNEUMONIA. Un seuil bas détecte plus de cas, "
                 "au prix de davantage de fausses alertes.",
        )

        if img is not None:
            st.image(img, width="stretch")

    with col_result:
        section("Résultat")
        if img is None:
            html('<div class="empty">Importez une radiographie, prenez une photo ou choisissez un exemple '
                 'pour lancer l\'analyse.</div>')
        else:
            label, confidence, prob = predict(model, img, threshold)
            result_card(label, confidence, prob, threshold, filename, model_name)

            if not looks_like_xray(img):
                st.warning("L'image est en couleur : il ne s'agit probablement pas d'une radiographie. "
                           "Le résultat n'est pas interprétable.")

            st.write("")
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "Télécharger le compte-rendu",
                    data=build_pdf(img, filename, label, confidence, prob, threshold, model_name),
                    file_name=f"compte_rendu_{Path(filename).stem}.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                )
            with b2:
                if st.button("Enregistrer dans l'historique", width="stretch"):
                    save_prediction(filename, label, prob, threshold, model_name)
                    st.toast("Analyse enregistrée")

        html('<div class="note">Outil d\'aide à la décision développé dans un cadre pédagogique. '
             'Il ne remplace pas l\'interprétation d\'un médecin.</div>')


# ---------------------------------------------------------------------------
# Page : Vue d'ensemble
# ---------------------------------------------------------------------------
elif page == "Vue d'ensemble":
    stats = compute_stats(str(base_dir)) if base_dir else SAVED_STATS

    section("Performances sur le jeu de test (624 images)")
    stats_row(TEST_METRICS)

    section("Répartition du dataset")
    stats_row([
        ("Entraînement", sum(stats["train"].values())),
        ("Validation", sum(stats["val"].values())),
        ("Test", sum(stats["test"].values())),
    ])

    df_long = pd.DataFrame(
        [{"Split": split, "Classe": cls, "Images": n} for split, row in stats.items() for cls, n in row.items()]
    )
    chart = (
        alt.Chart(df_long)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Split:N", sort=["train", "val", "test"], title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="Classe:N",
            y=alt.Y("Images:Q", title=None),
            color=alt.Color(
                "Classe:N",
                scale=alt.Scale(domain=CLASSES, range=[CLASS_COLORS[c] for c in CLASSES]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Split", "Classe", "Images"],
        )
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#eceff3", domain=False, tickColor="#dfe3e8")
    )
    st.altair_chart(chart, width="stretch")

    ratio = stats["train"]["PNEUMONIA"] / max(stats["train"]["NORMAL"], 1)
    html(f'<div class="note">Le jeu d\'entraînement contient {ratio:.1f} fois plus d\'images PNEUMONIA que NORMAL. '
         'Ce déséquilibre est compensé par des poids de classe pendant l\'entraînement.'
         + ('' if base_dir else ' Statistiques enregistrées (dataset complet non présent).') + '</div>')

    section("Évaluation détaillée")
    if Path("evaluation_results.png").exists():
        with st.expander("Matrice de confusion et courbe ROC"):
            st.image("evaluation_results.png", width="stretch")
    if Path("model_comparison.png").exists():
        with st.expander("Comparaison VGG19, ResNet50 et MobileNetV2"):
            st.image("model_comparison.png", width="stretch")
            html('<div class="note">Comparaison rapide sur 5 époques, sans les deux phases d\'entraînement '
                 'du modèle final.</div>')


# ---------------------------------------------------------------------------
# Page : Dataset
# ---------------------------------------------------------------------------
elif page == "Dataset":
    c1, c2, c3 = st.columns([1, 1, 2])
    if base_dir:
        split = c1.selectbox("Split", ["train", "val", "test"])
        folder_root = base_dir / split
    else:
        c1.selectbox("Split", ["exemples"], disabled=True)
        folder_root = SAMPLES_DIR
    cls = c2.selectbox("Classe", CLASSES)
    n_images = c3.slider("Nombre d'images", 3, 12, 6, step=3)

    images = list_images(folder_root / cls)
    if not images:
        html('<div class="empty">Aucune image dans ce dossier.</div>')
    else:
        picks = random.sample(images, k=min(n_images, len(images)))
        cols = st.columns(3)
        for i, path in enumerate(picks):
            with cols[i % 3]:
                st.image(str(path), caption=path.name, width="stretch")


# ---------------------------------------------------------------------------
# Page : Historique
# ---------------------------------------------------------------------------
elif page == "Historique":
    rows = get_history()
    if not rows:
        html('<div class="empty">Aucune analyse enregistrée.<br>'
             'Après une analyse, cliquez sur « Enregistrer dans l\'historique ».</div>')
    else:
        df = pd.DataFrame(rows)
        section("Résumé")
        stats_row([
            ("Analyses", len(df)),
            ("PNEUMONIA", int((df["Résultat"] == "PNEUMONIA").sum())),
            ("NORMAL", int((df["Résultat"] == "NORMAL").sum())),
        ])

        section("Analyses enregistrées")
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Probabilité pneumonie": st.column_config.ProgressColumn(
                    "Probabilité pneumonie", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "Seuil": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Exporter en CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="historique_analyses.csv",
                mime="text/csv",
                width="stretch",
            )
        with c2:
            with st.popover("Vider l'historique", width="stretch"):
                st.write("Toutes les analyses enregistrées seront supprimées.")
                if st.button("Confirmer la suppression", type="primary"):
                    clear_history()
                    st.rerun()


# ---------------------------------------------------------------------------
# Pied de page
# ---------------------------------------------------------------------------
dataset_info = "dataset complet" if base_dir else "mode démonstration"
html(f"""
<div class="footer">
  <span>Modèle : {Path(model_name).name if model_name else "aucun"} · {dataset_info}</span>
  <span>Ne remplace pas un avis médical.</span>
</div>
""")
