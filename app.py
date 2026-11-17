"""Interface Streamlit : visualisation du dataset + prédiction NORMAL / PNEUMONIA.

Lancer en local :  streamlit run app.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from history import clear_history, get_history, save_prediction
from report import build_pdf

st.set_page_config(
    page_title="Pneumonia X-Ray Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VALID_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")
CLASSES = ["NORMAL", "PNEUMONIA"]

# Ordre de priorité : le modèle allégé (utilisé en ligne), puis les modèles du notebook.
MODEL_CANDIDATES = [
    "model/pneumonia_model.keras",
    "best_model_vgg19_ft.h5",
    "best_model_vgg19.h5",
]

SAMPLES_DIR = Path("samples")  # quelques radios incluses pour la démo en ligne

# Statistiques du dataset, utilisées quand le dataset complet (2,4 Go) n'est pas présent
# (cas de la version en ligne).
SAVED_STATS = {
    "train": {"NORMAL": 1341, "PNEUMONIA": 3875},
    "val": {"NORMAL": 8, "PNEUMONIA": 8},
    "test": {"NORMAL": 234, "PNEUMONIA": 390},
}

# Résultats obtenus sur le jeu de test (624 images) dans le notebook.
TEST_METRICS = {
    "Accuracy": "92.5 %",
    "Recall": "98.5 %",
    "Precision": "90.4 %",
    "F1-score": "94.2 %",
    "AUC": "0.975",
}

# ---------------------------------------------------------------------------
# Style responsive (ordinateur + mobile)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .app-title {font-size: 2rem; font-weight: 800; margin-bottom: 0.2rem;}
    .app-subtitle {opacity: 0.7; margin-bottom: 1rem;}

    /* Téléphone : moins de marges, titres plus petits */
    @media (max-width: 640px) {
        .block-container {padding: 1rem 0.8rem 3rem 0.8rem;}
        .app-title {font-size: 1.45rem;}
        h3 {font-size: 1.15rem;}
        [data-testid="stMetricValue"] {font-size: 1.4rem;}
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
    color_level = (np.abs(r - g) + np.abs(g - b)).mean()
    return color_level < 20


# ---------------------------------------------------------------------------
# En-tête + navigation (en haut de page pour être visible aussi sur mobile)
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">🩺 Détection de pneumonie</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Classification de radiographies thoraciques par Deep Learning (VGG19)</div>',
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    ["🔍 Prédiction", "📊 Vue globale", "🖼️ Dataset", "🕓 Historique"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

base_dir = dataset_base_dir()
model, model_name = load_model()

with st.sidebar:
    st.header("Informations")
    st.write(f"**Modèle :** `{model_name or 'aucun'}`")
    st.write(f"**Dataset :** `{base_dir}`" if base_dir else "**Dataset :** non présent (mode démo)")
    st.caption("Outil pédagogique, ne remplace pas un avis médical.")


# ---------------------------------------------------------------------------
# Page : Prédiction
# ---------------------------------------------------------------------------
if page == "🔍 Prédiction":
    if model is None:
        st.error("Aucun modèle trouvé. Entraînez le modèle dans le notebook.")
        st.stop()

    source = st.radio(
        "Source de l'image",
        ["📁 Importer", "📷 Appareil photo", "🧪 Exemple"],
        horizontal=True,
    )

    img, filename = None, None
    if source == "📁 Importer":
        uploaded = st.file_uploader("Choisir une radio thoracique", type=[e.strip(".") for e in VALID_EXTS])
        if uploaded:
            img, filename = Image.open(uploaded), uploaded.name
    elif source == "📷 Appareil photo":
        st.caption("Sur téléphone, prenez la radio en photo bien à plat et sans reflet.")
        shot = st.camera_input("Prendre une photo")
        if shot:
            img, filename = Image.open(shot), "photo_camera.jpg"
    else:
        samples = sorted(list_images(SAMPLES_DIR / "NORMAL") + list_images(SAMPLES_DIR / "PNEUMONIA"))
        if samples:
            choice = st.selectbox(
                "Choisir une radio d'exemple (jeu de test)",
                samples,
                format_func=lambda p: f"{p.parent.name} — {p.name}",
            )
            img, filename = Image.open(choice), choice.name
        else:
            st.warning("Aucune image d'exemple dans le dossier samples/.")

    threshold = st.slider(
        "Seuil de décision",
        0.1, 0.9, 0.5, 0.05,
        help="Au-dessus du seuil, l'image est classée PNEUMONIA. "
             "Un seuil bas détecte plus de malades mais donne plus de fausses alertes.",
    )

    if img is not None:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(img, caption=filename, width="stretch")

        with col_result:
            label, confidence, prob = predict(model, img, threshold)

            st.markdown("### Résultat")
            if label == "PNEUMONIA":
                st.error(f"**{label}** — signes de pneumonie détectés")
            else:
                st.success(f"**{label}** — pas de signe de pneumonie détecté")

            st.metric("Confiance", f"{confidence * 100:.1f} %")
            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption(f"Probabilité brute de pneumonie : {prob:.4f} (seuil {threshold:.2f})")

            if not looks_like_xray(img):
                st.warning("⚠️ Cette image est très colorée : ce n'est peut-être pas une radio thoracique. "
                           "Le résultat n'a alors aucun sens.")

            if st.button("💾 Enregistrer dans l'historique", width="stretch"):
                save_prediction(filename, label, prob, threshold, model_name)
                st.toast("Prédiction enregistrée ✅")

            st.download_button(
                "📄 Télécharger le compte-rendu PDF",
                data=build_pdf(img, filename, label, confidence, prob, threshold, model_name),
                file_name=f"compte_rendu_{Path(filename).stem}.pdf",
                mime="application/pdf",
                width="stretch",
            )

        st.info("ℹ️ Outil d'aide à la décision réalisé dans un cadre pédagogique : "
                "il ne remplace pas le diagnostic d'un médecin.")


# ---------------------------------------------------------------------------
# Page : Vue globale
# ---------------------------------------------------------------------------
elif page == "📊 Vue globale":
    if base_dir:
        stats = compute_stats(str(base_dir))
    else:
        stats = SAVED_STATS
        st.caption("Dataset complet non présent : affichage des statistiques enregistrées.")

    st.subheader("Le dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Entraînement", sum(stats["train"].values()))
    c2.metric("Validation", sum(stats["val"].values()))
    c3.metric("Test", sum(stats["test"].values()))

    df_stats = pd.DataFrame(stats).T  # lignes = split, colonnes = classes
    st.bar_chart(df_stats, stack=False)
    ratio = stats["train"]["PNEUMONIA"] / max(stats["train"]["NORMAL"], 1)
    st.caption(f"Déséquilibre : {ratio:.1f}x plus d'images PNEUMONIA que NORMAL dans l'entraînement "
               "(compensé par des poids de classe).")

    st.subheader("Performances du modèle (jeu de test, 624 images)")
    cols = st.columns(len(TEST_METRICS))
    for col, (name, value) in zip(cols, TEST_METRICS.items()):
        col.metric(name, value)

    if Path("evaluation_results.png").exists():
        with st.expander("Matrice de confusion et courbe ROC"):
            st.image("evaluation_results.png", width="stretch")
    if Path("model_comparison.png").exists():
        with st.expander("Comparaison VGG19 / ResNet50 / MobileNetV2"):
            st.image("model_comparison.png", width="stretch")
            st.caption("Comparaison rapide sur 5 époques seulement (sans les 2 phases d'entraînement).")


# ---------------------------------------------------------------------------
# Page : Dataset
# ---------------------------------------------------------------------------
elif page == "🖼️ Dataset":
    if base_dir:
        split = st.selectbox("Split", ["train", "val", "test"])
        folder_root = base_dir / split
    else:
        st.caption("Dataset complet non présent : affichage des images d'exemple.")
        folder_root = SAMPLES_DIR

    cls = st.selectbox("Classe", CLASSES)
    n_images = st.slider("Nombre d'images", 1, 12, 6)

    images = list_images(folder_root / cls)
    if not images:
        st.warning("Aucune image trouvée dans ce dossier.")
    else:
        picks = random.sample(images, k=min(n_images, len(images)))
        cols = st.columns(3)
        for i, path in enumerate(picks):
            with cols[i % 3]:
                st.image(str(path), caption=path.name, width="stretch")


# ---------------------------------------------------------------------------
# Page : Historique
# ---------------------------------------------------------------------------
elif page == "🕓 Historique":
    rows = get_history()
    if not rows:
        st.info("Aucune prédiction enregistrée pour l'instant. "
                "Faites une prédiction puis cliquez sur « Enregistrer dans l'historique ».")
    else:
        df = pd.DataFrame(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Analyses", len(df))
        c2.metric("PNEUMONIA", int((df["Résultat"] == "PNEUMONIA").sum()))
        c3.metric("NORMAL", int((df["Résultat"] == "NORMAL").sum()))

        st.dataframe(df, width="stretch", hide_index=True)

        st.download_button(
            "⬇️ Exporter en CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="historique_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

        with st.expander("🗑️ Vider l'historique"):
            if st.checkbox("Je confirme vouloir tout supprimer"):
                if st.button("Supprimer définitivement"):
                    clear_history()
                    st.rerun()
