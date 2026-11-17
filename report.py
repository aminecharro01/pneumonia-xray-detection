"""Génération d'un compte-rendu PDF d'une prédiction (librairie fpdf2)."""

from datetime import datetime
from io import BytesIO

from fpdf import FPDF
from PIL import Image


def build_pdf(img: Image.Image, filename, label, confidence, probability, threshold, model_name) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Titre
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Compte-rendu d'analyse radiographique", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # Image (centrée, 90 mm de large)
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    width = 90
    height = width * img.height / img.width
    pdf.image(buffer, x=(210 - width) / 2, w=width, h=height)
    pdf.ln(6)

    # Résultat en couleur
    pdf.set_font("Helvetica", "B", 16)
    if label == "PNEUMONIA":
        pdf.set_text_color(200, 40, 40)
    else:
        pdf.set_text_color(30, 140, 70)
    pdf.cell(0, 10, f"Résultat : {label}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Détails
    pdf.set_font("Helvetica", "", 11)
    details = [
        ("Fichier", filename or "-"),
        ("Confiance", f"{confidence * 100:.2f} %"),
        ("Probabilité de pneumonie", f"{probability:.4f}"),
        ("Seuil de décision", f"{threshold:.2f}"),
        ("Modèle", model_name),
    ]
    for key, value in details:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(70, 8, key)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(value), new_x="LMARGIN", new_y="NEXT")

    # Avertissement
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(
        0, 5,
        "Avertissement : ce résultat est produit par un modèle de Deep Learning réalisé dans un cadre "
        "pédagogique. Il s'agit d'un outil d'aide et ne remplace en aucun cas l'avis d'un médecin.",
    )

    return bytes(pdf.output())
