"""Génération du compte-rendu PDF d'une analyse (librairie fpdf2)."""

from datetime import datetime
from io import BytesIO
from pathlib import Path

from fpdf import FPDF
from PIL import Image

# Couleurs (identiques à l'application)
INK = (27, 36, 48)
MUTED = (91, 102, 117)
LINE = (223, 227, 232)
ACCENT = (31, 78, 121)
COLORS = {
    "NORMAL": {"main": (47, 111, 94), "bg": (234, 243, 240)},
    "PNEUMONIA": {"main": (180, 65, 47), "bg": (249, 236, 234)},
}

MARGIN = 18
PAGE_W = 210
CONTENT_W = PAGE_W - 2 * MARGIN


class Report(FPDF):
    def footer(self):
        self.set_y(-22)
        self.set_draw_color(*LINE)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.multi_cell(
            CONTENT_W - 20, 3.6,
            "Document généré automatiquement par Souffle, un modèle de Deep Learning développé dans un cadre "
            "pédagogique. Il constitue une aide à la lecture et ne remplace pas l'interprétation d'un médecin.",
        )
        self.set_xy(PAGE_W - MARGIN - 20, -19)
        self.cell(20, 3.6, f"Page {self.page_no()}/{{nb}}", align="R")


def _section_title(pdf, text, x=MARGIN, w=CONTENT_W):
    pdf.set_x(x)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(w, 5, text.upper(), new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.set_draw_color(*LINE)
    pdf.line(x, y, x + w, y)
    pdf.ln(2.5)


def _info_row(pdf, x, label, value, w):
    pdf.set_x(x)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(w * 0.42, 6.5, label)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*INK)
    pdf.cell(w * 0.58, 6.5, value, new_x="LMARGIN", new_y="NEXT")


def build_pdf(img: Image.Image, filename, label, confidence, probability, threshold, model_name) -> bytes:
    now = datetime.now()
    reference = now.strftime("AN-%Y%m%d-%H%M%S")
    color = COLORS[label]

    pdf = Report(format="A4")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(False)
    pdf.add_page()

    # --- Bandeau supérieur -------------------------------------------------
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 0, PAGE_W, 4, "F")

    pdf.set_xy(MARGIN, 14)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(*INK)
    pdf.cell(120, 8, "Compte-rendu d'analyse")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.set_xy(PAGE_W - MARGIN - 60, 14)
    pdf.cell(60, 4.5, "Référence", align="R", new_x="LEFT", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*INK)
    pdf.cell(60, 4.5, reference, align="R")

    pdf.set_xy(MARGIN, 22.5)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(120, 5, "Souffle - lecture assistée des radiographies pulmonaires")

    pdf.set_draw_color(*LINE)
    pdf.line(MARGIN, 32, PAGE_W - MARGIN, 32)

    # --- Informations sur l'examen ----------------------------------------
    pdf.set_y(38)
    _section_title(pdf, "Informations sur l'analyse")
    half = CONTENT_W / 2 - 4
    top = pdf.get_y()
    left_rows = [
        ("Fichier", filename if len(filename) <= 34 else filename[:31] + "..."),
        ("Date", now.strftime("%d/%m/%Y à %H:%M")),
    ]
    right_rows = [
        ("Modèle", "VGG19 (fine-tuning)"),
        ("Seuil de décision", f"{threshold:.2f}"),
    ]
    for lbl, val in left_rows:
        _info_row(pdf, MARGIN, lbl, val, half)
    pdf.set_y(top)
    for lbl, val in right_rows:
        _info_row(pdf, MARGIN + half + 8, lbl, val, half)

    # --- Image + Résultat --------------------------------------------------
    y0 = top + 2 * 6.5 + 8
    col_w = CONTENT_W / 2 - 4
    right_x = MARGIN + col_w + 8

    pdf.set_y(y0)
    _section_title(pdf, "Image analysée", MARGIN, col_w)
    pdf.set_xy(right_x, y0)
    _section_title(pdf, "Résultat", right_x, col_w)
    body_y = pdf.get_y()

    # Image dans un cadre noir, proportions conservées
    frame_h = 92
    pdf.set_fill_color(17, 17, 17)
    pdf.rect(MARGIN, body_y, col_w, frame_h, "F")
    ratio = img.width / img.height
    w, h = (col_w, col_w / ratio) if ratio >= col_w / frame_h else (frame_h * ratio, frame_h)
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=88)
    buffer.seek(0)
    pdf.image(buffer, x=MARGIN + (col_w - w) / 2, y=body_y + (frame_h - h) / 2, w=w, h=h)

    # Encadré du verdict
    pdf.set_fill_color(*color["bg"])
    pdf.rect(right_x, body_y, col_w, 30, "F")
    pdf.set_fill_color(*color["main"])
    pdf.rect(right_x, body_y, 1.6, 30, "F")
    pdf.set_xy(right_x + 6, body_y + 5)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(col_w - 10, 4.5, "Classe prédite", new_x="LEFT", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*color["main"])
    pdf.cell(col_w - 10, 10, label, new_x="LEFT", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    pdf.cell(col_w - 10, 5, f"Confiance : {confidence * 100:.1f} %")

    # Jauge de probabilité avec repère du seuil
    gy = body_y + 40
    pdf.set_xy(right_x, gy)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(col_w / 2, 4.5, "Probabilité de pneumonie")
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*INK)
    pdf.cell(col_w / 2, 4.5, f"{probability:.3f}", align="R")
    bar_y = gy + 7
    pdf.set_fill_color(237, 240, 243)
    pdf.rect(right_x, bar_y, col_w, 3.2, "F")
    pdf.set_fill_color(*color["main"])
    pdf.rect(right_x, bar_y, max(col_w * probability, 0.4), 3.2, "F")
    tx = right_x + col_w * threshold
    pdf.set_fill_color(*INK)
    pdf.rect(tx - 0.3, bar_y - 1.5, 0.6, 6.2, "F")
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*MUTED)
    pdf.set_xy(right_x, bar_y + 5)
    pdf.cell(10, 3.5, "0")
    pdf.set_xy(tx - 12, bar_y + 5)
    pdf.cell(24, 3.5, f"seuil {threshold:.2f}", align="C")
    pdf.set_xy(right_x + col_w - 10, bar_y + 5)
    pdf.cell(10, 3.5, "1", align="R")

    # Interprétation
    pdf.set_xy(right_x, bar_y + 15)
    _section_title(pdf, "Interprétation", right_x, col_w)
    if label == "PNEUMONIA":
        text = (
            f"La probabilité estimée ({probability:.3f}) dépasse le seuil de décision ({threshold:.2f}). "
            "Le modèle a identifié des caractéristiques compatibles avec une pneumonie. "
            "Une confirmation par un médecin est nécessaire."
        )
    else:
        text = (
            f"La probabilité estimée ({probability:.3f}) est inférieure au seuil de décision ({threshold:.2f}). "
            "Le modèle n'a pas identifié de signe de pneumonie sur cette image. "
            "Ce résultat ne dispense pas d'un avis médical."
        )
    pdf.set_x(right_x)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    pdf.multi_cell(col_w, 4.8, text)

    # --- Méthode -------------------------------------------------------------
    pdf.set_y(body_y + frame_h + 10)
    _section_title(pdf, "Méthode")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    pdf.multi_cell(
        CONTENT_W, 4.8,
        "Réseau de neurones convolutif VGG19 pré-entraîné sur ImageNet puis ajusté (fine-tuning) sur "
        "5 216 radiographies thoraciques pédiatriques. Image redimensionnée en 224 x 224 pixels. "
        "Performances sur 624 images de test : accuracy 92,5 %, sensibilité (recall) 98,5 %, "
        "précision 90,4 %, AUC 0,975.",
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(CONTENT_W, 4, f"Fichier du modèle : {Path(model_name).name}")

    return bytes(pdf.output())
