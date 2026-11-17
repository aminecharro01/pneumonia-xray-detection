"""Historique des prédictions, stocké dans une petite base SQLite (fichier local)."""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/history.db")


def _connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT    NOT NULL,
            filename    TEXT,
            label       TEXT    NOT NULL,
            probability REAL    NOT NULL,
            threshold   REAL    NOT NULL,
            model       TEXT
        )
        """
    )
    return conn


def save_prediction(filename, label, probability, threshold, model):
    """Ajoute une ligne dans l'historique."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, filename, label, probability, threshold, model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, label, probability, threshold, model),
        )


def get_history():
    """Retourne toutes les prédictions, de la plus récente à la plus ancienne."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT created_at, filename, label, probability, threshold, model "
            "FROM predictions ORDER BY id DESC"
        ).fetchall()
    columns = ["Date", "Fichier", "Résultat", "Probabilité pneumonie", "Seuil", "Modèle"]
    return [dict(zip(columns, row)) for row in rows]


def clear_history():
    with _connect() as conn:
        conn.execute("DELETE FROM predictions")
