# Guide Complet du Projet DL

## 1) Objectif du projet

Ce projet fait une classification binaire de radiographies thoraciques :

- `NORMAL`
- `PNEUMONIA`

Le coeur du projet est dans le notebook `deepLearning.ipynb` (TensorFlow/Keras, transfer learning).

---

## 2) Contenu du dossier

- `deepLearning.ipynb` : entraînement, évaluation, inférence
- `app.py` : interface graphique Streamlit
- `requirements.txt` : dépendances Python
- `best_model_vgg19.h5` / `best_model_vgg19_ft.h5` : modèles entraînés
- `evaluation_results.png`, `model_comparison.png` : résultats générés
- `chest_xray/` : dataset local

---

## 3) Installation

Dans un terminal, depuis le dossier `DL` :

```bash
pip install -r requirements.txt
```

---

## 4) Dataset

Le projet utilise le dataset Kaggle:
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Méthode utilisée ici

- téléchargement manuel du fichier zip
- extraction dans `chest_xray/`
- structure attendue:
  - `chest_xray/chest_xray/train/NORMAL`
  - `chest_xray/chest_xray/train/PNEUMONIA`
  - `chest_xray/chest_xray/val/NORMAL`
  - `chest_xray/chest_xray/val/PNEUMONIA`
  - `chest_xray/chest_xray/test/NORMAL`
  - `chest_xray/chest_xray/test/PNEUMONIA`

---

## 5) Exécution du notebook

Ouvrir `deepLearning.ipynb` puis exécuter les cellules dans l’ordre.

### Ce que font les cellules

- Cellules 1-2 : imports + vérification/extraction dataset
- Cellules 3-6 : visualisation dataset et data pipeline
- Cellules 7-10 : construction + entraînement modèle
- Cellules 11-13 : évaluation et comparaison des modèles
- Cellules 14+ : inférence sur images

### Robustesse ajoutée

Le notebook filtre les extensions image valides pour éviter les erreurs dues aux fichiers cachés (ex: `.DS_Store`).

---

## 6) Interface graphique Streamlit

Lancer l’application :

```bash
streamlit run app.py
```

### Sections disponibles

1. **Overview**
   - métriques globales (`train`, `val`, `test`)
   - distribution `NORMAL` vs `PNEUMONIA`
   - statut du modèle chargé

2. **Dataset Explorer**
   - sélection du split (`train`, `val`, `test`)
   - sélection de classe
   - affichage d’images aléatoires

3. **Prediction**
   - upload d’une image
   - choix du seuil de décision
   - prédiction finale + confiance + probabilité brute

---

## 7) Que uploader dans Prediction

Uploader une **radio thoracique** (formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`).

Source recommandée pour test :

- `chest_xray/chest_xray/test/NORMAL/`
- `chest_xray/chest_xray/test/PNEUMONIA/`

---

## 8) Scénario de test rapide (QA)

### Test A - Démarrage app

- lancer `streamlit run app.py`
- vérifier qu’il n’y a pas d’erreur au démarrage

### Test B - Overview

- vérifier que les compteurs sont > 0
- vérifier les graphiques de distribution

### Test C - Dataset Explorer

- sélectionner plusieurs combinaisons split/classe
- vérifier que les images s’affichent

### Test D - Prediction

- uploader une image `NORMAL` puis `PNEUMONIA`
- vérifier l’affichage de la prédiction
- tester plusieurs seuils (0.3, 0.5, 0.7)

---

## 9) Erreurs fréquentes et solutions

### `UnicodeDecodeError` au lancement Streamlit

- cause : fichier `app.py` non encodé UTF-8
- solution : réenregistrer `app.py` en UTF-8

### `UnidentifiedImageError` dans notebook

- cause : fichier non image (ex: `.DS_Store`) dans les dossiers dataset
- solution : filtrer les extensions image valides (déjà appliqué)

### `Aucun modèle .h5 trouvé` dans Streamlit

- cause : modèle non entraîné ou absent du dossier projet
- solution : relancer l’entraînement dans le notebook

### `Dataset non trouvé`

- cause : mauvais emplacement du dataset
- solution : vérifier la structure `chest_xray/chest_xray/...`

---

## 10) Commandes utiles

Installer dépendances :

```bash
pip install -r requirements.txt
```

Lancer l’interface :

```bash
streamlit run app.py
```

---

## 11) Résumé

Ce projet est maintenant utilisable en local avec :

- notebook robuste pour entraînement/évaluation
- interface Streamlit pour démonstration et tests
- workflow clair pour un utilisateur ou testeur externe

