# Explication du code du projet

Ce document explique les deux fichiers principaux du projet :

- `deepLearning.ipynb` (pipeline IA / entraînement)
- `app.py` (interface graphique Streamlit)

---

## 1) `deepLearning.ipynb`

Le notebook est le coeur du projet de classification de radiographies thoraciques (`NORMAL` vs `PNEUMONIA`).

### A. Préparation et imports

Les premières cellules :

- importent les bibliothèques (`tensorflow`, `numpy`, `matplotlib`, `sklearn`, etc.)
- initialisent l’environnement
- préparent le dataset local

Objectif : rendre l’environnement prêt avant toute phase d’entraînement.

### B. Chargement et vérification du dataset

Le notebook utilise un dossier racine de données :

- `BASE_DIR = 'chest_xray/chest_xray'`

Les cellules associées :

- vérifient que la structure `train/val/test` existe
- comptent le nombre d’images dans `NORMAL` et `PNEUMONIA`
- filtrent les fichiers non-image (ex: `.DS_Store`) pour éviter les erreurs

### C. Visualisation des données

Plusieurs cellules affichent :

- la distribution des classes par split
- des exemples d’images
- des images après data augmentation

Objectif : comprendre la qualité et l’équilibre des données avant entraînement.

### D. Générateurs d’images (`ImageDataGenerator`)

Le notebook crée :

- `train_gen` pour l’entraînement (avec augmentation)
- `val_gen` pour validation
- `test_gen` pour test

Pourquoi : charger les images en batch et appliquer des transformations utiles au modèle.

### E. Construction du modèle

La fonction `build_model(...)` :

- charge une base pré-entraînée (`VGG19`, `ResNet50`, `MobileNetV2`)
- ajoute une tête de classification (dense + dropout + sortie sigmoïde)
- compile le modèle avec :
  - `binary_crossentropy`
  - `Adam`
  - métriques : `accuracy`, `precision`, `recall`, `auc`

### F. Entraînement en 2 phases

1. **Phase 1 (feature extraction)** :
   - backbone gelé
   - entraînement des couches de tête
2. **Phase 2 (fine-tuning)** :
   - dégel partiel du backbone
   - ajustement fin des poids

Callbacks utilisés :

- `EarlyStopping`
- `ModelCheckpoint` (sauvegarde `best_model_*.h5`)
- `ReduceLROnPlateau`

### G. Évaluation

Le notebook :

- charge le meilleur modèle sauvegardé
- génère les prédictions sur `test`
- calcule :
  - rapport de classification
  - accuracy, precision, recall, F1, AUC
- affiche :
  - matrice de confusion
  - courbe ROC

Fichiers de sortie :

- `evaluation_results.png`
- `model_comparison.png`

### H. Inférence (prédiction sur une image)

La fonction `predict_xray(...)` :

- charge et redimensionne l’image
- normalise les pixels
- obtient la probabilité
- retourne la classe prédite et la confiance

Elle est utilisée pour tester des images des dossiers `test/NORMAL` et `test/PNEUMONIA`.

---

## 2) `app.py` (interface Streamlit)

Ce fichier crée l’application web pour visualiser le dataset et faire des prédictions.

### A. Configuration générale

`st.set_page_config(...)` définit :

- titre de la page
- icône
- layout large

Un bloc CSS léger est injecté pour améliorer le rendu visuel.

### B. Fonctions utilitaires

- `dataset_base_dir()` : trouve automatiquement le bon chemin dataset
- `list_images()` : liste les images valides d’un dossier
- `count_images()` : compte les images
- `compute_stats()` : calcule les stats train/val/test (avec cache)
- `load_model()` : charge automatiquement le modèle `.h5` disponible
- `preprocess_image()` : prétraitement image (RGB, resize 224, normalisation)
- `predict()` : inférence sur image uploadée

### C. Sidebar et navigation

La barre latérale contient :

- la navigation entre 3 sections
- le chemin du dataset détecté

Les sections :

1. `Overview`
2. `Dataset Explorer`
3. `Prediction`

### D. Section `Overview`

Affiche :

- le nombre d’images dans train/val/test
- la distribution des classes
- l’état de chargement du modèle

### E. Section `Dataset Explorer`

Permet de :

- choisir un split (`train`, `val`, `test`)
- choisir une classe (`NORMAL`, `PNEUMONIA`)
- choisir le nombre d’images à afficher

Puis l’application affiche une galerie d’images aléatoires.

### F. Section `Prediction`

Permet de :

- uploader une image radio thoracique
- régler le seuil de décision
- obtenir :
  - la classe prédite
  - la confiance
  - la probabilité brute

---

## 3) Comment les deux fichiers travaillent ensemble

- `deepLearning.ipynb` sert à **entraîner** et **évaluer** les modèles
- `app.py` sert à **utiliser** un modèle entraîné dans une interface graphique

En résumé :

1. entraîner dans le notebook -> produit un `.h5`
2. lancer Streamlit -> charge ce `.h5`
3. tester des images dans l’interface

---

## 4) Commandes utiles

Installer dépendances :

```bash
pip install -r requirements.txt
```

Lancer l’interface :

```bash
streamlit run app.py
```

