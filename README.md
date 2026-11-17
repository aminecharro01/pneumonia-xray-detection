# Souffle : lecture assistée des radiographies pulmonaires

Classification de radiographies thoraciques (**NORMAL** vs **PNEUMONIA**) par transfer learning (VGG19, TensorFlow/Keras), avec une application web Streamlit utilisable sur ordinateur et sur téléphone.

**Résultats sur le jeu de test (624 images) :** accuracy 92,5 % · recall 98,5 % · AUC 0,975

## Structure du projet

```
DL/
  deepLearning.ipynb          # Notebook : préparation, entraînement, évaluation
  app.py                      # Application Streamlit (interface)
  history.py                  # Historique des prédictions (SQLite)
  report.py                   # Génération du compte-rendu PDF
  model/pneumonia_model.keras # Modèle final allégé (81 Mo), utilisé par l'app
  samples/                    # 8 radios d'exemple (démo en ligne)
  .streamlit/config.toml      # Thème et configuration Streamlit
  requirements.txt            # Dépendances de l'application
  requirements-notebook.txt   # Dépendances du notebook
```

## Fonctionnalités de l'application

- **Prédiction** : importer une image, prendre une photo (téléphone) ou choisir un exemple. Seuil de décision réglable.
- **Compte-rendu PDF** téléchargeable après chaque prédiction.
- **Historique** des analyses (base SQLite), avec export CSV.
- **Vue globale** : statistiques du dataset et performances du modèle.
- **Dataset** : galerie d'images.
- **Vérification simple** : un avertissement s'affiche si l'image ne ressemble pas à une radio (image colorée).
- **Responsive** : l'interface s'adapte au téléphone (colonnes empilées, navigation en haut de page).

## Lancer en local

TensorFlow ne supporte pas Python 3.14 : il faut **Python 3.11**.

```bash
uv venv venv --python 3.11              # ou : py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

L'app s'ouvre sur http://localhost:8501.

**Tester sur son téléphone (même Wi-Fi) :** Streamlit affiche une « Network URL » (ex. `http://192.168.1.20:8501`). Ouvrez-la dans le navigateur du téléphone.

## Ré-entraîner le modèle (optionnel)

1. Télécharger le dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) et le placer dans le dossier sous le nom `chest-xray-pneumonia.zip`.
2. `pip install -r requirements-notebook.txt`
3. Exécuter `deepLearning.ipynb` dans l'ordre (sur CPU : plusieurs heures ; conseillé : Google Colab avec GPU).

## Mise en ligne (Streamlit Community Cloud, gratuit)

1. Créer un dépôt GitHub (public) et y envoyer le projet **avec git en ligne de commande**. L'upload via le site GitHub est limité à 25 Mo, or le modèle fait 81 Mo.
   ```bash
   git init
   git add .
   git commit -m "Projet détection pneumonie"
   git branch -M main
   git remote add origin https://github.com/<utilisateur>/<depot>.git
   git push -u origin main
   ```
   Le `.gitignore` exclut le dataset (2,4 Go), les anciens `.h5` et l'environnement `venv/`.
2. Aller sur https://share.streamlit.io et se connecter avec GitHub.
3. **Create app**, puis choisir le dépôt, la branche `main` et le fichier `app.py`.
4. Dans **Advanced settings**, choisir **Python 3.11**.
5. **Deploy**. Le premier démarrage prend quelques minutes (installation de TensorFlow).

On obtient un lien public du type `https://<nom>.streamlit.app`, utilisable sur ordinateur et sur téléphone.

**Limites de la version gratuite :**
- L'app se met en veille après quelques jours sans visite. Il suffit de cliquer sur « wake up ».
- L'historique SQLite est réinitialisé à chaque redémarrage du serveur.

## Version mobile

L'application est **responsive** et fonctionne dans le navigateur du téléphone, avec l'appareil photo pour prendre une radio en photo.

Pour l'avoir comme une application : dans Chrome (Android) ou Safari (iPhone), ouvrir le lien puis choisir **« Ajouter à l'écran d'accueil »**. Une icône apparaît et l'app s'ouvre en un clic.

> Streamlit ne permet pas de créer une vraie PWA (mode hors-ligne). De toute façon, le modèle tourne sur le serveur, donc une connexion internet reste nécessaire.

## Avertissement

Projet pédagogique : outil d'aide qui ne remplace pas l'avis d'un médecin.
