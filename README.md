# 🎬 IMDB Sentiment Classifier

Projet académique de Bac +4 — Ce projet compare plusieurs modèles de Machine Learning et Deep Learning pour la **classification d’émotions simples** (positif / négatif) à partir de critiques de films du dataset IMDB.  
Il inclut une interface interactive développée avec **Streamlit** pour explorer les performances des modèles.

---

## 📌 Objectifs

- Implémenter différents classificateurs (ML/DL) pour l’analyse de sentiments binaires.
- Optimiser les performances d’un DNN à l’aide de `GridSearchCV`.
- Comparer les résultats à l’aide d’une interface interactive.
- Fournir un code propre, modulaire, et reproductible.

---

## 🧠 Modèles comparés

- Régression Logistique
- Arbre de Décision
- Forêt Aléatoire
- K-Nearest Neighbors
- MLP (Multilayer Perceptron)
- Deep Neural Network (DNN) optimisé

---

## 🗂️ Structure du projet

```bash

├── main.py # Interface principale Streamlit
├── optimize_dnn.py # Optimisation d’un DNN avec GridSearchCV
├── optimize_logreg.py # Tuning de la régression logistique
├── README.md # Ce fichier
│
└── archive/
├── test.py
├── french_sentiment_logreg.py # Analyse de sentiments sur des textes français (expérimental)
├── retropropagation.py
├── test_optimize.py
└── tensortest.py
