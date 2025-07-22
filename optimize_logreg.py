import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential

# Télécharger les données IMDB Movie Reviews de tfds
tfds_dataset = tfds.load('imdb_reviews', as_supervised=True)
tfds_train_dataset, tfds_test_dataset = tfds_dataset['train'], tfds_dataset['test']

reviews = []
labels = []
num_reviews_to_extract = 10000
count = 0

num_positive_reviews = 0
num_negative_reviews = 0

for example, label in tfds_train_dataset:
    if count >= num_reviews_to_extract:
        break
    if label == 1:
        num_positive_reviews += 1
    else:
        num_negative_reviews += 1
    reviews.append(example.numpy())
    labels.append(label.numpy())
    count += 1

print("Nombre de critiques positives :", num_positive_reviews)
print("Nombre de critiques négatives :", num_negative_reviews)

# Extraction des caractéristiques (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(reviews)
y = np.array(labels)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transformation des matrices en tableaux numpy si nécessaire
if not isinstance(X_train, np.ndarray):
    X_train = X_train.toarray()
if not isinstance(X_test, np.ndarray):
    X_test = X_test.toarray()

# Optimisation des hyperparamètres avec GridSearchCV
def optimize_classifier(classifier, param_grid):
    grid_search = GridSearchCV(classifier, param_grid, cv=5,verbose=1, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_


logistic_reg = optimize_classifier(LogisticRegression(), {
    'C': [0.5,1, 10],  # Paramètre de régularisation inverse
    'solver': ['liblinear', 'saga', 'lbfgs'],  # Algorithme à utiliser dans le problème d'optimisation
    'max_iter': [500, 1000, 1500]  # Nombre maximum d'itérations
})
""""
random_forest_clf = optimize_classifier(RandomForestClassifier(), {
    'n_estimators': [100,200],
    'max_depth': [20,25],
    'max_features': ['sqrt', 'log2']
})

mlp_clf = optimize_classifier(MLPClassifier(max_iter=1000), {
    'hidden_layer_sizes': [(5,), (3,), (2,), (1,)],
    'alpha': [0.005, 0.05, 0.5],
    'learning_rate': ['adaptive']
})
"""



