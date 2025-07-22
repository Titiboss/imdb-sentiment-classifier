import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import nltk
import tensorflow_datasets as tfds
from nltk.corpus import movie_reviews
import requests
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

# Titre de l'application Streamlit
st.title("Comparateur de Classificateurs avec IMDB")

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

# Instanciation des classificateurs
logistic_reg = LogisticRegression(max_iter=1000)
decision_tree_clf = DecisionTreeClassifier()
random_forest_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier(max_iter=1000)

dnn_clf = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(1000,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Correct number of output units
])

dnn_clf.compile(optimizer='adam',
                loss='categorical_crossentropy', # Ensure using 'categorical_crossentropy' with one-hot labels
                metrics=['accuracy'])

# Liste des classificateurs avec leurs noms
classifiers = [
    ('Logistic Regression', logistic_reg),
    ('Decision Tree', decision_tree_clf),
    ('Random Forest', random_forest_clf),
    ('K-Nearest Neighbors', knn_clf),
    ('MLP', mlp_clf),
    ('DNN', dnn_clf)
]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if not isinstance(X_train, np.ndarray):
    X_train = X_train.toarray()
if not isinstance(X_test, np.ndarray):
    X_test = X_test.toarray()



# Affichage de la performance comparée des classificateurs
results = []
for name, classifier in classifiers:
    if name == 'DNN':
        classifier.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=10, batch_size=16)
        loss, accuracy = classifier.evaluate(X_test, tf.keras.utils.to_categorical(y_test))
        results.append((name, accuracy))
    else:
        classifier.fit(X_train, y_train)
        # Prédiction des étiquettes sur l'ensemble de test
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((name, accuracy))

results_df = pd.DataFrame(results, columns=['Classificateur', 'Accuracy'])
st.write(results_df)

# Obtenir les noms des 3 meilleurs classificateurs selon le nombre d'itérations
top_classifiers = results_df.nlargest(3, 'Accuracy')['Classificateur'].values


st.subheader("Prédiction d'un critique personnalisée")
# Sélection du classificateur
selected_classifier = st.selectbox("Sélectionnez un Classificateur", [name for name, _ in classifiers])

# Entraînement et évaluation du modèle sélectionné
for name, classifier in classifiers:
    if name == selected_classifier:
        st.subheader(f"Évaluation de {name}")
        # Entraînement du classificateur sur l'ensemble d'entraînement
        if name == 'DNN':
            loss, accuracy = classifier.evaluate(X_test, tf.keras.utils.to_categorical(y_test))
        else:
            # Prédiction des étiquettes sur l'ensemble de test
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # Affichage de l'accuracy
        st.write(f"Accuracy: {accuracy:.4f}")

# L'utilisateur peut faire entrer un avis
avis = st.text_area("Entrez un avis de film en anglais pour tester","The movie was bad. The animation and the graphics were poor. I would not recommend this movie.")
example = vectorizer.transform([avis])
# Check type
if not isinstance(example, np.ndarray):
    example = example.toarray()

# Classification selon le choix
if selected_classifier == 'DNN':
    if (np.argmax(dnn_clf.predict(example)) == 1):
        st.write('Avis positif')
    else:
        st.write('Avis négatif')
else:
    for name, classifier in classifiers:
        if name == selected_classifier:
            prediction = classifier.predict(example)
            st.write('Avis positif' if prediction == 1 else 'Avis négatif')

