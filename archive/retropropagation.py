import tensorflow as tf
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
from tensorflow.keras.layers import Dense, Flatten, Input
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
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1, 2),stop_words='english')
X = vectorizer.fit_transform(reviews)
y = np.array(labels)

# Instanciation des classificateurs
logistic_reg = LogisticRegression(C=1, max_iter=500, solver='liblinear')#saga ,liblinear,lbfgs max_iter=1000 si sample 1000
decision_tree_clf = DecisionTreeClassifier()
random_forest_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()

dnn_clf = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(1000,)),#128#256
    tf.keras.layers.Dropout(0.9), # On désative qql neuronnes pour eviter un chemin plus important
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.9),
    tf.keras.layers.Dense(2, activation='softmax')
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
        # Configuration de TensorBoard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
        classifier.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=10, batch_size=16, callbacks=[tensorboard_callback])#16#32
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

# Instructions pour exécuter TensorBoard
st.write("Pour visualiser la rétropropagation et d'autres métriques, exécutez la commande suivante dans votre terminal :")
st.code("tensorboard --logdir=./logs", language='bash')
st.write("Ensuite, ouvrez votre navigateur et accédez à l'URL fournie par TensorBoard (généralement http://localhost:6006).")


