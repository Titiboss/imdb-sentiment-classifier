import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import Sequential
from tensorflow.keras import EarlyStopping
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import time


# Start the timer
start_time = time.time()


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
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(reviews)
y = np.array(labels)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transformation des matrices en tableaux numpy si nécessaire
if not isinstance(X_train, np.ndarray):
    X_train = X_train.toarray()
if not isinstance(X_test, np.ndarray):
    X_test = X_test.toarray()

# Définir une fonction pour créer le modèle
def create_model(optimizer='adam', neurons1=128, neurons2=128, neurons3=128, dropout_rate=0.5):
    model = Sequential()
    model.add(Input(shape=(1000,)))
    model.add(Dense(neurons1, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrappez le modèle Keras dans un KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Définir la grille de recherche des hyperparamètres
param_grid = {
    'batch_size': [16, 32],
    'epochs': [10, 15],
    'optimizer': ['adam'],
    'model__neurons1': [128],  # Préfixe 'model__' ajouté
    'model__neurons2': [128],  # Préfixe 'model__' ajouté
    'model__dropout_rate': [0.7,0.9]  # Préfixe 'model__' ajouté
}

# Utiliser GridSearchCV pour rechercher les meilleurs hyperparamètres
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, return_train_score=True)
grid_result = grid.fit(X_train, tf.keras.utils.to_categorical(y_train))


# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres : %s avec une accuracy de %f" % (grid_result.best_params_, grid_result.best_score_))

# Entraîner le modèle avec les meilleurs hyperparamètres et enregistrer l'historique
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
best_model = grid_result.best_estimator_
history = best_model.fit(X_train, tf.keras.utils.to_categorical(y_train),
                         validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
                         callbacks=[early_stopping])
# Évaluer le modèle sur l'ensemble de test
accuracy = best_model.score(X_test, tf.keras.utils.to_categorical(y_test))
print("Accuracy sur l'ensemble de test : %.2f%%" % (accuracy * 100))

# Utiliser Streamlit pour l'interface utilisateur
st.title("UltraOptimisation des Hyperparamètres 3 couches de neurones pour DNN avec IMDB")

# Affichage de la performance du modèle
st.write(f"Meilleurs hyperparamètres : {grid_result.best_params_}")
st.write(f"Accuracy sur l'ensemble de test : {accuracy:.4f}")

# Afficher les graphiques de la perte et de la précision au fil des époques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graphique de la perte (loss) par rapport aux époques
ax1.plot(history.history_['loss'], label='Training Loss')
ax1.plot(history.history_['val_loss'], label='Validation Loss')
ax1.set_title('Loss par Epoch')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Graphique de la précision (accuracy) par rapport aux époques
ax2.plot(history.history_['accuracy'], label='Training Accuracy')
ax2.plot(history.history_['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Accuracy par Epoch')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Afficher les graphiques dans Streamlit
st.pyplot(fig)

# Afficher les résultats des performances pour les différents paramètres
results = pd.DataFrame(grid_result.cv_results_)
results = results[['param_batch_size', 'param_epochs', 'param_optimizer', 'param_model__neurons1', 'param_model__neurons2', 'param_model__dropout_rate', 'mean_test_score', 'mean_train_score']]
st.write("Performance par Hyperparamètres")
st.write(results)
# End the timer
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print("Total Time Taken: {:.2f} seconds".format(total_time))
