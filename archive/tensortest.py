import numpy as np
import nltk
import tensorflow_datasets as tfds
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')

# Charger les critiques de films positives et négatives
positive_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('pos')]
negative_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('neg')]

# Combinez les critiques positives et négatives
reviews = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
# Afficher le type de données des caractéristiques (X) et des étiquettes (y)

# Extraction des caractéristiques (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X1 = vectorizer.fit_transform(reviews)
y1 = np.array(labels)
print("Type de données de X (caractéristiques) :", type(X1))
print("Type de données de y (étiquettes) :", type(y1))

# Charger l'ensemble de données IMDB de tfds
tfds_dataset = tfds.load('imdb_reviews', as_supervised=True)
tfds_train_dataset = tfds_dataset['train']

reviews = []
labels = []
for example, label in tfds_train_dataset:
    reviews.append(example.numpy())
    labels.append(label.numpy())

# Extraction des caractéristiques (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(reviews)
y = np.array(labels)
# Afficher le type de données des caractéristiques (X) et des étiquettes (y)
print("Type de données de X (caractéristiques) :", type(X2))
print("Type de données de y (étiquettes) :", type(y2))

num_samples_to_print = 5  # Nombre d'échantillons à imprimer
print("Extrait des caractéristiques (X) pour nlk :")
for i in range(num_samples_to_print):
    print("Commentaire", i+1, ":", X1[i])

# Imprimer les étiquettes (y)
print("\nExtrait des étiquettes (y) pour nlk :")
for i in range(num_samples_to_print):
    print("Étiquette", i+1, ":", y1[i])

print("Extrait des caractéristiques (X) pour TFDS :")
for i in range(num_samples_to_print):
    print("Commentaire", i+1, ":", X2[i])

# Imprimer les étiquettes (y)
print("\nExtrait des étiquettes (y) pour TFDS :")
for i in range(num_samples_to_print):
    print("Étiquette", i+1, ":", y2[i])