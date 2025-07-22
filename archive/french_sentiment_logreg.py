import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
import spacy
import re
import joblib  # For saving the model

# Télécharger les stopwords en français si nécessaire
nltk.download('stopwords')
nltk.download('punkt')

# Charger le modèle français de spaCy
try:
    nlp = spacy.load('fr_core_news_sm')
except IOError:
    print("Téléchargement du modèle linguistique français de spaCy...")
    spacy.cli.download('fr_core_news_sm')
    nlp = spacy.load('fr_core_news_sm')

# Initialisation des outils de prétraitement
french_stopwords = stopwords.words('french')
stemmer = SnowballStemmer('french')


# Fonction de prétraitement des textes
def preprocess_text(text):
    # Mise en minuscule et suppression de la ponctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenisation
    tokens = word_tokenize(text, language='french')

    # Suppression des stopwords et racinisation
    tokens = [word for word in tokens if word not in french_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatisation avec spaCy
    doc = nlp(' '.join(tokens))
    tokens = [token.lemma_ for token in doc]

    return ' '.join(tokens)


# ----------------------------
# Traitement du fichier train_set.csv
# ----------------------------
print("Lecture des données d'entraînement")
train_df = pd.read_csv('train_set.csv', header=None, names=['label', 'text'], sep=',', on_bad_lines='skip')

# Filtrage des données et vérification de l'intégrité
print("Nettoyage des données")
if 'label' not in train_df.columns or 'text' not in train_df.columns:
    raise ValueError("Le fichier CSV doit contenir les colonnes 'label' et 'text'.")
train_df.dropna(inplace=True)
train_df.drop_duplicates(inplace=True)
train_df = train_df[train_df['label'].isin([0, 1])]  # Filtrage des labels valides

# Vérification de l'équilibrage des classes
label_counts = train_df['label'].value_counts()
print("Répartition des classes :")
print(label_counts)
if abs(label_counts[0] - label_counts[1]) > len(train_df) * 0.1:
    print("Déséquilibre détecté. Utilisation de class_weight='balanced'")
    class_weight = 'balanced'
else:
    print("Données équilibrées.")
    class_weight = None

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42)

# ----------------------------
# Vectorisation des textes avec TF-IDF
# ----------------------------
print("Vectorisation des textes avec TF-IDF")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Instanciation du modèle avec les paramètres spécifiés
# ----------------------------
print("Entraînement du modèle de régression logistique")

logistic_model = LogisticRegression(
    C=1,
    penalty='elasticnet',
    solver='saga',
    max_iter=100,
    l1_ratio=1.0,
    class_weight=class_weight
)

# Entraînement du modèle
logistic_model.fit(X_train_vec, y_train)

# Prédiction et évaluation
print("Évaluation du modèle sur l'ensemble de test")
y_pred = logistic_model.predict(X_test_vec)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Précision finale sur l'ensemble de test : {final_accuracy:.2f}")

# ----------------------------
# Sauvegarde du modèle
# ----------------------------
print("Sauvegarde du modèle")
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Modèle et vectorizer sauvegardés sous 'logistic_model.pkl' et 'tfidf_vectorizer.pkl'")

# ----------------------------
# Sauvegarde des prédictions
# ----------------------------
print("Sauvegarde des prédictions sur l'ensemble de test")
test_results = pd.DataFrame({'id': X_test.index, 'sentiment': y_pred})

# Format du fichier de sortie
output_filename = 'predictions_train-set.csv'
test_results.to_csv(output_filename, sep=';', index=False, header=False)
print(f"Fichier '{output_filename}' créé avec le bon format")

# ----------------------------
# Traitement du fichier score_set.csv
# ----------------------------
print("Lecture et traitement du fichier score_set.csv")
score_df = pd.read_csv('score_set.csv')

# Prétraitement des textes
tqdm.pandas(desc="Prétraitement des textes")
score_df['clean_text'] = score_df['Text'].progress_apply(preprocess_text)

# Vectorisation des textes de score_set.csv avec TF-IDF
print("Vectorisation des textes de score_set.csv avec TF-IDF")
X_score_vec = vectorizer.transform(score_df['clean_text'])

# Prédiction des sentiments avec le modèle optimisé
print("Prédiction des sentiments avec le modèle TF-IDF")
score_df['sentiment'] = logistic_model.predict(X_score_vec)

# Sauvegarde des prédictions dans le format requis
print("Sauvegarde des prédictions dans 'predictions_score.csv'")
output_filename = 'predictions_score.csv'
score_df[['id', 'sentiment']].to_csv(output_filename, sep=';', index=False, header=False)
print(f"Fichier '{output_filename}' créé avec le bon format")

print("Traitement terminé")
