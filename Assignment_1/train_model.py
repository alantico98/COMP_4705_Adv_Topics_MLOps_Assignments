import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
imdb_df = pd.read_csv('./imdb_dataset.csv')

# Split the data into features
X = imdb_df['review']
y = imdb_df['sentiment']

# Train the model by making a pipeline
steps = [
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
]

pipeline = Pipeline(steps)
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, 'sentiment_model.pkl')
joblib.dump(X, 'sentiment_labels.pkl')

print("Model trained and saved as sentiment_model.pkl")