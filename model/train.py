# Train a TF-IDF + Logistic Regression model on resume_dataset.csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('data/resume_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
    ('clf', LogisticRegression(max_iter=200))
])

pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
joblib.dump(pipe, 'model/tfidf_logreg.joblib')
print('Model saved to model/tfidf_logreg.joblib')
