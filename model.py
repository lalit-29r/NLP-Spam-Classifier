import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from preprocessing import preprocess_emails
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

nlp = spacy.load('en_core_web_sm')

df = pd.read_csv("dataset/email_dataset.csv")

df['text'] = df['text'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['spam'], test_size=0.2, random_state=2025, stratify=df['spam'])

knn = Pipeline([
    ('preprocess', FunctionTransformer(preprocess_emails, validate=False)),
    ('vector', TfidfVectorizer()),
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])

rf = Pipeline([
    ('preprocess', FunctionTransformer(preprocess_emails, validate=False)),
    ('vector', TfidfVectorizer()),
    ('rf', RandomForestClassifier(n_estimators=29, random_state=2025))
])

knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(classification_report(y_test, y_pred_knn))

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(classification_report(y_test, pred_rf))

joblib.dump(knn, 'model_knn.pkl')
joblib.dump(rf, 'model_rf.pkl')