import pandas as pd
import re
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('data sets/Resume dataset.csv')

print(data.columns)

before = data.shape[0]
data = data.drop_duplicates()
after = data.shape[0]

print(f"Removed {before - after} duplicate rows. New shape: {data.shape}")
print(f"Remaining duplicates: {data.duplicated().sum()}")

data["Text"] = data["Text"].fillna("").str.lower()

url_pattern = (
    r'https?://\S+|'
    r'www\.\S+|'
    r'\b\S+\.(?:com|org|net|io|edu|gov|co|in|ai|info|biz|me|us|uk|ca|au)\S*'
)

data["Text"] = data["Text"].str.replace(url_pattern, "", regex=True)
data["Text"] = data["Text"].str.replace(r"\s+", " ", regex=True).str.strip()

remaining_urls = data["Text"].str.contains(
    url_pattern,
    regex=True,
    case=False,
    na=False
).sum()

print("Rows with URLs remaining:", remaining_urls)
data["Text"] = data["Text"].str.replace(r"\s+", " ", regex=True).str.strip()

remaining_urls = data["Text"].str.contains(url_pattern, case=False, regex=True).sum()
print(f"Rows with URLs remaining: {remaining_urls}")


email_pattern = r'\S+@\S+\.\S+'
data['Text'] = data['Text'].apply(lambda x: re.sub(email_pattern, '', x))
data['Text'] = data['Text'].str.replace(r'\s+', ' ', regex=True).str.strip()

remaining_emails = data['Text'].str.contains(email_pattern, case=False, regex=True, na=False).sum()
print(f"Rows with emails remaining: {remaining_emails}")

data['Text'] = data['Text'].apply(lambda x: re.sub(r'\+?\d{10,13}', '', x))
data['Text'] = data['Text'].str.replace(r'\s+', ' ', regex=True).str.strip()

remaining_phones = data['Text'].str.contains(r'\d{10}', regex=True).sum()
print(f"Rows with phone numbers remaining: {remaining_phones}")


data['Text'] = data['Text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
data['Text'] = data['Text'].str.replace(r'\s+', ' ', regex=True).str.strip()

remaining_punct = data['Text'].str.contains(r'[^a-z0-9\s]', regex=True).sum()
print(f"Rows with punctuation remaining: {remaining_punct}")

data['Tokens'] = data['Text'].str.split()

print(data[['Text', 'Tokens']].head())

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = ENGLISH_STOP_WORDS


data['Tokens'] = data['Tokens'].apply(
    lambda words: [word for word in words if word not in stop_words]
)

print(data['Tokens'].head())

data['Text'] = data['Tokens'].apply(lambda words: " ".join(words))

from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['Text'])

y = data['category']

print("TF-IDF Shape:", X.shape)

print("Target Shape:", y.shape)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, "resume_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")


from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=30,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
