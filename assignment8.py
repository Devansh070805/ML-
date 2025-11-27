import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import string
import re
import nltk

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print("Dataset Loaded:")
print(df.head(), "\n")


df['label'] = df['label'].map({'spam': 1, 'ham': 0})


nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

print("Sample cleaned text:")
print(df[['text', 'clean_text']].head(), "\n")


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape, "\n")


print("Class distribution:")
print(df['label'].value_counts(), "\n")

df['label'].value_counts().plot(kind='bar')
plt.title("Spam vs Ham Distribution")
plt.xticks([0,1], ['Ham', 'Spam'], rotation=0)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=column_names, na_values="?")
df["target"] = df["target"].astype(int)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
df = df.dropna()

print("Dataset loaded successfully!")
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)

y_pred_train = stump.predict(X_train)
y_pred_test = stump.predict(X_test)

print("\n===== BASELINE DECISION STUMP =====")
print("Train Accuracy :", accuracy_score(y_train, y_pred_train))
print("Test Accuracy  :", accuracy_score(y_test, y_pred_test))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_test))

n_estimators_list = [5, 10, 25, 50, 100]
learning_rates = [0.1, 0.3, 1.0]

results = []

print("\n===== ADABOOST TRAINING =====\n")

for n in n_estimators_list:
    for lr in learning_rates:
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=n,
            learning_rate=lr,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append([n, lr, acc])
        print(f"n_estimators={n}, learning_rate={lr} → Test Accuracy = {acc:.4f}")

results_df = pd.DataFrame(results, columns=["n_estimators", "learning_rate", "accuracy"])
best_row = results_df.iloc[results_df["accuracy"].idxmax()]

print("\n===== BEST CONFIGURATION =====")
print(best_row)

best_n = int(best_row["n_estimators"])
best_lr = float(best_row["learning_rate"])

best_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=best_n,
    learning_rate=best_lr,
    random_state=42
)

best_model.fit(X_train, y_train)

errors = []

for est in best_model.estimators_:
    pred = est.predict(X_train)
    misclassified = (pred != y_train)
    error = np.mean(misclassified)
    errors.append(error)

alphas = best_model.estimator_weights_

plt.figure(figsize=(6, 4))
plt.plot(range(1, best_n + 1), errors)
plt.xlabel("Boosting Round")
plt.ylabel("Error")
plt.title("Boosting Round vs Training Error")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(range(1, best_n + 1), alphas)
plt.xlabel("Boosting Round")
plt.ylabel("Alpha (Estimator Weight)")
plt.title("Boosting Round vs Alpha")
plt.tight_layout()
plt.show()

final_pred_train = best_model.predict(X_train)
final_pred_test = best_model.predict(X_test)

print("\n===== FINAL MODEL RESULTS (BEST ADABOOST) =====")
print("Train Accuracy :", accuracy_score(y_train, final_pred_train))
print("Test Accuracy  :", accuracy_score(y_test, final_pred_test))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, final_pred_test))
print("\nClassification Report (Test):\n", classification_report(y_test, final_pred_test))

importances = best_model.feature_importances_

plt.figure(figsize=(8, 6))
plt.barh(X.columns, importances)
plt.xlabel("Feature Importance Score")
plt.title("Feature Importance from AdaBoost")
plt.tight_layout()
plt.show()

print("\nTop important features:")
feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances})
print(feat_imp.sort_values("Importance", ascending=False).head())