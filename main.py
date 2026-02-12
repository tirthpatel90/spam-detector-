import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------
# 1. Load Dataset
# ------------------------------
print("Loading dataset...")

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Dataset loaded successfully.\n")


# ------------------------------
# 2. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)


# ------------------------------
# 3. TF-IDF Vectorization
# ------------------------------
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ------------------------------
# 4. Model Training
# ------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model trained successfully.\n")


# ------------------------------
# 5. Evaluation
# ------------------------------
y_pred = model.predict(X_test_tfidf)

print("========== MODEL EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------
# 6. Save Model
# ------------------------------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")


# ------------------------------
# 7. Manual Testing Loop
# ------------------------------
print("\n========== TEST YOUR OWN MESSAGE ==========")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Enter a message: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    msg_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(msg_tfidf)

    if prediction[0] == 1:
        print("🚨 SPAM MESSAGE\n")
    else:
        print("✅ NOT SPAM\n")
