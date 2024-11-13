import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Sample dataset (replace this with your actual dataset)

df = pd.read_csv('IMDB_Movie_Review.csv')

# Text and labels
texts = df['review'].tolist()
labels = df['sentiment'].apply(lambda x: 1 if x == "positive" else 0).tolist()  # Convert labels to 0 (negative) and 1 (positive)

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Model evaluation
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
# Calculate additional metrics for evaluation

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred, pos_label='positive')  # Change 'positive' to your label if necessary
recall = recall_score(y_test, y_pred, pos_label='positive')
f1 = f1_score(y_test, y_pred, pos_label='positive')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Classification Report (includes precision, recall, F1-score, and support)
class_report = classification_report(y_test, y_pred)

# ROC-AUC Score (for binary classification)
roc_auc = roc_auc_score(y_test, y_pred)

# Display the metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Confusion Matrix: \n{conf_matrix}")
print(f"Classification Report: \n{class_report}")
print(f"ROC-AUC Score: {roc_auc}")


# Save the model and vectorizer
joblib.dump(clf, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
