import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

blue = "#2563EB"
green = "#16A34A"
red = "#DC2626"
orange = "#EA580C"

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def print_step(number, title):
    print("\nSTEP " + str(number) + ": " + title)
    print("-" * 50)



# Step 1: Load the dataset
print_step(1, "Load the dataset")

csv_path = os.path.join(BASE_DIR, "product_reviews.csv")
df = pd.read_csv(csv_path)

print("Rows:", df.shape[0], "Columns:", df.shape[1])
print(df.head())
print("\nSentiment counts:")
print(df["sentiment"].value_counts())



# Step 2: Clean and preprocess the text
print_step(2, "Clean and preprocess the text")

def clean_text(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch.isalpha() or ch == " ")
    text = " ".join(text.split())
    return text

df["clean_review"] = df["review"].apply(clean_text)

print("Example before and after cleaning:")
print("Before:", df["review"].iloc[0])
print("After :", df["clean_review"].iloc[0])

df = df[df["sentiment"] != "neutral"].reset_index(drop=True)
print("\nDropped neutral reviews, keeping positive vs negative only")
print("Remaining rows:", len(df))



# Step 3: Visualize sentiment distribution
print_step(3, "Visualize sentiment distribution")

fig, ax = plt.subplots(figsize=(6, 4))
sentiment_counts = df["sentiment"].value_counts()
color_map = {"positive": green, "negative": red}
bar_colors = [color_map[s] for s in sentiment_counts.index]
ax.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, width=0.4)
ax.set_title("Sentiment Distribution in Dataset")
ax.set_ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "sentiment_distribution.png"), dpi=150)
plt.close()

print("Saved sentiment_distribution.png")


# Step 4: Convert text to TF-IDF features
print_step(4, "Convert text to TF-IDF features")

# tokenization and stopword removal happen inside TfidfVectorizer
# max_features increased and min_df added to drop very rare/noisy words
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, min_df=2)
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

print("TF-IDF matrix shape:", X.shape)


# Step 5: Split data into train and test sets

print_step(5, "Split data into train and test sets")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])



# Step 6: Train the models
print_step(6, "Train the models")

# Naive Bayes - simple and works well for text
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("Naive Bayes trained")

svm = SVC(kernel="linear", random_state=42,
          class_weight="balanced")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM trained")


nn = MLPClassifier(hidden_layer_sizes=(30,),
                   max_iter=800,
                   random_state=42,
                   alpha=0.01)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)
print("Neural Network trained")



# Step 7: Evaluate the models
print_step(7, "Evaluate the models")

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

nb_metrics = get_metrics(y_test, nb_pred)
svm_metrics = get_metrics(y_test, svm_pred)
nn_metrics = get_metrics(y_test, nn_pred)

print("Naive Bayes metrics:")
for k, v in nb_metrics.items():
    print(" ", k, round(v, 4))

print("\nSVM metrics:")
for k, v in svm_metrics.items():
    print(" ", k, round(v, 4))

print("\nNeural Network metrics:")
for k, v in nn_metrics.items():
    print(" ", k, round(v, 4))



# Step 8: Visualize model performance comparison
print_step(8, "Visualize model performance comparison")

fig, ax = plt.subplots(figsize=(9, 5))
metric_names = list(nb_metrics.keys())
x = range(len(metric_names))
width = 0.25

ax.bar([i - width for i in x], list(nb_metrics.values()), width, label="Naive Bayes", color=blue)
ax.bar(x, list(svm_metrics.values()), width, label="SVM", color=green)
ax.bar([i + width for i in x], list(nn_metrics.values()), width, label="Neural Network", color=orange)

ax.set_xticks(list(x))
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.1)
ax.set_title("Model Performance Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "model_comparison.png"), dpi=150)
plt.close()

print("Saved model_comparison.png")



# Step 9: Confusion matrix for the best model
print_step(9, "Confusion matrix for the best model")

scores = {"Naive Bayes": nb_metrics["F1-Score"], "SVM": svm_metrics["F1-Score"], "Neural Network": nn_metrics["F1-Score"]}
best_model_name = max(scores, key=scores.get)
best_pred = {"Naive Bayes": nb_pred, "SVM": svm_pred, "Neural Network": nn_pred}[best_model_name]

labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title("Confusion Matrix - " + best_model_name)
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

print("Best model:", best_model_name)
print("Saved confusion_matrix.png")



# Step 10: Summary

print_step(10, "Summary")

print("Best model based on F1-score:", best_model_name)
print("Accuracy:", round(scores[best_model_name] * 100, 1), "%")