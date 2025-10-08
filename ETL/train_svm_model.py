import pandas as pd
import numpy as np
df = pd.read_csv('data/review_play_with_sentiment.csv')

# == TF-IDF ==

# Data Preparation
from sklearn.model_selection import train_test_split
# Feature
X = df['ulasan_bersih']

# Target
y_tfidf = df['sentiment_category']
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X, y_tfidf, test_size=0.2, random_state=42
)

print("TF-IDF Data Split:")
print(f"X_train_tfidf: {X_train_tfidf.shape}, X_test_tfidf: {X_test_tfidf.shape}")
print(f"y_train_tfidf: {y_train_tfidf.shape}, y_test_tfidf: {y_test_tfidf.shape}")

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

X_train_tfidf = X_train_tfidf.astype(str)
X_test_tfidf = X_test_tfidf.astype(str)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_tfidf)
X_test_tfidf = tfidf_vectorizer.transform(X_test_tfidf)

print("TF-IDF Shape:")
print("X_train_tfidf shape: {X_train_tfidf.shape}")
print("X_test_tfidf shape: {X_test_tfidf.shape}")

# Model Training
from sklearn.svm import SVC
svm_tfidf = SVC(kernel='linear', random_state=42)
svm_tfidf.fit(X_train_tfidf, y_train_tfidf)
print("SVM (TF-IDF) Model Trained")

# Prediction
y_pred_tfidf = svm_tfidf.predict(X_test_tfidf)
print("First 10 Predictions (TF-IDF):")
print(y_pred_tfidf[:10])

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report (TF-IDF):")
print(classification_report(y_test_tfidf, y_pred_tfidf))

import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure()
cm = confusion_matrix(y_test_tfidf, y_pred_tfidf, labels=svm_tfidf.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_tfidf.classes_, yticklabels=svm_tfidf.classes_)
plt.title("Confusion Matrix - TF-IDF")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# Save Confusion Matrix
fig.savefig("Confusion_Matrix_TFIDF.png", dpi=300)
plt.show()


# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf_smote, y_train_tfidf_smote = smote.fit_resample(X_train_tfidf, y_train_tfidf)
print("Original Training Set Size (TF-IDF):", X_train_tfidf.shape)
print("Resampled Training Set Size (TF-IDF):", X_train_tfidf_smote.shape)

# Model Training with SMOTE
svm_tfidf_smote = SVC(kernel='linear', random_state=42)
svm_tfidf_smote.fit(X_train_tfidf_smote, y_train_tfidf_smote)
print("SVM (TF-IDF + SMOTE) Model Trained")
# Prediction with SMOTE
y_pred_tfidf_smote = svm_tfidf_smote.predict(X_test_tfidf)
print("First 10 Predictions (TF-IDF + SMOTE):")
print(y_pred_tfidf_smote[:10])
# Evaluation with SMOTE
print("Classification Report (TF-IDF + SMOTE):")
print(classification_report(y_test_tfidf, y_pred_tfidf_smote))
fig = plt.figure()
cm = confusion_matrix(y_test_tfidf, y_pred_tfidf_smote, labels=svm_tfidf_smote.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_tfidf_smote.classes_, yticklabels=svm_tfidf_smote.classes_)
plt.title("Confusion Matrix - TF-IDF + SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# Save Confusion Matrix
fig.savefig("Confusion_matrix_tfidf_smote.png", dpi=300)
plt.show()



# Comparison (TF-IDF)
print("TF-IDF Without SMOTE")
print(classification_report(y_test_tfidf, y_pred_tfidf))
print("TF-IDF With SMOTE")
print(classification_report(y_test_tfidf, y_pred_tfidf_smote))
# End of TF-IDF

# == IndoBERT Embedding ==
import torch
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from tqdm import tqdm

model_name = "indobenchmark/indobert-base-p1"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

import torch

# Pilih Device (GPU jika tersedia)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
print(f"Using device: {device}")

# Fungsi untuk mendapatkan embedding dari teks
def get_bert_embedding(text):
    inputs = tokenizer(
        text,
        padding = True,
        truncation = True,
        return_tensors= "pt",
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Ambil embedding dari [CLS] token
    embedding = outputs.last_hidden_state[:,0,:].squeeze().cpu().numpy()
    return embedding

# Ambil fitur (X) dan target (y)
X = df['ulasan_bersih'].astype(str).tolist()
y_bert = df['sentiment_category']

# Split data menjadi training dan testing
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
    X, y_bert, test_size=0.2, random_state=42
)

# Fungsi Evaluasi
def evaluate_model(model, X_test, y_test, labels, title=None):
    y_pred = model.predict(X_test)
    if title:
        print(f"=== {title} ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Ubah teks jadi embedding IndoBERT
X_train_bert_embed = np.array([get_bert_embedding(text) for text in tqdm(X_train_bert, desc="Embedding train")])
X_test_bert_embed = np.array([get_bert_embedding(text) for text in tqdm(X_test_bert, desc="Embedding test")])

# Model Training
svm_bert = SVC(kernel='linear')
svm_bert.fit(X_train_bert_embed, y_train_bert)
print("SVM (IndoBERT) Model Trained")
# Prediction
y_pred_bert = svm_bert.predict(X_test_bert_embed)
print("First 10 Predictions (IndoBERT):")
print(y_pred_bert[:10])
# Evaluation
print("Classification Report (IndoBERT):")
print(classification_report(y_test_bert, y_pred_bert))
fig = plt.figure()
cm = confusion_matrix(y_test_bert, y_pred_bert, labels=svm_bert.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_bert.classes_, yticklabels=svm_bert.classes_)
plt.title("Confusion Matrix - IndoBERT")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# Save Confusion Matrix
fig.savefig("Confusion_Matrix_BERT.png", dpi=300)
plt.show()


# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_bert_smote, y_train_bert_smote = smote.fit_resample(X_train_bert_embed, y_train_bert)
print("Original Training Set Size (IndoBERT):" , X_train_bert_embed.shape)
print("Resampled Training Set Size (IndoBERT):" , X_train_bert_smote.shape)
# Model Training with SMOTE
svm_bert_smote = SVC(kernel='linear', random_state=42)
svm_bert_smote.fit(X_train_bert_smote, y_train_bert_smote)
print("SVM (IndoBERT + SMOTE) Model Trained")
# Prediction with SMOTE
y_pred_bert_smote = svm_bert_smote.predict(X_test_bert_embed)
print("First 10 Predictions (IndoBERT + SMOTE):")
print(y_pred_bert_smote[:10])
# Evaluation with SMOTE
print("Classification Report (IndoBERT + SMOTE):")
print(classification_report(y_test_bert, y_pred_bert_smote))
fig = plt.figure()
cm = confusion_matrix(y_test_bert, y_pred_bert_smote, labels=svm_bert_smote.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_bert_smote.classes_, yticklabels=svm_bert_smote.classes_)
plt.title("Confusion Matrix - IndoBERT + SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# Save Confusion Matrix
fig.savefig("Confusion_matrix_bert_smote.png", dpi=300)
plt.show()


# Comparison (IndoBERT)
print("IndoBERT Without SMOTE")
print(classification_report(y_test_bert, y_pred_bert))
print("IndoBERT With SMOTE")
print(classification_report(y_test_bert, y_pred_bert_smote))
# End of IndoBERT