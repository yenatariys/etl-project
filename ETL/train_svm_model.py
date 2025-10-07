# svm_model.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ==============================
# 1️⃣ KONEKSI KE DATABASE
# ==============================
db_user = "myuser"
db_password = "mypassword"
db_host = "localhost"
db_port = "5432"
db_name = "mydatabase"

engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Ambil data dari tabel yang sudah kamu load
query = "SELECT * FROM app_reviews;"
df = pd.read_sql(query, engine)
print(f"Jumlah data dari database: {len(df)}")

# Pastikan kolom sesuai
print(df.columns)

# ==============================
# 2️⃣ PERSIAPAN DATA
# ==============================
from sklearn.model_selection import train_test_split

X = df['ulasan_bersih']  # sesuaikan nama kolom
y = df['sentiment_category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

# ==============================
# 3️⃣ FEATURE EXTRACTION: TF-IDF
# ==============================
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Dimensi fitur TF-IDF: {X_train_tfidf.shape[1]}")

# ==============================
# 4️⃣ FEATURE EXTRACTION: IndoBERT
# ==============================
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model_bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")

def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

print("Menghasilkan embedding IndoBERT (ini bisa agak lama)...")
X_train_bert = get_bert_embeddings(X_train)
X_test_bert = get_bert_embeddings(X_test)
print(f"Dimensi fitur IndoBERT: {X_train_bert.shape[1]}")

# ==============================
# 5️⃣ TRAINING MODEL (TF-IDF)
# ==============================
from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model SVM (TF-IDF) berhasil dilatih.")

# ==============================
# 6️⃣ EVALUASI MODEL
# ==============================
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test_tfidf)
print("\n📊 Laporan Klasifikasi TF-IDF:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix TF-IDF:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 7️⃣ SIMPAN MODEL
# ==============================
import joblib
joblib.dump(model, 'svm_model_tfidf.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model dan vectorizer berhasil disimpan.")

# ==============================
# 8️⃣ LATIH MODEL DENGAN IndoBERT
# ==============================
model_bert_svm = SVC(kernel='linear', random_state=42)
model_bert_svm.fit(X_train_bert, y_train)
y_pred_bert = model_bert_svm.predict(X_test_bert)

print("\n📊 Laporan Klasifikasi IndoBERT:")
print(classification_report(y_test, y_pred_bert))
print("Confusion Matrix IndoBERT:")
print(confusion_matrix(y_test, y_pred_bert))

joblib.dump(model_bert_svm, 'svm_model_bert.pkl')
print("Model IndoBERT berhasil disimpan.")

# ==============================
# 9️⃣ SMOTE HANDLING
# ==============================
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

X_train_tfidf_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
X_train_bert_smote, y_train_bert_smote = smote.fit_resample(X_train_bert, y_train)
print(f"Jumlah data training setelah SMOTE (TF-IDF): {len(y_train_smote)}")
print(f"Jumlah data training setelah SMOTE (IndoBERT): {len(y_train_bert_smote)}")

# ==============================
# 🔁 LATIH ULANG DENGAN SMOTE
# ==============================
model_smote = SVC(kernel='linear', random_state=42)
model_smote.fit(X_train_tfidf_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test_tfidf)

print("\n📊 Laporan Klasifikasi TF-IDF setelah SMOTE:")
print(classification_report(y_test, y_pred_smote))

joblib.dump(model_smote, 'svm_model_tfidf_smote.pkl')

model_bert_smote = SVC(kernel='linear', random_state=42)
model_bert_smote.fit(X_train_bert_smote, y_train_bert_smote)
y_pred_bert_smote = model_bert_smote.predict(X_test_bert)

print("\n📊 Laporan Klasifikasi IndoBERT setelah SMOTE:")
print(classification_report(y_test, y_pred_bert_smote))

joblib.dump(model_bert_smote, 'svm_model_bert_smote.pkl')
print("✅ Semua model berhasil disimpan.")
