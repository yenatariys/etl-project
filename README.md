# 🧩 App Review ETL & Sentiment Analysis Pipeline

## 📖 Project Overview
Proyek ini bertujuan untuk membangun **ETL pipeline (Extract, Transform, Load)** yang memproses **ulasan pengguna aplikasi dari Play Store dan App Store**, kemudian melakukan **analisis sentimen otomatis** menggunakan **lexicon-based method** dan **machine learning (SVM)**.

Pipeline ini dirancang agar proses berjalan otomatis — mulai dari scraping data, cleaning, analisis sentimen, hingga menyimpan hasil akhir ke dalam **database PostgreSQL**.

---

## 🚀 Project Workflow

Berikut alur utama dari pipeline:

1. **Extract**  
   - Mengambil data ulasan dari Play Store / App Store (file: `extract_scraper.py`)
   - Hasilnya disimpan dalam format `.csv`

2. **Transform & Clean**  
   - Translasi bahasa ulasan (jika masih berbahasa asing)
   - Text preprocessing: lowercasing, punctuation removal, tokenization, stopword removal, dan stemming (file: `transform_clean.py`)
   - Pelabelan sentimen otomatis menggunakan **InSet Lexicon** (positive & negative word dictionary)

3. **Analyze & Visualize**  
   - Analisis statistik deskriptif dari data ulasan (file: `analyze_statistics.py`)
   - Visualisasi distribusi sentimen, frekuensi kata, dan pola ulasan (file: `visualize_data.py`)

4. **Model Training (Machine Learning)**  
   - Training model **Support Vector Machine (SVM)** untuk klasifikasi sentimen berdasarkan dataset yang sudah dilabeli (file: `train_svm_model.py`)

5. **Load to Database**  
   - Menyimpan hasil akhir ke **PostgreSQL database** menggunakan `SQLAlchemy` (file: `load_to_sql.py`)

6. **Pipeline Orchestrator**  
   - Semua langkah di atas dijalankan secara otomatis melalui file utama:  
     👉 `pipeline.py`

---

## 🧱 Project Structure
```
APP REVIEW ETL Project/
│
├── ETL/
│ ├── extract_scraper.py
│ ├── transform_clean.py
│ ├── analyze_statistics.py
│ ├── visualize_data.py
│ ├── train_svm_model.py
│ ├── load_to_sql.py
│ └── pipeline.py
│
├── data/
│ ├── review_play_combined.csv
│ ├── review_play_cleaned.csv
│ └── review_play_with_sentiment.csv
│
├── lexicon/
│ ├── positive.txt
│ └── negative.txt
│
├── requirements.txt
└── README.md
```

---

## 🧰 Tech Stack & Libraries

| Layer | Tools / Libraries |
|-------|--------------------|
| **Extract** | `google-play-scraper` |
| **Transform** | `pandas`, `googletrans`, `nltk`, `Sastrawi` |
| **Analyze & Visualize** | `matplotlib`, `seaborn`, `wordcloud` |
| **Machine Learning** | `scikit-learn`, `numpy` |
| **Load** | `SQLAlchemy`, `psycopg2`, `PostgreSQL` |
| **Automation** | `subprocess`, `os` |

---

## ⚙️ How to Run

1. **Clone repository**
   ```bash
   git clone https://github.com/yenatariys/etl-project.git
   cd etl-project/ETL
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Jalankan PostgreSQL (Via Docker)**
   ```bash
   docker run -d --name postgres-db \
   -e POSTGRES_USER=myuser \
   -e POSTGRES_PASSWORD=mypassword \
   -e POSTGRES_DB=mydatabase \
   -p 5432:5432 postgres:15
4. **Run Pipeline Utama**
   ```bash
   python pipeline.py
5. **Cek Hasil**
   - File hasil akhir (file: `review_play_with_sentiment.csv`)
   - Tabel hasil di PostgreSQL: `app_reviews`

💡 Key Learnings
- Mendesain alur ETL otomatis dari raw data hingga database.
- Penerapan text preprocessing pipeline yang lengkap untuk Bahasa Indonesia.
- Penggunaan InSet Lexicon dan SVM untuk klasifikasi sentimen.
- Mengintegrasikan hasil analisis dengan PostgreSQL melalui SQLAlchemy.
- Membangun pipeline terotomasi yang mudah dijalankan dengan satu perintah.
