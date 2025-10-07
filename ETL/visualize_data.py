# visualize_data.py

import pandas as pd
from sqlalchemy import create_engine
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ KONEKSI KE DATABASE
# ==============================
db_user = "myuser"
db_password = "mypassword"
db_host = "localhost"
db_port = "5432"
db_name = "mydatabase"

engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Ambil data dari tabel cleaned_reviews
query = "SELECT * FROM cleaned_reviews;"
df = pd.read_sql(query, engine)
print(f"Jumlah data dari database: {len(df)}")

# ==============================
# 2️⃣ DISTRIBUSI SENTIMEN
# ==============================
sentiment_categories = df['sentiment_category'].value_counts()
print("\nDistribusi Kategori Sentimen:")
print(sentiment_categories)

plt.figure(figsize=(6, 4))
sentiment_categories.plot(kind='bar', color=['red', 'blue', 'green'])
plt.title('Distribusi Kategori Sentimen')
plt.xlabel('Kategori Sentimen')
plt.ylabel('Jumlah Ulasan')
plt.tight_layout()
plt.savefig('sentiment_category_distribution.png')
plt.show()

# ==============================
# 3️⃣ WORDCLOUD UNTUK SETIAP SENTIMEN
# ==============================
def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# WordCloud Positif
positive_reviews = ' '.join(df[df['sentiment_category'] == 'Positif']['ulasan_bersih'])
generate_wordcloud(positive_reviews, 'WordCloud Ulasan Positif', 'wordcloud_positive.png')

# WordCloud Negatif
negative_reviews = ' '.join(df[df['sentiment_category'] == 'Negatif']['ulasan_bersih'])
generate_wordcloud(negative_reviews, 'WordCloud Ulasan Negatif', 'wordcloud_negative.png')

# WordCloud Netral
neutral_reviews = ' '.join(df[df['sentiment_category'] == 'Netral']['ulasan_bersih'])
generate_wordcloud(neutral_reviews, 'WordCloud Ulasan Netral', 'wordcloud_neutral.png')

# ==============================
# 4️⃣ DISTRIBUSI RATING ASLI VS PREDIKSI (JIKA ADA)
# ==============================
if 'predicted_rating' in df.columns:
    plt.figure(figsize=(8, 5))
    df['rating'].value_counts().sort_index().plot(kind='bar', color='orange', alpha=0.6, position=0, width=0.4, label='Rating Asli')
    df['predicted_rating'].value_counts().sort_index().plot(kind='bar', color='purple', alpha=0.6, position=1, width=0.4, label='Rating Prediksi')
    plt.title('Distribusi Rating Asli vs Prediksi')
    plt.xlabel('Rating Bintang')
    plt.ylabel('Jumlah Ulasan')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rating_distribution_comparison.png')
    plt.show()
else:
    print("\nKolom 'predicted_rating' belum ada — lewati visualisasi perbandingan rating.")

# ==============================
# 5️⃣ DISTRIBUSI SENTIMEN PER TAHUN (JIKA ADA KOLOM TANGGAL)
# ==============================
if 'date' in df.columns:
    df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    sentiment_by_year = df.groupby(['year', 'sentiment_category']).size().unstack(fill_value=0)
    sentiment_by_year.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Distribusi Kategori Sentimen per Tahun')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Ulasan')
    plt.legend(title='Kategori Sentimen')
    plt.tight_layout()
    plt.savefig('sentiment_distribution_by_year.png')
    plt.show()
else:
    print("\nKolom 'date' tidak ditemukan — lewati visualisasi per tahun.")