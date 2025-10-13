# visualize_data.py

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

custom_stopwords = {
    'nya', 'aja', 'dong', 'banget', 'sih', 'deh', 'kayak', 'bgt', 'udh', 'udah', 'pas', 'aplikasi',
    'gak', 'ga', 'gk', 'kaga', 'ya', 'lah', 'nih', 'tu', 'aja', 'kok', 'oh', 'dah', 'yg', 'gt', 'oke', 'nya'
}
stop_words = stop_words.union(custom_stopwords)


# Baca file
df = pd.read_csv('data/review_play_with_sentiment.csv')
df['ulasan_bersih'] = df['ulasan_bersih'].fillna('').astype(str)

# ==============================
# DISTRIBUSI SENTIMEN
# ==============================
sentiment_categories = df['sentiment_category'].value_counts()

plt.figure(figsize=(6, 6))

# Total keseluruhan data
total = sentiment_categories.sum()

# Fungsi custom buat nampilin persen + jumlah
def autopct_func(pct):
    count = int(pct * total / 100.0)
    return f'{pct:.1f}%\n({count})'   # tampilkan di dua baris: persen dan jumlah

plt.pie(
    sentiment_categories,
    labels=sentiment_categories.index,
    autopct=autopct_func,
    startangle=90,
    colors=['#e74c3c', '#95a5a6', '#2ecc71'],
    explode=(0.05, 0.05, 0.05)
)

plt.title('Distribusi Sentimen')
plt.tight_layout()
plt.savefig('sentiment_pie_chart.png', dpi=300)
plt.show()

# ==============================
# WORDCLOUD UNTUK SETIAP SENTIMEN
# ==============================

def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=stop_words,
        collocations=False
    ).generate(text)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# WordCloud Positif
positive_reviews = ' '.join(df[df['sentiment_category'] == 'Positif']['ulasan_bersih'])
generate_wordcloud(positive_reviews, 'WordCloud Ulasan Positif', 'wordcloud_positive.png')

# from collections import Counter
# words = positive_reviews.split()
# word_freq = Counter([w for w in words if w not in stop_words])
# print(word_freq.most_common(20))

# WordCloud Negatif
negative_reviews = ' '.join(df[df['sentiment_category'] == 'Negatif']['ulasan_bersih'])
generate_wordcloud(negative_reviews, 'WordCloud Ulasan Negatif', 'wordcloud_negative.png')

# WordCloud Netral
neutral_reviews = ' '.join(df[df['sentiment_category'] == 'Netral']['ulasan_bersih'])
generate_wordcloud(neutral_reviews, 'WordCloud Ulasan Netral', 'wordcloud_neutral.png')

# ==============================
# DISTRIBUSI RATING ASLI VS PREDIKSI (JIKA ADA)
# ==============================
if 'predicted_rating' in df.columns:
    plt.figure(figsize=(8, 5))

    # Ambil index rating (1–5) dan jumlah ulasan per rating
    rating_counts = df['score'].value_counts().sort_index()
    predicted_counts = df['predicted_rating'].value_counts().sort_index()

    # Buat posisi bar
    x = np.arange(len(rating_counts))
    width = 0.4

    # Plot dua bar chart berdampingan
    bars1 = plt.bar(x - width/2, rating_counts, width, color='orange', alpha=0.8, label='Rating Asli')
    bars2 = plt.bar(x + width/2, predicted_counts, width, color='purple', alpha=0.8, label='Rating Prediksi')

    plt.title('Distribusi Rating Asli vs Prediksi')
    plt.xlabel('Rating Bintang')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(x, rating_counts.index)
    plt.legend()
    plt.tight_layout()

    # === Tambahkan label di dalam batang ===
    for bar in bars1:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,  # posisi horizontal di tengah
            height / 2,                       # posisi vertikal di tengah bar
            f"{int(height)}",                 # isi label
            ha='center', va='center', fontsize=9, color='white', fontweight='bold'
        )

    for bar in bars2:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height / 2,
            f"{int(height)}",
            ha='center', va='center', fontsize=9, color='white', fontweight='bold'
        )

    plt.savefig('rating_distribution_comparison.png', dpi=300)
    plt.show()
else:
    print("\nKolom 'predicted_rating' belum ada — lewati visualisasi perbandingan rating.")

# ==============================
# DISTRIBUSI SENTIMEN PER TAHUN (JIKA ADA KOLOM TANGGAL)
# ==============================
if 'at' in df.columns:
    # Konversi kolom 'at' ke datetime dan ambil tahun
    df['year'] = pd.to_datetime(df['at'], errors='coerce').dt.year

    # Hitung jumlah ulasan per tahun dan kategori sentimen
    sentiment_by_year = df.groupby(['year', 'sentiment_category']).size().unstack(fill_value=0)

    # Plot stacked bar chart
    ax = sentiment_by_year.plot(kind='bar', stacked=True, figsize=(12, 6),
                                color=['red', 'green', 'blue'])  # bisa custom warnanya juga

    plt.title('Distribusi Kategori Sentimen per Tahun')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Ulasan')
    plt.legend(title='Kategori Sentimen')
    plt.tight_layout()

    # Tambahkan label jumlah di setiap bar
    for container in ax.containers:
        ax.bar_label(container, label_type='center', color='white', fontsize=9, fontweight='bold')

    plt.savefig('sentiment_distribution_by_year.png', dpi=300)
    plt.show()
else:
    print("\nKolom 'date' tidak ditemukan — lewati visualisasi per tahun.")

# =======================
# Sentiment by Rating
# =======================
import matplotlib.pyplot as plt
import pandas as pd

# Grouping: jumlah ulasan per rating & kategori sentimen
sentiment_by_score = df.groupby(['score', 'sentiment_category']).size().unstack(fill_value=0)

# Plot stacked bar chart
ax = sentiment_by_score.plot(
    kind='bar',
    stacked=True,
    figsize=(8, 5),
    color=['#e74c3c', '#95a5a6', '#2ecc71']  # negatif, netral, positif
)

plt.title('Distribusi Sentimen berdasarkan Rating')
plt.xlabel('Rating (Bintang)')
plt.ylabel('Jumlah Ulasan')
plt.legend(title='Kategori Sentimen')
plt.tight_layout()

# === Tambahkan label di dalam setiap segmen bar ===
for container in ax.containers:
    # Tambah label hanya kalau tinggi bar > 0
    ax.bar_label(
        container,
        fmt='%d',               # format label (angka bulat)
        label_type='center',    # posisi di tengah bar
        color='white',          # warna teks biar kontras
        fontsize=9,
        fontweight='bold'
    )

plt.savefig('sentiment_by_rating_labeled.png', dpi=300)
plt.show()

# =======================
# Sentiment Trend Over Time (Jika ada kolom tanggal)
# =======================
df['date'] = pd.to_datetime(df['at'], errors='coerce')
sentiment_trend = df.groupby([df['date'].dt.to_period('M'), 'sentiment_category']).size().unstack(fill_value=0)

sentiment_trend.plot(kind='line', figsize=(10,5), marker='o')
plt.title('Tren Sentimen per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Ulasan')
plt.legend(title='Kategori Sentimen')
plt.tight_layout()
plt.savefig('sentiment_trend.png', dpi=300)
plt.show()

# =======================
# Top Words per Sentiment
# =======================
from collections import Counter
import seaborn as sns

def plot_top_words(text, title):
    words = text.split()
    word_freq = Counter(words)
    common_words = pd.DataFrame(word_freq.most_common(10), columns=['word', 'count'])
    plt.figure(figsize=(8,5))
    sns.barplot(y='word', x='count', data=common_words, palette='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_top_words(positive_reviews, 'Top 10 Kata Ulasan Positif')
plot_top_words(negative_reviews, 'Top 10 Kata Ulasan Negatif')
plot_top_words(neutral_reviews, 'Top 10 Kata Ulasan Netral')

# =======================
# Confusion Matrix (Jika ada prediksi)
# =======================
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(df['score'], df['predicted_rating'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')
plt.title('Confusion Matrix - Rating Prediction')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()