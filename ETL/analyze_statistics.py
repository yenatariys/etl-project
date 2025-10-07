import pandas as pd

# Baca hasil preprocessing terakhir
df = pd.read_csv('review_play_with_sentiment.csv')

print("=== 📊 ANALISIS STATISTIK DATA ===\n")

# 1️⃣ Statistik umum
print("Jumlah total ulasan:", len(df))
print("\nStatistik skor sentimen:")
print(df['sentiment_score'].describe())

# 2️⃣ Distribusi kategori sentimen
print("\nDistribusi kategori sentimen:")
print(df['sentiment_category'].value_counts())

# 3️⃣ Distribusi rating prediksi
print("\nDistribusi rating prediksi:")
print(df['predicted_rating'].value_counts())

# 4️⃣ Panjang rata-rata ulasan
df['panjang_ulasan'] = df['ulasan_bersih'].apply(lambda x: len(str(x).split()))
print("\nPanjang ulasan (kata):")
print(df['panjang_ulasan'].describe())

# Simpan hasil statistik ke file
df[['ulasan_bersih', 'sentiment_score', 'sentiment_category', 'predicted_rating', 'panjang_ulasan']].to_csv(
    'review_play_stats.csv', index=False
)

print("\n✅ Statistik berhasil dihitung dan disimpan ke 'review_play_stats.csv'")
