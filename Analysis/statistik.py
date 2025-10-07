# Hitung jumlah kata di setiap ulasan
df['word_count'] = df['ulasan_bersih'].apply(lambda x: len(str(x).split()))

stat_summary = df['word_count'].describe()
print("Statistik Deskriptif Jumlah Kata per Ulasan:")
print(stat_summary)

# Tampilkan histogram distribusi jumlah kata
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Jumlah Kata per Ulasan')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Simpan visualisasi
plt.savefig('word_count_distribution.png')

