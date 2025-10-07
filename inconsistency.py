# Cek ulasan yang inkonsisten
inconsistent_reviews = df[df['predicted_rating'] != df['rating']]
print("Jumlah Ulasan Inkonsisten:", len(inconsistent_reviews))

# Tampilkan beberapa ulasan inkonsisten
print(inconsistent_reviews[['ulasan_bersih', 'rating', 'predicted_rating', 'sentiment_score']].head())
inconsistent_reviews.to_csv('inconsistent_reviews.csv', index=False)