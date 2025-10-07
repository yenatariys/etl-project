import pandas as pd
import string, re, ast
from googletrans import Translator
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === 1. Load Dataset ===
df = pd.read_csv('review_play_combined.csv')
print("Dataset loaded, jumlah baris:", len(df))
print("Contoh data:\n", df[['content', 'at']].head())

# === 2. Translate ke Bahasa Indonesia ===
translator = Translator()

def translate_to_indo(text):
    try:
        translated = translator.translate(str(text), dest='id')
        return translated.text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text

df['translated_content'] = df['content'].apply(translate_to_indo)
print("Proses Translasi selesai")
print("Contoh hasil translasi:\n", df[['content', 'translated_content']].head())

# === 3. Cleaning ===
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_content'] = df['translated_content'].apply(clean_text)

print("Proses Cleaning Selesai")
print("Contoh hasil cleaning:\n", df[['translated_content', 'cleaned_content']].head())

# === 4. Tokenization ===
nltk.download('punkt')
df['tokenized_content'] = df['cleaned_content'].apply(word_tokenize)
print("Proses Tokenization selesai")
print("Contoh hasil tokenization:\n", df[['cleaned_content', 'tokenized_content']].head())

# === 5. Stopword Removal ===
nltk.download('stopwords')
indonesian_stopwords = set(stopwords.words('indonesian'))
df['stopword_removed_content'] = df['tokenized_content'].apply(
    lambda tokens: [t for t in tokens if t.lower() not in indonesian_stopwords]
)
print("Proses Stopword Removal selesai")
print("Contoh hasil stopword removal:\n", df[['tokenized_content', 'stopword_removed_content']].head())

# === 6. Stemming ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()
df['stemmed_content'] = df['stopword_removed_content'].apply(
    lambda tokens: [stemmer.stem(t) for t in tokens]
)
print("Proses Stemming selesai")
print("Contoh hasil stemming:\n", df[['stopword_removed_content', 'stemmed_content']].head())

# === 7. Lexicon-based Sentiment ===
path_lexicon = './lexicon/InSet-master'
df_positif = pd.read_csv(f'{path_lexicon}/positive.tsv', names=['kata', 'nilai'], sep='\t')
df_negatif = pd.read_csv(f'{path_lexicon}/negative.tsv', names=['kata', 'nilai'], sep='\t')

kamus_lexicon = dict(zip(df_positif['kata'], df_positif['nilai'].astype(int)))
kamus_lexicon.update(dict(zip(df_negatif['kata'], df_negatif['nilai'].astype(int))))
print(f"Kamus lexicon dimuat, total kata: {len(kamus_lexicon)}")

def calculate_sentiment_score(tokens, lexicon):
    score = 0
    for token in tokens:
        score += lexicon.get(token, 0)
    return score

df['sentiment_score'] = df['stemmed_content'].apply(lambda x: calculate_sentiment_score(x, kamus_lexicon))
print("Proses Lexicon-based Sentiment selesai")
print("Contoh hasil sentiment score:\n", df[['stemmed_content', 'sentiment_score']].head())

# === 8. Mapping score ke rating dan kategori ===
print("Mapping sentiment score ke predicted rating dan kategori sentiment...")
def map_sentiment_to_rating(score):
    if score <= -5:
        return 1
    elif -4 <= score <= -1:
        return 2
    elif score == 0:
        return 3
    elif 1 <= score <= 4:
        return 4
    else:
        return 5

df['predicted_rating'] = df['sentiment_score'].apply(map_sentiment_to_rating)
print("Contoh hasil predicted rating:\n", df[['sentiment_score', 'predicted_rating']].head())

print("Mapping predicted rating ke kategori sentiment...")
def categorize_sentiment(rating):
    if rating in [1, 2]:
        return 'Negatif'
    elif rating == 3:
        return 'Netral'
    else:
        return 'Positif'

df['sentiment_category'] = df['predicted_rating'].apply(categorize_sentiment)
print("Contoh hasil kategori sentiment:\n", df[['predicted_rating', 'sentiment_category']].head())

# === 9. Simpan hasil akhir ===
df['ulasan_bersih'] = df['stemmed_content'].apply(lambda x: ' '.join(x))
df.to_csv('review_play_with_sentiment.csv', index=False)
print("✅ Hasil akhir disimpan ke 'review_play_with_sentiment.csv'")