import pandas as pd
from sqlalchemy import create_engine

# === 1. Konfigurasi koneksi PostgreSQL ===
username = "myuser"         # sesuai dengan docker run
password = "mypassword"
host = "localhost"          # karena PostgreSQL dijalankan di Docker lokal
port = "5432"
database = "mydatabase"

# Buat koneksi ke database PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

# === 2. Membaca data hasil preprocessing ===
csv_file = "data/review_play_with_sentiment.csv"  # hasil dari transform_clean.py
df = pd.read_csv(csv_file)

# === 3. Simpan ke tabel PostgreSQL ===
table_name = "app_reviews"   # nama tabel tujuan
df.to_sql(table_name, engine, if_exists="replace", index=False)

print(f"Data dari '{csv_file}' berhasil dimuat ke tabel '{table_name}' di database '{database}'.")