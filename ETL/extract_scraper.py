from google_play_scraper import Sort, reviews
import pandas as pd
from datetime import datetime
import time

app_id = 'in.startv.hotstar.dplus'

MAX_2020_2022 = 200
MAX_2023_2025 = 200

reviews_2020_2022 = []
reviews_2023_2025 = []

continuation_token = None
count = 0

while True:
    result, continuation_token = reviews(
        app_id,
        lang='id',
        country='id',
        sort=Sort.NEWEST,
        count=200,
        continuation_token=continuation_token
    )

    for r in result:
        review_year=r['at'].year 

        if 2020 <= review_year <= 2022:
            if len(reviews_2020_2022) < MAX_2020_2022:
                reviews_2020_2022.append(r)
        
        elif 2023 <= review_year <=2025:
            if len(reviews_2023_2025) < MAX_2023_2025:
                reviews_2023_2025.append(r)

    print(f"Batch {count} — Total 2020-2022: {len(reviews_2020_2022)}, Total 2023-2025: {len(reviews_2023_2025)}")
    count += 1
    time.sleep(1)

    # Berhenti kalau dua duanya sudah cukup
    if len(reviews_2020_2022) >= MAX_2020_2022 and len(reviews_2023_2025) >= MAX_2023_2025:
        break
    if result and result[-1]['at'].year < 2020:
        break

# Simpan ke CSV
pd.DataFrame(reviews_2020_2022).to_csv("review_play_2020_2022.csv", index=False)
print("Simpan review 2020-2022 ke review_play_2020_2022.csv")
pd.DataFrame(reviews_2023_2025).to_csv("review_play_2023_2025.csv", index=False)
print("Simpan review 2023-2025 ke review_play_2023_2025.csv")

import glob
# Get all CSV files in the folder
csv_files = glob.glob("review_play_*.csv")
# Read and concatenate all CSV files
df_combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
# Save the combined DataFrame to a new CSV
df_combined.to_csv("review_play_combined.csv", index=False)
print("Simpan gabungan review ke review_play_combined.csv")