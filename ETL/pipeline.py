import os
import subprocess

def run_script(script_name, log_file):
    """
    Menjalankan file .py lain secara berurutan.
    """
    print(f"\n Menjalankan {script_name} ...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)

    # Simpan log ke file
    log_file.write(f"\n=== {script_name} ===\n")
    log_file.write(result.stdout)
    if result.stderr:
        log_file.write(result.stderr)
    log_file.flush()

    # Cek hasil eksekusi
    if result.returncode == 0:
        print(f"✅ {script_name} selesai dijalankan dengan sukses.\n")
    else:
        print(f"Terjadi kesalahan saat menjalankan {script_name}.\n")
        print(result.stderr)
        exit(1)  # hentikan pipeline jika ada error


def main():
    print("=== 🧩 PIPELINE SENTIMEN ANALISIS ULASAN ===\n")

    steps = [
        "extract_scraper.py",      # 1️⃣ Scraping dari Play Store / App Store
        "transform_clean.py",      # 2️⃣ Cleaning, tokenizing, lexicon labeling
        "analyze_statistics.py",   # 3️⃣ Statistik deskriptif
        "visualize_data.py",       # 4️⃣ Visualisasi hasil
        "train_svm_model.py",      # 5️⃣ Training model SVM
        "load_to_sql.py"           # Muat ke PostgreSQL
    ]

    with open("pipeline_log.txt", "a", encoding="utf-8") as log_file:
        for step in steps:
            if os.path.exists(step):
                run_script(step, log_file)
            else:
                print(f"File {step} tidak ditemukan, dilewati.\n")
                log_file.write(f"File {step} tidak ditemukan.\n")

    print("\n Semua tahap pipeline selesai dijalankan!\n")
    print("Log tersimpan di: pipeline_log.txt\n")


if __name__ == "__main__":
    main()