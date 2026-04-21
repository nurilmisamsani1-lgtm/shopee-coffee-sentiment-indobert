# 📊 Shopee Review Scraping & Sentiment Analysis

Panduan sederhana untuk mengambil data review Shopee, membersihkan data, melakukan analisis sentimen, dan melihat hasil dalam dashboard interaktif.

---

# 🧩 Persiapan Awal (Wajib)

## ✅ STEP 1 — Jalankan Chrome Mode Khusus

1. Buka **Command Prompt (CMD)**
2. Jalankan perintah berikut:

```
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=D:\chrome_shopee_profile
```

📌 Chrome akan terbuka dalam mode khusus untuk proses scraping.

---

## ✅ STEP 2 — Install Library (Sekali Saja)

Jalankan:

```
pip install -r requirements.txt
pip install -r project\requirements_eda.txt
```

📌 Tunggu hingga proses selesai.

---

# 📥 Scraping Data Review

## ✅ STEP 3 — Ambil Data Review Shopee

⚠️ Penting:
Pastikan **halaman review sudah terbuka di Chrome** sebelum menjalankan script.

### 🏪 Scraping Review Toko

```
python scrape_shopee_shop_reviews.py
```

### 📦 Scraping Review Produk

```
python scrape_shopee_product_reviews.py
```

📌 Dataset review akan tersimpan otomatis setelah proses selesai.

---

# 🧹 Pengolahan & Analisis Data

## ✅ STEP 4 — Masuk ke Folder Project

```
cd project
```

## ✅ STEP 5 — Jalankan Pipeline Analisis

```
python build_pipeline.py
```

📌 Proses ini akan melakukan:

* Data cleaning (pembersihan data)
* Exploratory Data Analysis (EDA)
* Sentiment analysis (analisis sentimen)

---

# 📊 Dashboard Visualisasi

## ✅ STEP 6 — Jalankan Dashboard

```
streamlit run dashboard_app.py
```

📌 Dashboard akan otomatis terbuka di browser.

Di dashboard Anda dapat melihat:

* Distribusi sentimen review
* Insight dari komentar pelanggan
* Statistik hasil analisis

---

# ⚠️ Tips Penting

✔ Selalu jalankan Chrome menggunakan command pada STEP 1
✔ Pastikan tab review sudah terbuka sebelum scraping
✔ Jangan menutup Chrome saat proses scraping berlangsung
✔ Jika terjadi error → ulangi dari STEP 1

---

# 🎯 Output yang Akan Didapat

Setelah seluruh proses selesai, Anda akan mendapatkan:

* Dataset review Shopee
* Hasil analisis sentimen otomatis
* Dashboard visual interaktif untuk insight bisnis

---

# 🆘 Jika Mengalami Kendala

Coba langkah berikut:

1. Tutup semua Chrome
2. Jalankan ulang STEP 1
3. Jalankan ulang script

---

✨ Sistem ini membantu memahami opini pelanggan secara otomatis.
