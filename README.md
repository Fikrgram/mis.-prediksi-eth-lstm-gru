# Analisis Perbandingan Kinerja GRU dan LSTM dalam Prediksi Harga Ethereum

Aplikasi penelitian untuk skripsi berjudul **"Analisis Perbandingan Kinerja Algoritma Gated Recurrent Unit (GRU) dan Long Short-Term Memory (LSTM) dalam Prediksi Harga Ethereum Berbasis Data Time Series"**.

Penulis: **Fikri Hasyim Al Rasyid**

---

## Deskripsi

Sistem membandingkan kinerja dua arsitektur deep learning — **LSTM** dan **GRU** — dalam memprediksi harga penutupan (Close Price) Ethereum berbasis data time series (univariate). Perbandingan diukur dengan **RMSE, MAE, dan MAPE** pada data uji (out-of-sample).

Sistem terdiri dari dua jalur yang berbagi satu pipeline inti yang sama (`pipeline.py`), sehingga hasilnya konsisten:

1. **Pelatihan offline** (`train_model.py`) — melatih LSTM & GRU beberapa kali (multi-run), memilih model terbaik, menyimpan model produksi + plot + tabel + log untuk lampiran skripsi. Boleh memakan waktu lama.
2. **Aplikasi web** (`app.py` + `index.html`) — mengunggah CSV lalu **memuat model matang** dan memprediksi dengan cepat (hitungan detik, tanpa melatih ulang).

## Sorotan metodologi

- **Univariate** — hanya kolom `Close` / `Close Price` yang dipakai.
- **Perbandingan adil** — LSTM & GRU memakai arsitektur, hyperparameter, dan seed identik (`config.py`).
- **Anti-kebocoran** — `MinMaxScaler` di-fit hanya pada data train; pemilihan model terbaik berdasarkan **`val_loss` (validasi)**, bukan performa data test.
- **Multi-run** — training diulang N kali (seed berbeda), hasil dilaporkan sebagai **mean ± std** agar kredibel secara statistik.
- **EarlyStopping** (`restore_best_weights=True`) — bobot yang disimpan adalah bobot pada epoch dengan `val_loss` terendah.

## Prasyarat

- Python 3.12
- Dependensi pada `requirements.txt`

```bash
pip install -r requirements.txt
```

## Cara pakai

### 1. Latih model (sekali, offline)

```bash
python train_model.py --data data_eth_idr.csv --n-runs 10
```

Menyimpan model terbaik ke `output_model/best_model/` (dipakai web) dan arsip lengkap (plot, CSV, log) ke folder `output_model/run_<timestamp>/`.

> Repositori ini sudah menyertakan `output_model/best_model/` hasil pelatihan, jadi langkah ini opsional bila hanya ingin mencoba web.

### 2. Jalankan aplikasi web

```bash
python app.py
```

Buka http://localhost:5000, unggah dataset CSV, lalu tekan **Proses Prediksi**. Sistem memuat model dari `output_model/best_model/` dan menampilkan metrik (RMSE/MAE/MAPE), grafik Actual vs Prediction, serta kesimpulan model terbaik.

## Format dataset

CSV dengan header memuat kolom `Close` atau `Close Price` (kolom Open/High/Low/Volume/Adj Close diabaikan). Harga boleh berformat `"Rp 42,500,000.00"`. Kolom tanggal opsional namun disarankan agar urutan kronologis terjamin benar.

## Struktur proyek

```
├── app.py              # Server Flask: upload CSV -> muat model -> prediksi (SSE)
├── train_model.py      # CLI: pelatihan & evaluasi offline, promosi model produksi
├── pipeline.py         # Pipeline inti: preprocessing, model, training, evaluasi
├── storage.py          # Simpan/muat artefak & model produksi (best_model)
├── config.py           # Konfigurasi global (lookback, split, arsitektur, dll.)
├── jobs.py             # Registri job untuk streaming progres (SSE)
├── index.html          # Antarmuka web
├── data_eth_idr.csv    # Dataset harga Ethereum
├── requirements.txt
└── output_model/
    └── best_model/     # Model produksi (LSTM + GRU + scaler + metadata)
```

## Konfigurasi

Parameter utama ada di `config.py`: `LOOKBACK` (window), rasio split train/val/test, arsitektur (units/dropout/learning rate), `EPOCHS`, `N_RUNS`, dll.
