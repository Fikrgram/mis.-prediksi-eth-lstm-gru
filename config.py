# =============================================================================
# KONFIGURASI GLOBAL — dipakai bersama oleh pipeline.py, app.py, train_model.py
# =============================================================================
# Satu sumber konfigurasi agar LSTM dan GRU selalu dilatih dengan parameter
# yang identik (window size, rasio split, arsitektur, hyperparameter training),
# sehingga perbandingan performa kedua model bersifat adil (apple-to-apple).

SEED = 42

# --- Preprocessing ---
LOOKBACK    = 30     # Window size (hari historis sebagai input)
TRAIN_RATIO = 0.70   # 70% data untuk training
VAL_RATIO   = 0.15   # 15% data untuk validasi
# Sisa 15% otomatis menjadi data test

# --- Arsitektur Model (identik untuk LSTM & GRU) ---
UNITS_LAYER_1 = 64
UNITS_LAYER_2 = 32
DROPOUT_RATE  = 0.2
LEARNING_RATE = 0.001

# --- Training ---
EPOCHS      = 200    # Maksimum epoch (EarlyStopping menghentikan lebih awal)
BATCH_SIZE  = 32
ES_PATIENCE = 10
LR_PATIENCE = 5

# --- Jumlah baris minimum: pemeriksaan cepat di awal untuk menolak dataset yang
# jelas terlalu kecil. Batas yang SEBENARNYA mengikat (Val dan Test masing-masing
# >= LOOKBACK baris, lihat pipeline.split_data) dihitung dinamis dari LOOKBACK dan
# rasio split di atas -- dengan nilai default ini kira-kira 200 baris.
MIN_ROWS_REQUIRED = 80

# --- Repetisi Training (multi-run) ---
# Setiap model dilatih ulang N_RUNS kali (seed berbeda tiap run) agar hasil
# perbandingan dilaporkan sebagai rata-rata +/- std, bukan angka dari satu kali
# training saja -- non-determinism TensorFlow di CPU membuat satu run tunggal
# tidak cukup meyakinkan secara statistik untuk klaim "model A lebih baik".
N_RUNS = 10

# --- Model Produksi Permanen (dipakai oleh web upload untuk prediksi cepat) ---
# Setelah train_model.py memilih model TERBAIK (val_loss validasi terendah lintas run), model
# itu "dipromosikan" (disalin) ke satu folder tetap bernama di bawah ini, di
# dalam output_model/. Web (app.py) SELALU memuat model dari folder ini lalu
# hanya memprediksi (tanpa training), sehingga upload dataset jadi cepat
# (hitungan detik). Folder ini SATU-SATUNYA yang boleh tertimpa saat promosi --
# arsip run unik (run_YYYYMMDD_...) tetap tidak pernah tertimpa.
BEST_MODEL_DIRNAME = "best_model"
