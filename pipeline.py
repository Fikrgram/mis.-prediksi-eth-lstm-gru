# =============================================================================
# PIPELINE INTI: preprocessing, windowing, model LSTM/GRU, training, evaluasi
# =============================================================================
# Modul ini murni logika (tidak bergantung pada Flask maupun path file apa pun).
# Dipakai bersama oleh app.py (endpoint upload CSV) dan train_model.py (skrip
# CLI offline untuk reproduksibilitas skripsi), sehingga kedua jalur pemakaian
# menjalankan pipeline yang PERSIS SAMA — tidak ada logika yang terduplikasi
# dan berpotensi menyimpang (drift) satu sama lain.

import io
import re
import warnings as _std_warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import config


class DatasetError(ValueError):
    """Error yang pesannya aman ditampilkan langsung ke pengguna (validasi input)."""
    pass


def set_seed(seed: int = config.SEED):
    """
    Reset seed NumPy & TensorFlow. Dipanggil ulang tepat sebelum build+train
    MASING-MASING model (LSTM lalu GRU) agar kondisi inisialisasi bobot dan
    operasi acak (mis. Dropout) tereproduksi secara konsisten untuk kedua
    arsitektur — prasyarat agar perbandingan performa bersifat adil.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =============================================================================
# BAGIAN 1: PEMUATAN & VALIDASI DATASET (UNIVARIATE — HANYA CLOSE PRICE)
# =============================================================================

def _find_column(columns, candidates_exact, must_contain):
    """Cari nama kolom: cocok persis (case-insensitive) dulu, baru substring."""
    lower_map = {c.lower().strip(): c for c in columns}

    for cand in candidates_exact:
        if cand in lower_map:
            return lower_map[cand]

    for lower_name, original in lower_map.items():
        if any(token in lower_name for token in must_contain):
            return original

    return None


def _clean_price_value(raw) -> float:
    """Bersihkan satu nilai harga: hapus 'Rp', pemisah ribuan, spasi -> float."""
    s = str(raw).strip()
    s = re.sub(r"(?i)rp", "", s).strip()
    # Buang pemisah ribuan (koma) yang diikuti tepat 3 digit lalu non-digit/akhir
    s = re.sub(r",(?=\d{3}(\D|$))", "", s)
    s = re.sub(r"[^\d.\-]", "", s)
    if s in ("", "-", "."):
        return np.nan
    return float(s)


# Menangkap 3 kelompok angka pertama pada sebuah string tanggal, mis. "05-06-2024"
# -> ("05", "06", "2024"). Tidak match kalau ada nama bulan tekstual (mis. "Jan"),
# karena nama bulan tidak lolos pola \d.
_NUMERIC_DATE_RE = re.compile(r"^\s*(\d{1,4})\D+(\d{1,4})\D+(\d{1,4})")


def _infer_dayfirst(date_strings: pd.Series) -> tuple:
    """
    Menentukan apakah tanggal numerik pada dataset berformat dayfirst
    (DD-MM-YYYY, mis. "05-06-2024" = 5 Juni) atau monthfirst/format Amerika
    (MM-DD-YYYY, mis. "05-06-2024" = 6 Mei), berdasarkan BUKTI pada data itu
    sendiri — bukan tebakan buta seperti sebelumnya (dayfirst=True selalu).

    Kenapa ini penting: kalau asumsi dayfirst/monthfirst salah, seluruh urutan
    kronologis dataset akan salah TANPA memunculkan error apa pun (tanggal tetap
    berhasil diparsing, hanya urutannya yang keliru) — merusak validitas seluruh
    hasil training & evaluasi tanpa disadari.

    Returns:
        (dayfirst: bool, certain: bool). `certain=False` berarti tidak ada
        bukti pasti dari data (harus diberi peringatan eksplisit ke pengguna).
    """
    sample = date_strings.dropna().astype(str)
    tokens = sample.str.extract(_NUMERIC_DATE_RE)

    if tokens[0].isna().all():
        # Tidak ada pola numerik murni ditemukan (mis. format "31-Jan-26" dengan
        # nama bulan tekstual) -> tidak ada ambiguitas dayfirst/monthfirst sama
        # sekali, pandas selalu menafsirkannya benar dari nama bulannya.
        return True, True

    valid = tokens.dropna()
    first = valid[0].astype(int)
    second = valid[1].astype(int)

    if (first > 12).any():
        return True, True    # token pertama > 12 -> pasti hari -> format dayfirst
    if (second > 12).any():
        return False, True   # token kedua > 12 -> pasti hari -> format monthfirst (Amerika)

    # Kedua token selalu <= 12 -> tidak bisa dipastikan dari nilai angka saja.
    # Tie-break: pilih interpretasi yang membuat URUTAN ASLI baris pada file
    # (sebelum kita sortir sendiri) monoton naik/turun — asumsi wajar karena
    # data historis harga biasanya sudah terurut kronologis saat diekspor.
    with _std_warnings.catch_warnings():
        _std_warnings.simplefilter("ignore", UserWarning)
        parsed_true = pd.to_datetime(date_strings, format="mixed", errors="coerce", dayfirst=True)
        parsed_false = pd.to_datetime(date_strings, format="mixed", errors="coerce", dayfirst=False)

    s_true = parsed_true.dropna()
    s_false = parsed_false.dropna()
    mono_true = s_true.is_monotonic_increasing or s_true.is_monotonic_decreasing
    mono_false = s_false.is_monotonic_increasing or s_false.is_monotonic_decreasing

    if mono_true and not mono_false:
        return True, True
    if mono_false and not mono_true:
        return False, True

    return True, False   # benar-benar ambigu -> fallback dayfirst, tandai tidak pasti


def load_close_dataframe(file_obj) -> tuple:
    """
    Membaca dataset CSV yang diunggah pengguna menggunakan pandas, lalu
    memvalidasi dan mengekstrak HANYA kolom Close Price (univariate).

    Args:
        file_obj: path string ATAU file-like object (mis. request.files['dataset']).

    Returns:
        (df, warnings) dengan df berindex tanggal (atau RangeIndex bila kolom
        tanggal tidak ditemukan) dan satu kolom 'Close', terurut kronologis
        (lama -> baru).

    Raises:
        DatasetError: bila kolom Close tidak ditemukan atau data tidak valid.
    """
    warnings = []

    try:
        raw = pd.read_csv(file_obj)
    except Exception as exc:
        raise DatasetError(
            f"Gagal membaca file sebagai CSV. Pastikan format file benar. Detail: {exc}"
        )

    if raw.empty or raw.shape[1] == 0:
        raise DatasetError("File CSV kosong atau tidak memiliki kolom apa pun.")

    close_col = _find_column(
        raw.columns,
        candidates_exact=("close", "close price", "close_price"),
        must_contain=("close",),
    )
    if close_col is None:
        raise DatasetError(
            "Kolom 'Close' atau 'Close Price' tidak ditemukan pada file CSV. "
            f"Kolom yang terdeteksi: {list(raw.columns)}. "
            "Pastikan dataset memiliki kolom harga penutupan bernama 'Close' "
            "atau 'Close Price'."
        )

    date_col = _find_column(
        raw.columns,
        candidates_exact=("date", "tanggal"),
        must_contain=("date", "tanggal"),
    )

    # Ambil HANYA kolom Close (+ tanggal bila ada) — abaikan Open/High/Low/
    # Volume/Adj Close/kolom lain apa pun agar penelitian tetap univariate.
    if date_col is not None:
        df = raw[[date_col, close_col]].copy()
        df.columns = ["Date", "Close"]
    else:
        df = raw[[close_col]].copy()
        df.columns = ["Close"]
        warnings.append(
            "Kolom tanggal tidak ditemukan pada CSV. Sistem mengasumsikan "
            "baris data SUDAH terurut kronologis (lama ke baru) sesuai urutan "
            "pada file. Jika tidak, hasil prediksi tidak akan valid."
        )

    # Bersihkan kolom Close menjadi numerik
    df["Close"] = df["Close"].apply(_clean_price_value)

    if date_col is not None:
        dayfirst, certain = _infer_dayfirst(df["Date"])
        with _std_warnings.catch_warnings():
            _std_warnings.simplefilter("ignore", UserWarning)
            parsed_dates = pd.to_datetime(df["Date"], format="mixed", errors="coerce", dayfirst=dayfirst)
        n_failed = parsed_dates.isna().sum()
        if n_failed > 0.2 * len(df):
            raise DatasetError(
                "Kolom tanggal terdeteksi tetapi sebagian besar nilainya gagal "
                "diparsing sebagai tanggal yang valid. Periksa kembali format "
                "kolom tanggal pada CSV Anda."
            )

        if not certain:
            asumsi = "hari-bulan-tahun (DD-MM-YYYY)" if dayfirst else "bulan-hari-tahun (MM-DD-YYYY)"
            warnings.append(
                f"Format tanggal pada kolom '{date_col}' bersifat AMBIGU (mis. '05-06-2024' "
                "bisa berarti 5 Juni ATAU 6 Mei) dan tidak bisa dipastikan dari data. Sistem "
                f"mengasumsikan format {asumsi}. Jika asumsi ini salah, urutan kronologis "
                "dataset akan keliru dan seluruh hasil training tidak valid — periksa kembali "
                "format tanggal pada file CSV Anda."
            )

        n_duplicate_dates = int(parsed_dates.dropna().duplicated().sum())
        if n_duplicate_dates > 0:
            warnings.append(
                f"Ditemukan {n_duplicate_dates} tanggal duplikat pada dataset. Baris duplikat "
                "tetap dipakai sesuai urutan aslinya pada file — periksa apakah ini disengaja."
            )

        df["Date"] = parsed_dates
        df.dropna(subset=["Date", "Close"], inplace=True)
        df.sort_values(by="Date", inplace=True)
        df.set_index("Date", inplace=True)

        if len(df.index) > 2:
            diffs = df.index.to_series().diff().dropna()
            if not diffs.empty:
                mode_diff = diffs.mode().iloc[0]
                gap_count = int((diffs != mode_diff).sum())
                if gap_count > 0:
                    warnings.append(
                        f"Terdeteksi {gap_count} interval tanggal yang tidak konsisten dengan "
                        f"frekuensi paling umum ({mode_diff.days} hari) pada dataset. Kemungkinan "
                        "ada tanggal yang hilang (gap) pada data historis — periksa kembali jika "
                        "di luar dugaan, karena ini memengaruhi asumsi deret waktu berurutan."
                    )
    else:
        df.dropna(subset=["Close"], inplace=True)
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
        df.index.name = "Date"

    df = df[["Close"]]

    if df.shape[1] != 1:
        # Guard eksplisit: penelitian ini bersifat univariate (hanya Close Price).
        raise DatasetError("Terjadi kesalahan internal: dataset harus univariate (hanya Close).")

    if len(df) < config.MIN_ROWS_REQUIRED:
        raise DatasetError(
            f"Dataset terlalu sedikit ({len(df)} baris valid). Dibutuhkan minimal "
            f"{config.MIN_ROWS_REQUIRED} baris agar pembagian train/validation/test "
            f"dengan lookback window={config.LOOKBACK} dapat dilakukan dengan benar."
        )

    return df, warnings


# =============================================================================
# BAGIAN 2: SPLIT, NORMALISASI, DAN PEMBUATAN SEKUENS
# =============================================================================

def split_data(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """Membagi data secara kronologis (tanpa shuffle) menjadi Train/Val/Test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    # Val dan Test masing-masing HARUS >= lookback baris. Alasannya: preprocess_pipeline
    # meminjam `lookback` baris terakhir dari Train (untuk Val) dan dari Val (untuk Test)
    # agar sekuens pertama tiap subset tetap punya window penuh. Kalau Val/Test lebih
    # pendek dari lookback, peminjaman itu diam-diam terpotong (slicing Python tidak
    # error walau jumlahnya kurang) -- akibatnya jumlah sampel sekuens yang dihasilkan
    # JADI LEBIH SEDIKIT dari len(df_test)/len(df_val), sehingga tidak lagi selaras
    # dengan tanggal (df_test.index) yang dipakai untuk visualisasi. Validasi ini
    # mencegah korupsi data yang senyap tersebut sejak awal.
    if (len(df_train) <= config.LOOKBACK or len(df_val) < config.LOOKBACK
            or len(df_test) < config.LOOKBACK):
        raise DatasetError(
            "Dataset tidak cukup besar untuk menghasilkan sampel Train/Validation/"
            f"Test yang valid dengan lookback={config.LOOKBACK}. Val dan Test masing-masing "
            f"harus berisi minimal {config.LOOKBACK} baris (saat ini Train={len(df_train)}, "
            f"Val={len(df_val)}, Test={len(df_test)} baris). Tambahkan lebih banyak data "
            "historis, atau perkecil nilai LOOKBACK di config.py."
        )

    return df_train, df_val, df_test


def fit_scaler(df_train: pd.DataFrame) -> MinMaxScaler:
    """MinMaxScaler di-fit HANYA pada data Train untuk mencegah data leakage."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train.values)
    return scaler


def create_sequences(data: np.ndarray, lookback: int):
    """
    Ubah array 1D (sudah dinormalisasi) menjadi pasangan (X, y) bergaya
    sliding-window: X = lookback langkah waktu, y = nilai langkah berikutnya.
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def preprocess_pipeline(df: pd.DataFrame, lookback: int, train_ratio: float, val_ratio: float,
                        scaler: MinMaxScaler = None):
    """
    Pipeline preprocessing lengkap: split kronologis -> normalisasi -> pembuatan
    sekuens (X, y) untuk Train/Val/Test.

    Args:
        scaler: bila None (default, jalur TRAINING), scaler baru di-fit HANYA pada
            data Train (cegah leakage). Bila diberikan (jalur PREDIKSI memakai
            model tersimpan), scaler MATANG itu dipakai apa adanya -- tidak di-fit
            ulang -- agar normalisasi input konsisten persis dengan saat model
            dilatih; refit di sini justru akan merusak kecocokan skala.

    Returns dict berisi X/y setiap subset, scaler, dan index tanggal test
    (selaras dengan y_test, dipakai untuk visualisasi).
    """
    df_train, df_val, df_test = split_data(df, train_ratio, val_ratio)

    if scaler is None:
        scaler = fit_scaler(df_train)
    train_scaled = scaler.transform(df_train.values)
    val_scaled = scaler.transform(df_val.values)
    test_scaled = scaler.transform(df_test.values)

    # Val/Test butuh `lookback` titik tambahan di awal (dipinjam dari akhir
    # subset sebelumnya) agar tidak kehilangan sampel — bukan data leakage,
    # karena scaler tetap hanya di-fit pada Train.
    val_with_lookback = np.concatenate(
        [scaler.transform(df_train.values[-lookback:]), val_scaled], axis=0
    )
    test_with_lookback = np.concatenate(
        [scaler.transform(df_val.values[-lookback:]), test_scaled], axis=0
    )

    X_train, y_train = create_sequences(train_scaled, lookback)
    X_val, y_val = create_sequences(val_with_lookback, lookback)
    X_test, y_test = create_sequences(test_with_lookback, lookback)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "test_dates": df_test.index,
    }


# =============================================================================
# BAGIAN 3: ARSITEKTUR MODEL
# =============================================================================

def _build_stacked_rnn(recurrent_layer, lookback, units_l1, units_l2, dropout_rate,
                        learning_rate, name):
    """Arsitektur 2-layer stacked RNN (dipakai identik untuk LSTM dan GRU)."""
    model = Sequential(name=name)
    model.add(recurrent_layer(units=units_l1, return_sequences=True,
                               input_shape=(lookback, 1), name=f"{name}_Layer_1"))
    model.add(Dropout(rate=dropout_rate, name="Dropout_1"))
    model.add(recurrent_layer(units=units_l2, return_sequences=False,
                               name=f"{name}_Layer_2"))
    model.add(Dropout(rate=dropout_rate, name="Dropout_2"))
    model.add(Dense(units=1, activation="linear", name="Output_Layer"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model


def build_lstm_model(lookback, units_l1=config.UNITS_LAYER_1, units_l2=config.UNITS_LAYER_2,
                      dropout_rate=config.DROPOUT_RATE, learning_rate=config.LEARNING_RATE):
    return _build_stacked_rnn(LSTM, lookback, units_l1, units_l2, dropout_rate,
                               learning_rate, name="LSTM_Model")


def build_gru_model(lookback, units_l1=config.UNITS_LAYER_1, units_l2=config.UNITS_LAYER_2,
                     dropout_rate=config.DROPOUT_RATE, learning_rate=config.LEARNING_RATE):
    return _build_stacked_rnn(GRU, lookback, units_l1, units_l2, dropout_rate,
                               learning_rate, name="GRU_Model")


# =============================================================================
# BAGIAN 4: TRAINING
# =============================================================================

def get_callbacks(es_patience: int, lr_patience: int) -> list:
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=es_patience,
        restore_best_weights=True, verbose=0,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=lr_patience,
        min_lr=1e-6, verbose=0,
    )
    return [early_stopping, reduce_lr]


class EpochReporter(tf.keras.callbacks.Callback):
    """
    Callback Keras yang melaporkan loss & val_loss di akhir SETIAP epoch ke
    sebuah fungsi `epoch_cb`. Inilah yang memungkinkan proses pelatihan dipantau
    secara langsung (live) seperti di Google Colab -- baik dialirkan ke browser
    lewat Server-Sent Events (app.py) maupun dicetak ke terminal (train_model.py).
    """

    def __init__(self, epoch_cb, model_name: str, run_idx: int, n_runs: int,
                 max_epochs: int):
        super().__init__()
        self.epoch_cb = epoch_cb
        self.model_name = model_name
        self.run_idx = run_idx
        self.n_runs = n_runs
        self.max_epochs = max_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_cb({
            "type": "epoch",
            "model": self.model_name,
            "run": self.run_idx,
            "n_runs": self.n_runs,
            "epoch": int(epoch) + 1,
            "max_epochs": self.max_epochs,
            "loss": float(logs.get("loss", float("nan"))),
            "val_loss": float(logs.get("val_loss", float("nan"))),
            "lr": float(logs.get("learning_rate", logs.get("lr", 0.0)) or 0.0),
        })


def train_model(model, X_train, y_train, X_val, y_val,
                 epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                 es_patience=config.ES_PATIENCE, lr_patience=config.LR_PATIENCE,
                 verbose=0, extra_callbacks=None):
    callbacks = get_callbacks(es_patience, lr_patience)
    if extra_callbacks:
        callbacks = callbacks + list(extra_callbacks)
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=verbose,
    )


# =============================================================================
# BAGIAN 5: EVALUASI (RMSE, MAE, MAPE) — SETELAH INVERSE TRANSFORM
# =============================================================================

def calculate_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray,
                       scaler: MinMaxScaler) -> dict:
    """Denormalisasi prediksi & aktual ke skala harga asli, lalu hitung RMSE/MAE/MAPE."""
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    epsilon = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "y_true": y_true.flatten(),
        "y_pred": y_pred.flatten(),
    }


def simpulkan_model_terbaik(metrics_lstm: dict, metrics_gru: dict) -> dict:
    """Kesimpulan model terbaik per metrik dan keseluruhan (voting mayoritas 3 metrik)."""
    hasil = {}
    skor_lstm = 0
    skor_gru = 0

    for metrik in ("RMSE", "MAE", "MAPE"):
        nilai_lstm = metrics_lstm[metrik]
        nilai_gru = metrics_gru[metrik]

        if nilai_lstm < nilai_gru:
            pemenang = "LSTM"
            skor_lstm += 1
        elif nilai_gru < nilai_lstm:
            pemenang = "GRU"
            skor_gru += 1
        else:
            pemenang = "Seri"

        hasil[metrik] = pemenang

    if skor_lstm > skor_gru:
        kesimpulan_akhir = "LSTM"
    elif skor_gru > skor_lstm:
        kesimpulan_akhir = "GRU"
    else:
        kesimpulan_akhir = "Seri"

    hasil["kesimpulan_akhir"] = kesimpulan_akhir
    hasil["skor_lstm"] = skor_lstm
    hasil["skor_gru"] = skor_gru
    return hasil


# =============================================================================
# BAGIAN 6: SATU RUN TRAINING+EVALUASI (LSTM & GRU, SEED TERTENTU)
# =============================================================================

def _train_and_evaluate_once(prep: dict, seed: int, run_idx: int = 1,
                              n_runs: int = 1, epoch_cb=None) -> dict:
    """Latih LSTM & GRU sekali (dengan seed tertentu) dan evaluasi di data test."""
    def _cbs(model_name):
        if epoch_cb is None:
            return None
        return [EpochReporter(epoch_cb, model_name, run_idx, n_runs, config.EPOCHS)]

    set_seed(seed)
    model_lstm = build_lstm_model(config.LOOKBACK)
    history_lstm = train_model(model_lstm, prep["X_train"], prep["y_train"],
                                prep["X_val"], prep["y_val"],
                                extra_callbacks=_cbs("LSTM"))

    set_seed(seed)
    model_gru = build_gru_model(config.LOOKBACK)
    history_gru = train_model(model_gru, prep["X_train"], prep["y_train"],
                               prep["X_val"], prep["y_val"],
                               extra_callbacks=_cbs("GRU"))

    scaler = prep["scaler"]
    y_pred_lstm_scaled = model_lstm.predict(prep["X_test"], verbose=0).flatten()
    y_pred_gru_scaled = model_gru.predict(prep["X_test"], verbose=0).flatten()

    metrics_lstm = calculate_metrics(prep["y_test"], y_pred_lstm_scaled, scaler)
    metrics_gru = calculate_metrics(prep["y_test"], y_pred_gru_scaled, scaler)

    return {
        "model_lstm": model_lstm, "model_gru": model_gru,
        "history_lstm": history_lstm, "history_gru": history_gru,
        "metrics_lstm": metrics_lstm, "metrics_gru": metrics_gru,
    }


def _aggregate(values: list) -> dict:
    """Ringkas satu daftar nilai metrik dari N run menjadi mean/std/min/max."""
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# =============================================================================
# BAGIAN 7: ORKESTRASI END-TO-END MULTI-RUN (dipakai oleh app.py & train_model.py)
# =============================================================================

def run_full_pipeline(file_obj, n_runs: int = None, progress_cb=None,
                       epoch_cb=None) -> dict:
    """
    Menjalankan seluruh alur: baca CSV -> preprocessing -> (training LSTM ->
    training GRU -> evaluasi) diulang N_RUNS kali dengan seed berbeda -> agregasi
    mean/std -> kesimpulan model terbaik berdasarkan rata-rata metrik.

    Kenapa diulang N kali: TensorFlow di CPU tidak sepenuhnya deterministik
    (operasi paralel oneDNN), sehingga satu kali training saja tidak cukup
    meyakinkan secara statistik untuk mengklaim "model A lebih baik dari B".
    Mengulang training dengan seed berbeda dan melaporkan mean +/- std membuat
    perbandingan lebih kredibel (lihat [[config.N_RUNS]]).

    Args:
        file_obj    : path atau file-like object berisi CSV yang diunggah user.
        n_runs      : jumlah pengulangan training per model. Default: config.N_RUNS.
        progress_cb : callable opsional (str) -> None untuk melaporkan progres fase.
        epoch_cb    : callable opsional (dict) -> None yang dipanggil di akhir tiap
                      epoch (untuk pemantauan pelatihan live ala Colab).

    Returns:
        dict hasil lengkap siap diserialisasi ke JSON oleh pemanggil.
    """
    if n_runs is None:
        n_runs = config.N_RUNS
    n_runs = max(1, int(n_runs))

    def report(msg):
        if progress_cb:
            progress_cb(msg)

    report("Membaca dan memvalidasi dataset...")
    df, warnings = load_close_dataframe(file_obj)

    report("Melakukan preprocessing dan pembuatan sekuens...")
    prep = preprocess_pipeline(df, config.LOOKBACK, config.TRAIN_RATIO, config.VAL_RATIO)

    rmse_lstm, mae_lstm, mape_lstm, epochs_lstm = [], [], [], []
    rmse_gru, mae_gru, mape_gru, epochs_gru = [], [], [], []

    # Run TERBAIK dipilih berdasarkan val_loss (validasi) TERENDAH — lihat blok
    # pemilihan di dalam loop untuk alasan metodologisnya.
    best_lstm = None
    best_gru = None

    for i in range(n_runs):
        seed = config.SEED + i
        report(f"Menjalankan run {i + 1}/{n_runs} (LSTM lalu GRU, seed={seed})...")
        run = _train_and_evaluate_once(prep, seed, run_idx=i + 1, n_runs=n_runs,
                                        epoch_cb=epoch_cb)

        rmse_lstm.append(run["metrics_lstm"]["RMSE"])
        mae_lstm.append(run["metrics_lstm"]["MAE"])
        mape_lstm.append(run["metrics_lstm"]["MAPE"])
        epochs_lstm.append(len(run["history_lstm"].history["loss"]))

        rmse_gru.append(run["metrics_gru"]["RMSE"])
        mae_gru.append(run["metrics_gru"]["MAE"])
        mape_gru.append(run["metrics_gru"]["MAPE"])
        epochs_gru.append(len(run["history_gru"].history["loss"]))

        # Kriteria pemilihan model TERBAIK: val_loss (VALIDASI) TERENDAH, BUKAN
        # RMSE pada data TEST. Memilih model berdasarkan performa di data test
        # adalah kebocoran seleksi (model selection tidak boleh "mengintip" data
        # uji) -- itu membuat metrik test yang dilaporkan bias optimistis. Data
        # test dipakai HANYA untuk mengevaluasi, tidak untuk memilih. val_loss
        # terbaik sebuah run = nilai minimum sepanjang epoch, yaitu tepat bobot
        # yang direstore oleh EarlyStopping(restore_best_weights=True).
        val_lstm = float(min(run["history_lstm"].history["val_loss"]))
        val_gru = float(min(run["history_gru"].history["val_loss"]))

        if best_lstm is None or val_lstm < best_lstm["val_loss"]:
            best_lstm = {"model": run["model_lstm"], "history": run["history_lstm"],
                         "metrics": run["metrics_lstm"], "val_loss": val_lstm,
                         "epochs": len(run["history_lstm"].history["loss"])}
        if best_gru is None or val_gru < best_gru["val_loss"]:
            best_gru = {"model": run["model_gru"], "history": run["history_gru"],
                        "metrics": run["metrics_gru"], "val_loss": val_gru,
                        "epochs": len(run["history_gru"].history["loss"])}

    report("Menyusun agregasi statistik dan kesimpulan model terbaik...")
    metrics_summary = {
        "lstm": {"rmse": _aggregate(rmse_lstm), "mae": _aggregate(mae_lstm), "mape": _aggregate(mape_lstm)},
        "gru": {"rmse": _aggregate(rmse_gru), "mae": _aggregate(mae_gru), "mape": _aggregate(mape_gru)},
    }

    mean_metrics_lstm = {
        "RMSE": metrics_summary["lstm"]["rmse"]["mean"],
        "MAE": metrics_summary["lstm"]["mae"]["mean"],
        "MAPE": metrics_summary["lstm"]["mape"]["mean"],
    }
    mean_metrics_gru = {
        "RMSE": metrics_summary["gru"]["rmse"]["mean"],
        "MAE": metrics_summary["gru"]["mae"]["mean"],
        "MAPE": metrics_summary["gru"]["mape"]["mean"],
    }
    kesimpulan = simpulkan_model_terbaik(mean_metrics_lstm, mean_metrics_gru)

    per_run = {
        "lstm": [{"run": i + 1, "rmse": rmse_lstm[i], "mae": mae_lstm[i],
                  "mape": mape_lstm[i], "epochs": epochs_lstm[i]} for i in range(n_runs)],
        "gru": [{"run": i + 1, "rmse": rmse_gru[i], "mae": mae_gru[i],
                 "mape": mape_gru[i], "epochs": epochs_gru[i]} for i in range(n_runs)],
    }

    test_dates = prep["test_dates"]
    if isinstance(test_dates, pd.DatetimeIndex):
        date_labels = [d.strftime("%Y-%m-%d") for d in test_dates]
    else:
        date_labels = [f"Hari ke-{i + 1}" for i in range(len(test_dates))]

    # Grafik Actual vs Prediction memakai run TERBAIK (val_loss validasi terendah)
    # per arsitektur -- konsisten dengan model yang disimpan sebagai artefak di bawah.
    return {
        "warnings": warnings,
        "n_runs": n_runs,
        # Info run terpilih (dipakai untuk metadata model produksi & tampilan web).
        # Epoch di sini = total epoch run terbaik; val_loss = validasi terbaiknya.
        "best_run_epochs": {"lstm": best_lstm["epochs"], "gru": best_gru["epochs"]},
        "best_run_val_loss": {"lstm": best_lstm["val_loss"], "gru": best_gru["val_loss"]},
        "dataset_info": {
            "total_rows": int(len(df)),
            "train_rows": int(len(prep["df_train"])),
            "val_rows": int(len(prep["df_val"])),
            "test_rows": int(len(prep["df_test"])),
        },
        "config": {
            "lookback": config.LOOKBACK,
            "train_ratio": config.TRAIN_RATIO,
            "val_ratio": config.VAL_RATIO,
            "n_runs": n_runs,
        },
        "metrics": metrics_summary,
        "per_run": per_run,
        "conclusion": kesimpulan,
        "chart_data": {
            "labels": date_labels,
            "actual": best_lstm["metrics"]["y_true"].tolist(),
            "pred_lstm": best_lstm["metrics"]["y_pred"].tolist(),
            "pred_gru": best_gru["metrics"]["y_pred"].tolist(),
        },
        # Objek non-JSON, dipakai opsional oleh caller (mis. train_model.py CLI)
        "_artifacts": {
            "model_lstm": best_lstm["model"],
            "model_gru": best_gru["model"],
            "scaler": prep["scaler"],
            "history_lstm": best_lstm["history"],
            "history_gru": best_gru["history"],
            "metrics_lstm": best_lstm["metrics"],
            "metrics_gru": best_gru["metrics"],
            "df_test": prep["df_test"],
        },
    }


# =============================================================================
# BAGIAN 8: PREDIKSI CEPAT MEMAKAI MODEL TERSIMPAN (dipakai oleh app.py / web)
# =============================================================================

def _single_stat(value: float) -> dict:
    """Bungkus satu nilai metrik ke bentuk {mean/std/min/max} agar SELARAS dengan
    struktur hasil multi-run (frontend memakai bentuk yang sama). std=0 karena
    hanya ada satu model (bukan rata-rata dari beberapa run)."""
    v = float(value)
    return {"mean": v, "std": 0.0, "min": v, "max": v}


def run_predict_pipeline(file_obj, model_lstm, model_gru, scaler,
                         progress_cb=None, best_epochs: dict = None) -> dict:
    """
    Jalur PREDIKSI CEPAT: memakai model LSTM & GRU yang SUDAH matang (dilatih
    offline oleh train_model.py dan dimuat dari output_model/best_model/), TANPA
    melatih ulang. Karena tidak ada training, prosesnya hitungan detik.

    Bentuk dict yang dikembalikan dibuat SAMA PERSIS dengan run_full_pipeline
    (metrics/per_run/conclusion/chart_data) sehingga frontend (index.html) tidak
    perlu diubah. Karena hanya ada satu model per arsitektur, n_runs=1 dan std=0.

    Catatan validitas: scaler yang dipakai adalah scaler MATANG bawaan model.
    Evaluasi akan selaras dengan hasil training offline bila CSV yang diunggah
    adalah dataset yang sama dengan saat model dilatih.

    Args:
        file_obj     : path atau file-like object berisi CSV yang diunggah user.
        model_lstm   : model Keras LSTM yang sudah dimuat (compile tidak wajib).
        model_gru    : model Keras GRU yang sudah dimuat.
        scaler       : MinMaxScaler matang yang disimpan bersama model.
        progress_cb  : callable opsional (str) -> None untuk melaporkan fase.
        best_epochs  : dict opsional {"lstm": int, "gru": int} — jumlah epoch saat
                       model dilatih dulu (hanya untuk info di panel hasil).
    """
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    best_epochs = best_epochs or {}

    report("Membaca dan memvalidasi dataset...")
    df, warnings = load_close_dataframe(file_obj)

    report("Preprocessing dengan scaler matang bawaan model (tanpa fit ulang)...")
    prep = preprocess_pipeline(df, config.LOOKBACK, config.TRAIN_RATIO,
                               config.VAL_RATIO, scaler=scaler)

    report("Memprediksi data test dengan model LSTM & GRU tersimpan...")
    y_pred_lstm_scaled = model_lstm.predict(prep["X_test"], verbose=0).flatten()
    y_pred_gru_scaled = model_gru.predict(prep["X_test"], verbose=0).flatten()

    metrics_lstm = calculate_metrics(prep["y_test"], y_pred_lstm_scaled, scaler)
    metrics_gru = calculate_metrics(prep["y_test"], y_pred_gru_scaled, scaler)

    report("Menghitung metrik dan kesimpulan model terbaik...")
    kesimpulan = simpulkan_model_terbaik(metrics_lstm, metrics_gru)

    metrics_summary = {
        "lstm": {"rmse": _single_stat(metrics_lstm["RMSE"]),
                 "mae": _single_stat(metrics_lstm["MAE"]),
                 "mape": _single_stat(metrics_lstm["MAPE"])},
        "gru": {"rmse": _single_stat(metrics_gru["RMSE"]),
                "mae": _single_stat(metrics_gru["MAE"]),
                "mape": _single_stat(metrics_gru["MAPE"])},
    }

    per_run = {
        "lstm": [{"run": 1, "rmse": metrics_lstm["RMSE"], "mae": metrics_lstm["MAE"],
                  "mape": metrics_lstm["MAPE"], "epochs": int(best_epochs.get("lstm", 0))}],
        "gru": [{"run": 1, "rmse": metrics_gru["RMSE"], "mae": metrics_gru["MAE"],
                 "mape": metrics_gru["MAPE"], "epochs": int(best_epochs.get("gru", 0))}],
    }

    test_dates = prep["test_dates"]
    if isinstance(test_dates, pd.DatetimeIndex):
        date_labels = [d.strftime("%Y-%m-%d") for d in test_dates]
    else:
        date_labels = [f"Hari ke-{i + 1}" for i in range(len(test_dates))]

    return {
        "mode": "predict",
        "warnings": warnings,
        "n_runs": 1,
        "dataset_info": {
            "total_rows": int(len(df)),
            "train_rows": int(len(prep["df_train"])),
            "val_rows": int(len(prep["df_val"])),
            "test_rows": int(len(prep["df_test"])),
        },
        "config": {
            "lookback": config.LOOKBACK,
            "train_ratio": config.TRAIN_RATIO,
            "val_ratio": config.VAL_RATIO,
            "n_runs": 1,
        },
        "metrics": metrics_summary,
        "per_run": per_run,
        "conclusion": kesimpulan,
        "chart_data": {
            "labels": date_labels,
            "actual": metrics_lstm["y_true"].tolist(),
            "pred_lstm": metrics_lstm["y_pred"].tolist(),
            "pred_gru": metrics_gru["y_pred"].tolist(),
        },
    }
