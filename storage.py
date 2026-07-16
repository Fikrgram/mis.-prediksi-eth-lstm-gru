# =============================================================================
# PENYIMPANAN ARTEFAK PER-RUN (TIDAK PERNAH MENIMPA HASIL SEBELUMNYA)
# =============================================================================
# Setiap kali training selesai (baik lewat upload web maupun CLI), artefaknya
# (model .h5, scaler .pkl, plot, ringkasan) disimpan ke folder BARU dengan nama
# unik (timestamp + jumlah run) -- bukan ke nama file tetap seperti sebelumnya.
# Ini menjamin hasil training lama TIDAK PERNAH tertimpa oleh training baru,
# termasuk saat demo cepat (N kecil) dijalankan setelah training resmi (N besar).

import os
import json
import shutil
import datetime

import config


def make_run_dir(base_dir: str, n_runs: int) -> str:
    """
    Buat folder unik untuk satu run training. Nama folder mengandung timestamp
    dan jumlah run (mis. "run_20260707_233900_n10"). Kalau nama itu somehow
    sudah ada (tabrakan timestamp), ditambahkan suffix angka agar tetap unik.
    """
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}_n{n_runs}"
    run_dir = os.path.join(base_dir, base_name)

    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, f"{base_name}_{suffix}")
        suffix += 1

    os.makedirs(run_dir)
    return run_dir


def save_run_summary(run_dir: str, result: dict) -> None:
    """Simpan ringkasan hasil (metrics, conclusion, dataset_info, dst) sebagai summary.json."""
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def save_training_log(run_dir: str, result: dict, artifacts: dict) -> None:
    """
    Simpan log naratif lengkap proses training ke training_log.txt -- jejak
    audit untuk menjawab "darimana tahu model ini yang terbaik?" (mis. saat
    sidang skripsi): info dataset, peringatan validasi, rincian tiap run,
    ringkasan mean +/- std, kesimpulan, dan kurva loss epoch-per-epoch dari
    model LSTM & GRU terbaik (yang disimpan sebagai model_lstm.h5/model_gru.h5).
    """
    n_runs = result["n_runs"]
    cfg = result["config"]
    info = result["dataset_info"]
    conclusion = result["conclusion"]
    test_ratio = 1 - cfg["train_ratio"] - cfg["val_ratio"]

    lines = []
    lines.append("=" * 70)
    lines.append("LOG PROSES PELATIHAN & EVALUASI MODEL LSTM vs GRU")
    lines.append("=" * 70)
    lines.append(f"Tanggal/waktu   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Jumlah run      : {n_runs}")
    lines.append(f"Lookback window : {cfg['lookback']} hari")
    lines.append(f"Split data      : Train {cfg['train_ratio'] * 100:.0f}% / "
                  f"Validasi {cfg['val_ratio'] * 100:.0f}% / Test {test_ratio * 100:.0f}%")
    lines.append("")

    lines.append("-" * 70)
    lines.append("INFORMASI DATASET")
    lines.append("-" * 70)
    lines.append(f"Total baris     : {info['total_rows']}")
    lines.append(f"Data Train      : {info['train_rows']} baris")
    lines.append(f"Data Validasi   : {info['val_rows']} baris")
    lines.append(f"Data Test       : {info['test_rows']} baris")
    lines.append("")

    lines.append("-" * 70)
    lines.append("PERINGATAN VALIDASI DATASET")
    lines.append("-" * 70)
    if result["warnings"]:
        for w in result["warnings"]:
            lines.append(f"  - {w}")
    else:
        lines.append("  Tidak ada peringatan.")
    lines.append("")

    lines.append("-" * 70)
    lines.append("RINCIAN TIAP RUN (seed berbeda tiap run)")
    lines.append("-" * 70)
    lines.append(f"{'Run':<5}{'Model':<8}{'RMSE':>16}{'MAE':>16}{'MAPE (%)':>12}{'Epoch':>8}")
    for i in range(n_runs):
        l = result["per_run"]["lstm"][i]
        g = result["per_run"]["gru"][i]
        lines.append(f"{l['run']:<5}{'LSTM':<8}{l['rmse']:>16,.2f}{l['mae']:>16,.2f}"
                      f"{l['mape']:>12.4f}{l['epochs']:>8}")
        lines.append(f"{g['run']:<5}{'GRU':<8}{g['rmse']:>16,.2f}{g['mae']:>16,.2f}"
                      f"{g['mape']:>12.4f}{g['epochs']:>8}")
    lines.append("")

    lines.append("-" * 70)
    lines.append(f"RINGKASAN STATISTIK (MEAN +/- STD DARI {n_runs} RUN)")
    lines.append("-" * 70)
    for name in ("lstm", "gru"):
        m = result["metrics"][name]
        lines.append(f"[{name.upper()}] RMSE = {m['rmse']['mean']:,.2f} +/- {m['rmse']['std']:,.2f}   "
                      f"MAE = {m['mae']['mean']:,.2f} +/- {m['mae']['std']:,.2f}   "
                      f"MAPE = {m['mape']['mean']:.4f}% +/- {m['mape']['std']:.4f}%")
    lines.append("")

    lines.append("-" * 70)
    lines.append("KESIMPULAN MODEL TERBAIK (VOTING MAYORITAS 3 METRIK, BERDASARKAN MEAN)")
    lines.append("-" * 70)
    lines.append(f"RMSE terbaik      : {conclusion['RMSE']}")
    lines.append(f"MAE terbaik       : {conclusion['MAE']}")
    lines.append(f"MAPE terbaik      : {conclusion['MAPE']}")
    lines.append(f"Skor LSTM menang  : {conclusion['skor_lstm']} dari 3 metrik")
    lines.append(f"Skor GRU menang   : {conclusion['skor_gru']} dari 3 metrik")
    lines.append(f"KESIMPULAN AKHIR  : {conclusion['kesimpulan_akhir']}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("EPOCH TERBAIK MODEL YANG DISIMPAN (RINGKAS)")
    lines.append("-" * 70)
    lines.append("EarlyStopping memakai restore_best_weights=True: bobot yang DISIMPAN")
    lines.append("adalah bobot pada epoch dengan val_loss TERENDAH, BUKAN epoch terakhir.")
    lines.append(f"Training tetap lanjut {config.ES_PATIENCE} epoch (patience) setelah titik terbaik")
    lines.append("sebagai konfirmasi tidak ada perbaikan lagi, lalu berhenti.")
    lines.append("")
    best_epoch = {}
    for name in ("lstm", "gru"):
        val = artifacts[f"history_{name}"].history["val_loss"]
        best_idx = min(range(len(val)), key=lambda i: val[i])  # epoch val_loss terendah
        best_epoch[name] = best_idx
        lines.append(f"[{name.upper()}]  Total epoch dijalankan: {len(val):>3}   |   "
                      f"EPOCH TERBAIK (disimpan): {best_idx + 1:>3}   |   "
                      f"val_loss terbaik: {val[best_idx]:.6f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("KURVA TRAINING MODEL TERBAIK (RUN DENGAN val_loss VALIDASI TERENDAH PER ARSITEKTUR)")
    lines.append("-" * 70)
    for name in ("lstm", "gru"):
        history = artifacts[f"history_{name}"].history
        lines.append(f"[{name.upper()}] -- model yang disimpan sebagai model_{name}.h5")
        lines.append(f"{'Epoch':<8}{'loss':>14}{'val_loss':>14}")
        for epoch_idx in range(len(history["loss"])):
            marker = "   << EPOCH TERBAIK (bobot ini yang disimpan)" if epoch_idx == best_epoch[name] else ""
            lines.append(f"{epoch_idx + 1:<8}{history['loss'][epoch_idx]:>14.6f}"
                          f"{history['val_loss'][epoch_idx]:>14.6f}{marker}")
        lines.append("")

    log_path = os.path.join(run_dir, "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_training_history(run_dir: str, artifacts: dict) -> None:
    """
    Simpan kurva loss/val_loss LENGKAP per-epoch (LSTM & GRU) sebagai JSON
    terstruktur -- versi mesin-terbaca dari tabel yang sama di training_log.txt,
    untuk menjawab pertanyaan "di epoch berapa model ini dikatakan terbaik?"
    tanpa perlu mem-parsing teks log. Ikut dipromosikan ke best_model/ oleh
    [[promote_best_model]] agar menempel ke model produksi yang aktif dipakai.
    """
    history_json = {}
    for name in ("lstm", "gru"):
        history = artifacts[f"history_{name}"].history
        loss = history["loss"]
        val_loss = history["val_loss"]
        best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])

        history_json[name] = {
            "total_epochs": len(loss),
            "best_epoch": best_idx + 1,
            "best_val_loss": float(val_loss[best_idx]),
            "epochs": [
                {
                    "epoch": i + 1,
                    "loss": float(loss[i]),
                    "val_loss": float(val_loss[i]),
                    "is_best": i == best_idx,
                }
                for i in range(len(loss))
            ],
        }

    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=2, ensure_ascii=False)


# =============================================================================
# MODEL PRODUKSI PERMANEN — PROMOSI (SIMPAN) & PEMUATAN (LOAD)
# =============================================================================
# Web (app.py) tidak melatih model saat upload; ia hanya MEMUAT model matang
# yang sudah dilatih offline oleh train_model.py. Agar web tahu model mana yang
# harus dipakai, train_model.py "mempromosikan" model terbaik ke satu folder
# tetap (output_model/best_model/). Fungsi di bawah menangani salin ke sana dan
# pemuatannya kembali.

def best_model_dir(base_dir: str) -> str:
    """Path folder model produksi permanen (output_model/best_model/)."""
    return os.path.join(base_dir, config.BEST_MODEL_DIRNAME)


def _best_run_epochs(per_run_list: list) -> int:
    """FALLBACK (untuk hasil lama tanpa 'best_run_epochs'): perkiraan jumlah epoch
    run terbaik. Sumber utama epoch run terpilih adalah result['best_run_epochs']
    yang dihitung pipeline berdasarkan val_loss; fungsi ini hanya dipakai bila
    field itu tidak ada."""
    if not per_run_list:
        return 0
    best = min(per_run_list, key=lambda r: r["rmse"])
    return int(best.get("epochs", 0))


def promote_best_model(base_dir: str, run_dir: str, result: dict) -> str:
    """
    Salin model TERBAIK dari folder run arsip (run_dir) ke folder produksi
    permanen (output_model/best_model/), MENIMPA model produksi sebelumnya.

    Ini sengaja menimpa: folder produksi selalu berisi SATU model matang terkini
    yang dipakai web. Arsip run unik (run_YYYYMMDD_...) tetap utuh dan tidak
    pernah tertimpa, sehingga jejak setiap pelatihan untuk skripsi tetap ada.

    Menulis juga metadata.json (kapan dipromosikan, dari run mana, metrik, dan
    kesimpulan) sebagai bukti asal-usul model produksi saat sidang.

    Returns: path folder produksi.
    """
    dest = best_model_dir(base_dir)
    os.makedirs(dest, exist_ok=True)

    for filename in ("model_lstm.h5", "model_gru.h5", "scaler.pkl"):
        src = os.path.join(run_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Artefak '{filename}' tidak ditemukan di {run_dir}; promosi model "
                "produksi dibatalkan."
            )
        shutil.copy2(src, os.path.join(dest, filename))

    # training_history.json bersifat opsional (hasil run lama sebelum fitur ini
    # ada belum memilikinya) -- salin bila ada, jangan gagalkan promosi bila tidak.
    history_src = os.path.join(run_dir, "training_history.json")
    if os.path.exists(history_src):
        shutil.copy2(history_src, os.path.join(dest, "training_history.json"))

    metadata = {
        "promoted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_run": os.path.basename(run_dir.rstrip(os.sep)),
        "n_runs": result.get("n_runs"),
        "config": result.get("config"),
        "dataset_info": result.get("dataset_info"),
        "metrics": result.get("metrics"),
        "conclusion": result.get("conclusion"),
        # Total epoch run terbaik tiap arsitektur (model inilah yang disimpan) —
        # dipakai web untuk info "epoch" di panel hasil. Sumber: pipeline memilih
        # run terbaik via val_loss (validasi), lalu mencatat epochnya di sini.
        "best_epochs": result.get("best_run_epochs") or {
            "lstm": _best_run_epochs(result.get("per_run", {}).get("lstm", [])),
            "gru": _best_run_epochs(result.get("per_run", {}).get("gru", [])),
        },
        # val_loss validasi terbaik run terpilih — dasar pemilihan model (bukan
        # RMSE test), sebagai bukti metodologi bebas kebocoran saat sidang.
        "best_run_val_loss": result.get("best_run_val_loss"),
        # Epoch dengan val_loss TERENDAH (bobot yang benar-benar disimpan oleh
        # restore_best_weights) — bukti "model terbaik di epoch mana" untuk sidang.
        "best_saved_epoch": result.get("best_saved_epoch"),
    }
    with open(os.path.join(dest, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return dest


def load_best_model(base_dir: str):
    """
    Muat model produksi permanen (LSTM, GRU, scaler, metadata) dari
    output_model/best_model/. Impor TensorFlow/joblib dilakukan di dalam fungsi
    agar modul storage tetap ringan bila fungsi ini tidak dipakai.

    Returns:
        dict {model_lstm, model_gru, scaler, metadata, dir} bila lengkap,
        atau None bila folder/model belum ada (mis. train_model.py belum pernah
        dijalankan).
    """
    d = best_model_dir(base_dir)
    paths = {name: os.path.join(d, fn) for name, fn in (
        ("model_lstm", "model_lstm.h5"),
        ("model_gru", "model_gru.h5"),
        ("scaler", "scaler.pkl"),
    )}
    if not all(os.path.exists(p) for p in paths.values()):
        return None

    import joblib
    from tensorflow.keras.models import load_model

    # compile=False: kita hanya memprediksi, tidak melatih, sehingga optimizer/
    # loss tidak perlu direkonstruksi (pemuatan lebih cepat & bebas peringatan).
    model_lstm = load_model(paths["model_lstm"], compile=False)
    model_gru = load_model(paths["model_gru"], compile=False)
    scaler = joblib.load(paths["scaler"])

    metadata = None
    meta_path = os.path.join(d, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = None

    return {
        "model_lstm": model_lstm,
        "model_gru": model_gru,
        "scaler": scaler,
        "metadata": metadata,
        "dir": d,
    }
