# =============================================================================
# BACKEND API: UPLOAD CSV -> MUAT MODEL MATANG -> PREDIKSI CEPAT -> HASIL
# =============================================================================
# Deskripsi:
#   Server Flask yang menerima dataset CSV dari halaman index.html. Berbeda dari
#   versi sebelumnya yang MELATIH ulang tiap upload (lama), server ini hanya
#   MEMUAT model LSTM & GRU matang yang sudah dilatih offline lalu MEMPREDIKSI
#   -- prosesnya hitungan detik.
#
#   Model matang dihasilkan sekali (boleh lama) dengan menjalankan:
#       python train_model.py --data <dataset>.csv
#   yang menyimpan model terbaik secara permanen ke output_model/best_model/.
#   Web SELALU memuat model dari folder itu.
#
#   Prediksi berjalan di THREAD LATAR sementara progres fase dialirkan ke browser
#   via Server-Sent Events (SSE).
#
# Alur:
#   POST /api/process         -> mulai job prediksi, balas { job_id }
#   GET  /api/stream/<job_id> -> aliran event SSE: phase | done | error
#
# Dependensi:
#   pip install flask flask-cors tensorflow scikit-learn pandas numpy joblib
#
# Cara Menjalankan:
#   1) python train_model.py --data data_eth_idr.csv   (sekali, melatih model)
#   2) python app.py                                    (buka http://localhost:5000)
# =============================================================================

import io
import os
import json
import logging
import threading
import traceback

from flask import Flask, request, jsonify, send_from_directory, Response

import config
import jobs
import storage
from pipeline import run_predict_pipeline, DatasetError

# =============================================================================
# KONFIGURASI SERVER
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = False

MAX_UPLOAD_MB = 25
MAX_N_RUNS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# =============================================================================
# HALAMAN & HEALTH
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "config": {
            "lookback": config.LOOKBACK,
            "train_ratio": config.TRAIN_RATIO,
            "val_ratio": config.VAL_RATIO,
            "epochs": config.EPOCHS,
            "n_runs": config.N_RUNS,
            "max_n_runs": MAX_N_RUNS,
        },
    }), 200


# =============================================================================
# WORKER PELATIHAN (dijalankan di thread latar untuk satu job)
# =============================================================================

def _run_predict_job(job, file_bytes: bytes):
    """
    Untuk satu job upload: MEMUAT model matang dari output_model/best_model/ lalu
    hanya MEMPREDIKSI dataset yang diunggah (tanpa training) -- prosesnya cepat.
    Progres fase didorong ke antrean job untuk dialirkan ke browser via SSE.

    Model dilatih terlebih dahulu secara offline (python train_model.py), yang
    mempromosikan model terbaik ke folder produksi permanen. Bila folder itu
    belum ada, job gagal dengan pesan yang mengarahkan pengguna menjalankannya.
    """
    job.status = "running"

    def phase(msg):
        logger.info(f"   [job {job.id}] {msg}")
        job.emit({"type": "phase", "message": msg})

    try:
        phase("Memuat model produksi tersimpan (LSTM, GRU, scaler)...")
        bundle = storage.load_best_model(OUTPUT_DIR)
        if bundle is None:
            raise DatasetError(
                "Model produksi belum tersedia. Latih model terlebih dahulu secara "
                "offline dengan menjalankan: python train_model.py --data <dataset>.csv "
                "-- perintah itu akan menyimpan model matang ke folder "
                f"output_model/{config.BEST_MODEL_DIRNAME}/ yang dipakai untuk prediksi."
            )

        best_epochs = (bundle.get("metadata") or {}).get("best_epochs") or {}

        result = run_predict_pipeline(
            io.BytesIO(file_bytes),
            model_lstm=bundle["model_lstm"],
            model_gru=bundle["model_gru"],
            scaler=bundle["scaler"],
            progress_cb=phase,
            best_epochs=best_epochs,
        )

        # Sertakan asal-usul model (kapan dilatih & dari run mana) untuk transparansi.
        meta = bundle.get("metadata") or {}
        result["model_info"] = {
            "promoted_at": meta.get("promoted_at"),
            "source_run": meta.get("source_run"),
            "trained_n_runs": meta.get("n_runs"),
        }

        job.result = result
        job.status = "done"
        job.emit({"type": "done", "result": result})
        logger.info(
            f"<- job {job.id} selesai (prediksi) | "
            f"RMSE LSTM={result['metrics']['lstm']['rmse']['mean']:.2f} "
            f"GRU={result['metrics']['gru']['rmse']['mean']:.2f} | "
            f"Terbaik: {result['conclusion']['kesimpulan_akhir']}"
        )
    except DatasetError as e:
        job.status = "error"
        job.error = str(e)
        logger.warning(f"job {job.id} validasi gagal: {e}")
        job.emit({"type": "error", "message": str(e)})
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        logger.error(f"job {job.id} error:\n{traceback.format_exc()}")
        job.emit({"type": "error", "message": f"Terjadi kesalahan internal saat memproses dataset: {e}"})


# =============================================================================
# ENDPOINT: MULAI JOB PELATIHAN
# =============================================================================

@app.route("/api/process", methods=["POST"])
def process_dataset():
    """
    Menerima file CSV (field 'dataset'), memulai job PREDIKSI di thread latar
    (memuat model matang lalu memprediksi -- tanpa training), dan mengembalikan
    { job_id } agar frontend dapat menyambung ke aliran progres via
    /api/stream/<job_id>.
    """
    logger.info(f"-> POST /api/process | IP: {request.remote_addr}")

    if "dataset" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Tidak ada file yang diunggah. Sertakan file CSV pada field 'dataset'.",
        }), 400

    file = request.files["dataset"]
    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "Nama file kosong."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "status": "error",
            "message": "Format file tidak didukung. Harap unggah file berekstensi .csv.",
        }), 400

    # Field 'n_runs' dari frontend diabaikan: web tidak lagi melatih model, ia
    # hanya memprediksi memakai model matang tersimpan (satu model per arsitektur).

    # Baca seluruh isi file SEKARANG (stream request akan tertutup begitu handler
    # ini selesai; thread latar butuh byte-nya sesudah itu).
    file_bytes = file.read()

    job = jobs.create_job()
    threading.Thread(
        target=_run_predict_job,
        args=(job, file_bytes),
        daemon=True,
    ).start()

    return jsonify({"status": "started", "job_id": job.id}), 202


# =============================================================================
# ENDPOINT: ALIRAN PROGRES PELATIHAN (Server-Sent Events)
# =============================================================================

@app.route("/api/stream/<job_id>", methods=["GET"])
def stream(job_id):
    job = jobs.get_job(job_id)
    if job is None:
        return jsonify({"status": "error", "message": "Job tidak ditemukan atau sudah kedaluwarsa."}), 404

    def event_stream():
        while True:
            try:
                ev = job.queue.get(timeout=15)
            except Exception:
                # Tidak ada event 15 detik -> kirim komentar keep-alive agar
                # koneksi/proxy tidak menutup aliran.
                yield ": keep-alive\n\n"
                continue

            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

            if ev.get("type") in ("done", "error"):
                break

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # cegah buffering di proxy
            "Access-Control-Allow-Origin": "*",
        },
    )


# =============================================================================
# ERROR HANDLERS GLOBAL
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint tidak ditemukan."}), 404


@app.errorhandler(413)
def payload_too_large(e):
    return jsonify({
        "status": "error",
        "message": f"File terlalu besar. Maksimum ukuran unggah adalah {MAX_UPLOAD_MB} MB.",
    }), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Unhandled server error: {traceback.format_exc()}")
    return jsonify({"status": "error", "message": "Terjadi kesalahan internal pada server."}), 500


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  ETH Price Prediction (LSTM vs GRU) — Starting Server")
    logger.info("=" * 55)
    logger.info(f"  Host     : {HOST}")
    logger.info(f"  Port     : {PORT}")
    logger.info(f"  Lookback : {config.LOOKBACK} hari")
    logger.info(f"  Mode     : PREDIKSI (memuat model matang, tanpa training)")

    _best_dir = storage.best_model_dir(OUTPUT_DIR)
    _has_model = all(os.path.exists(os.path.join(_best_dir, fn))
                     for fn in ("model_lstm.h5", "model_gru.h5", "scaler.pkl"))
    if _has_model:
        logger.info(f"  Model    : ditemukan di {_best_dir}")
    else:
        logger.warning("  Model produksi BELUM ada di %s", _best_dir)
        logger.warning("  Jalankan dulu: python train_model.py --data <dataset>.csv")
    logger.info("=" * 55)

    # threaded=True wajib: satu thread melayani aliran SSE sementara thread lain
    # menjalankan pelatihan.
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
