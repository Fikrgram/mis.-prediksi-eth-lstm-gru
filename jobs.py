# =============================================================================
# REGISTRI JOB PELATIHAN (untuk streaming progres pelatihan live via SSE)
# =============================================================================
# Pelatihan model memakan waktu lama, sementara kita ingin browser dapat
# menampilkan progres per-epoch secara langsung (ala Colab). Karena itu:
#   1. Pelatihan dijalankan di THREAD LATAR (background), bukan menahan request.
#   2. Setiap job punya ANTREAN (queue) event yang thread-safe. Thread pelatihan
#      mendorong event (fase, epoch, selesai/error) ke antrean; endpoint SSE
#      di app.py membaca antrean itu dan mengalirkannya ke browser.
#
# Registri ini sengaja sederhana (in-memory) -- cukup untuk aplikasi penelitian
# satu-pengguna. Job lama otomatis dipangkas agar memori tidak menumpuk.

import queue
import uuid
import threading
import collections

# Batas jumlah job yang disimpan di memori (job terlama dibuang lebih dulu).
_MAX_JOBS = 20


class Job:
    """Satu sesi pelatihan: menyimpan antrean event, status, hasil akhir/error."""

    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.queue = queue.Queue()
        self.status = "pending"   # pending -> running -> done | error
        self.result = None
        self.error = None

    def emit(self, event: dict):
        """Dorong satu event ke antrean untuk dialirkan ke klien."""
        self.queue.put(event)


_jobs = collections.OrderedDict()
_lock = threading.Lock()


def create_job() -> Job:
    job = Job()
    with _lock:
        _jobs[job.id] = job
        # Pangkas job terlama bila melebihi kapasitas.
        while len(_jobs) > _MAX_JOBS:
            _jobs.popitem(last=False)
    return job


def get_job(job_id: str):
    with _lock:
        return _jobs.get(job_id)
