# =============================================================================
# SKRIP CLI OFFLINE: PELATIHAN & EVALUASI MODEL LSTM/GRU (UNTUK ARSIP SKRIPSI)
# =============================================================================
# Judul Skripsi:
#   "Analisis Perbandingan Kinerja Algoritma Gated Recurrent Unit (GRU) dan
#    Long Short-Term Memory (LSTM) dalam Prediksi Harga Ethereum Berbasis
#    Data Time Series"
#
# Deskripsi:
#   Skrip ini adalah pembungkus CLI di atas pipeline.py — pipeline INTI yang
#   sama persis dengan yang dipakai oleh endpoint upload di app.py. Skrip ini
#   ditujukan untuk dijalankan manual dari terminal guna menghasilkan
#   plot (.png) dan tabel ringkasan (.csv) untuk dilampirkan pada naskah
#   skripsi (Bab 4), dan TIDAK dipakai oleh alur web (index.html -> app.py).
#
# Dependensi:
#   pip install numpy pandas scikit-learn tensorflow matplotlib joblib
#
# Cara Menjalankan:
#   python train_model.py --data data_eth_idr.csv
#   (--data bersifat opsional; default ke data_eth_idr.csv di folder ini)
# =============================================================================

import os
import argparse

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
import storage
from pipeline import run_full_pipeline, DatasetError

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_history(history_lstm, history_gru, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Kurva Loss Pelatihan Model", fontsize=14, fontweight="bold")

    for ax, history, name, color in zip(
        axes, [history_lstm, history_gru], ["LSTM", "GRU"], ["#2563EB", "#16A34A"]
    ):
        ax.plot(history.history["loss"], label="Train Loss", color=color, linewidth=1.5)
        ax.plot(history.history["val_loss"], label="Validation Loss",
                 color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_title(f"Model {name}", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_training_loss.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [v] Plot disimpan di: {save_path}")


def plot_predictions(df_test, metrics_lstm, metrics_gru, output_dir: str):
    dates = df_test.index

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Perbandingan Prediksi vs Harga Aktual (Data Test)", fontsize=13, fontweight="bold")

    ax.plot(dates, metrics_lstm["y_true"], label="Harga Aktual", color="#374151", linewidth=2, zorder=3)
    ax.plot(dates, metrics_lstm["y_pred"], label="Prediksi LSTM", color="#2563EB",
            linewidth=1.5, linestyle="--", alpha=0.85)
    ax.plot(dates, metrics_gru["y_pred"], label="Prediksi GRU", color="#16A34A",
            linewidth=1.5, linestyle=":", alpha=0.85)

    if hasattr(dates, "to_pydatetime"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)

    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Close")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "plot_prediction_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [v] Plot disimpan di: {save_path}")


def plot_metrics_comparison(metrics_summary: dict, output_dir: str):
    """Bar chart mean +/- std (error bar) dari N run, bukan cuma satu angka."""
    metric_names = ["RMSE", "MAE", "MAPE (%)"]
    keys = ["rmse", "mae", "mape"]
    lstm_means = [metrics_summary["lstm"][k]["mean"] for k in keys]
    lstm_stds = [metrics_summary["lstm"][k]["std"] for k in keys]
    gru_means = [metrics_summary["gru"][k]["mean"] for k in keys]
    gru_stds = [metrics_summary["gru"][k]["std"] for k in keys]

    import numpy as np
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_lstm = ax.bar(x - width / 2, lstm_means, width, yerr=lstm_stds, capsize=4,
                        label="LSTM", color="#2563EB", alpha=0.85, edgecolor="white")
    bars_gru = ax.bar(x + width / 2, gru_means, width, yerr=gru_stds, capsize=4,
                       label="GRU", color="#16A34A", alpha=0.85, edgecolor="white")

    ax.set_title("Perbandingan Metrik Evaluasi LSTM vs GRU (mean ± std)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Nilai Metrik")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bars, color in ((bars_lstm, "#1e3a5f"), (bars_gru, "#14532d")):
        for bar in bars:
            ax.annotate(f"{bar.get_height():,.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=color)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_metrics_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [v] Plot disimpan di: {save_path}")


def save_results_csv(metrics_summary: dict, per_run: dict, output_dir: str, kesimpulan: dict):
    import pandas as pd

    summary = {"Model": ["LSTM", "GRU"]}
    for label, key in (("RMSE", "rmse"), ("MAE", "mae"), ("MAPE (%)", "mape")):
        summary[f"{label} (mean)"] = [metrics_summary["lstm"][key]["mean"], metrics_summary["gru"][key]["mean"]]
        summary[f"{label} (std)"] = [metrics_summary["lstm"][key]["std"], metrics_summary["gru"][key]["std"]]
    df_summary = pd.DataFrame(summary)
    csv_path = os.path.join(output_dir, "hasil_evaluasi.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"  [v] Ringkasan metrik (mean ± std dari seluruh run) disimpan di: {csv_path}")
    print(df_summary.to_string(index=False))

    df_kesimpulan = pd.DataFrame({
        "Metrik": ["RMSE", "MAE", "MAPE", "Kesimpulan Akhir"],
        "Model Unggul": [kesimpulan["RMSE"], kesimpulan["MAE"], kesimpulan["MAPE"], kesimpulan["kesimpulan_akhir"]],
    })
    kesimpulan_path = os.path.join(output_dir, "kesimpulan_model_terbaik.csv")
    df_kesimpulan.to_csv(kesimpulan_path, index=False)
    print(f"  [v] Kesimpulan model terbaik disimpan di: {kesimpulan_path}")
    print(df_kesimpulan.to_string(index=False))

    rows = []
    for i in range(len(per_run["lstm"])):
        rows.append({"Run": per_run["lstm"][i]["run"], "Model": "LSTM",
                      "RMSE": per_run["lstm"][i]["rmse"], "MAE": per_run["lstm"][i]["mae"],
                      "MAPE (%)": per_run["lstm"][i]["mape"], "Epoch": per_run["lstm"][i]["epochs"]})
        rows.append({"Run": per_run["gru"][i]["run"], "Model": "GRU",
                      "RMSE": per_run["gru"][i]["rmse"], "MAE": per_run["gru"][i]["mae"],
                      "MAPE (%)": per_run["gru"][i]["mape"], "Epoch": per_run["gru"][i]["epochs"]})
    df_per_run = pd.DataFrame(rows)
    per_run_path = os.path.join(output_dir, "hasil_per_run.csv")
    df_per_run.to_csv(per_run_path, index=False)
    print(f"  [v] Rincian tiap run disimpan di: {per_run_path}")
    print(df_per_run.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Pelatihan & evaluasi offline model LSTM/GRU harga ETH.")
    parser.add_argument("--data", default="data_eth_idr.csv",
                         help="Path ke file CSV dataset (kolom Close/Close Price wajib ada).")
    parser.add_argument("--n-runs", type=int, default=config.N_RUNS,
                         help=f"Jumlah pengulangan training per model (default: {config.N_RUNS}). "
                              "Gunakan angka besar (mis. 10) untuk hasil akhir skripsi yang dilaporkan "
                              "sebagai mean +/- std; gunakan angka kecil (mis. 1-2) untuk uji cepat.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SISTEM PREDIKSI HARGA ETHEREUM — TAHAP PELATIHAN MODEL (CLI)")
    print(f"  Jumlah run per model: {args.n_runs}")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # PELAPORAN PROGRES ALA GOOGLE COLAB (alur pelatihan terlihat jelas di
    # terminal, bisa di-screenshot untuk lampiran/sidang skripsi):
    #  - tiap model diberi header "RUN x/n . MODEL . seed=..."
    #  - tiap epoch menandai apakah val_loss MEMBAIK (jadi kandidat model terbaik)
    #    atau tidak (menampilkan hitung mundur "sabar k/patience" milik
    #    EarlyStopping), sehingga terlihat langsung KAPAN & KENAPA training berhenti
    #  - saat sebuah model selesai, dicetak ringkasan epoch TERBAIK (bobot yang
    #    disimpan oleh restore_best_weights).
    # Logika "membaik/sabar/berhenti" di bawah sengaja mereplikasi persis perilaku
    # EarlyStopping (monitor val_loss, min_delta=0) agar angka yang ditampilkan
    # konsisten dengan mekanisme yang benar-benar dipakai model.
    # -------------------------------------------------------------------------
    es_patience = config.ES_PATIENCE
    _st = {"key": None, "best": None, "best_epoch": None, "wait": 0, "last_epoch": 0}

    def _flush_model_summary():
        """Cetak baris penutup model yang barusan selesai. Idempoten: setelah
        dicetak, key dinolkan agar pemanggilan berikutnya tidak mencetak ulang."""
        if _st["key"] is None:
            return
        print(f"  +-- selesai: {_st['last_epoch']} epoch | TERBAIK epoch {_st['best_epoch']} "
              f"(val_loss {_st['best']:.6f}, bobot inilah yang disimpan)")
        _st["key"] = None

    def report(msg):
        # Fase non-epoch berikutnya (mis. "Menyusun agregasi...") menandai model
        # terakhir sudah selesai -> tutup ringkasannya dulu agar urutan benar.
        _flush_model_summary()
        # Baris "Menjalankan run..." sudah diwakili header kotak tiap model, jadi
        # dilewati agar tidak dobel; fase lain tetap dicetak sebagai penanda tahap.
        if msg.startswith("Menjalankan run"):
            return
        print(f"\n>> {msg}")

    def report_epoch(ev):
        key = (ev["run"], ev["model"])
        if key != _st["key"]:
            _flush_model_summary()  # tutup ringkasan model sebelumnya (bila ada)
            seed = config.SEED + ev["run"] - 1
            print(f"\n  +-- RUN {ev['run']}/{ev['n_runs']} . {ev['model']} . seed={seed} "
                  + "-" * 22)
            _st.update(key=key, best=None, best_epoch=None, wait=0, last_epoch=0)

        vl = ev["val_loss"]
        _st["last_epoch"] = ev["epoch"]
        if _st["best"] is None or vl < _st["best"]:
            _st.update(best=vl, best_epoch=ev["epoch"], wait=0)
            marker = "[v] membaik (terbaik)"
        else:
            _st["wait"] += 1
            tail = " -> BERHENTI" if _st["wait"] >= es_patience else ""
            marker = f".. sabar {_st['wait']}/{es_patience}{tail}"

        print(f"  |  epoch {ev['epoch']:>3}/{ev['max_epochs']} | "
              f"loss {ev['loss']:.6f} | val_loss {vl:.6f}   {marker}")

    try:
        result = run_full_pipeline(args.data, n_runs=args.n_runs,
                                   progress_cb=report, epoch_cb=report_epoch)
        _flush_model_summary()  # tutup ringkasan model terakhir yang dilatih
    except DatasetError as e:
        print(f"\n[GAGAL] {e}")
        raise SystemExit(1)

    artifacts = result["_artifacts"]
    metrics_summary = result["metrics"]

    print("\n" + "=" * 60)
    print(f"HASIL EVALUASI PADA DATA TEST (rata-rata dari {result['n_runs']} run)")
    print("=" * 60)
    for name in ("lstm", "gru"):
        m = metrics_summary[name]
        print(f"  [{name.upper()}] "
              f"RMSE={m['rmse']['mean']:,.2f} ± {m['rmse']['std']:,.2f}  "
              f"MAE={m['mae']['mean']:,.2f} ± {m['mae']['std']:,.2f}  "
              f"MAPE={m['mape']['mean']:.4f}% ± {m['mape']['std']:.4f}%")
    print(f"\n  [v] KESIMPULAN AKHIR (mayoritas 3 metrik, berdasarkan rata-rata): "
          f"{result['conclusion']['kesimpulan_akhir']}\n")

    # Folder BARU yang unik untuk run ini -- tidak pernah menimpa hasil run
    # sebelumnya, walau dataset atau --n-runs yang dipakai sama persis.
    run_dir = storage.make_run_dir(OUTPUT_DIR, args.n_runs)
    print(f"  [*] Artefak run ini disimpan di folder baru: {run_dir}\n")

    plot_training_history(artifacts["history_lstm"], artifacts["history_gru"], run_dir)
    plot_predictions(artifacts["df_test"], artifacts["metrics_lstm"], artifacts["metrics_gru"], run_dir)
    plot_metrics_comparison(metrics_summary, run_dir)

    artifacts["model_lstm"].save(os.path.join(run_dir, "model_lstm.h5"))
    artifacts["model_gru"].save(os.path.join(run_dir, "model_gru.h5"))
    joblib.dump(artifacts["scaler"], os.path.join(run_dir, "scaler.pkl"))
    print(f"  [v] Model & scaler dari run TERBAIK (val_loss validasi terendah) disimpan di: {run_dir}")

    save_results_csv(metrics_summary, result["per_run"], run_dir, result["conclusion"])
    result_without_artifacts = {k: v for k, v in result.items() if k != "_artifacts"}

    # Epoch dengan val_loss terendah = bobot yang disimpan (restore_best_weights).
    best_saved_epoch = {}
    for name in ("lstm", "gru"):
        val = artifacts[f"history_{name}"].history["val_loss"]
        best_saved_epoch[name] = int(min(range(len(val)), key=lambda i: val[i])) + 1
    result_without_artifacts["best_saved_epoch"] = best_saved_epoch

    storage.save_run_summary(run_dir, result_without_artifacts)
    storage.save_training_log(run_dir, result_without_artifacts, artifacts)
    storage.save_training_history(run_dir, artifacts)
    print(f"  [v] Log proses training disimpan di: {os.path.join(run_dir, 'training_log.txt')}")
    print(f"  [v] History epoch (JSON) disimpan di: {os.path.join(run_dir, 'training_history.json')}")

    # -------------------------------------------------------------------------
    # PROMOSI MODEL PRODUKSI: salin model TERBAIK ke folder permanen agar web
    # (app.py) bisa langsung memakainya untuk prediksi cepat tanpa melatih ulang.
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  PROMOSI MODEL PRODUKSI (dipakai web untuk prediksi cepat)")
    print("-" * 60)
    print(f"  Model terbaik dipilih dari run dengan val_loss VALIDASI terendah (bukan RMSE test,")
    print(f"  agar bebas kebocoran seleksi/model selection leakage).")
    print(f"  Kesimpulan model terbaik   : {result['conclusion']['kesimpulan_akhir']}")
    print(f"  Epoch terbaik (bobot disimpan): LSTM epoch {best_saved_epoch['lstm']} | "
          f"GRU epoch {best_saved_epoch['gru']}  (val_loss terendah)")
    best_dir = storage.promote_best_model(OUTPUT_DIR, run_dir, result_without_artifacts)
    print(f"  [v] Model produksi (LSTM + GRU + scaler + metadata) disimpan permanen di:")
    print(f"      {best_dir}")
    print(f"  Web (app.py) akan SELALU memuat model dari folder ini saat upload CSV.")
    print(f"  (Arsip run ini di {os.path.basename(run_dir)} tetap utuh, tidak tertimpa.)")

    print("\n" + "=" * 60)
    print("  PELATIHAN SELESAI")
    print("  - Arsip lengkap run ini :", run_dir)
    print("  - Model produksi web    :", best_dir)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
