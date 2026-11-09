"""Sonde mémoire exécutée dans un processus NEUF : RSS avant/après chargement du modèle et pic
pendant l'inférence en lot (échantillonnage toutes les 2 ms). Sortie : JSON (Mo)."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import psutil

from app.prediction import predict_fraud
from app.training import load_model

MB = 1024 * 1024


def main(model_path: str, csv_path: str) -> dict:
    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    df = pd.read_csv(csv_path)
    rss_data = proc.memory_info().rss
    model, thr, _ = load_model(model_path)
    rss_model = proc.memory_info().rss
    peak = [rss_model]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    for i in range(0, len(df), 10_000):
        predict_fraud(model, df.iloc[i : i + 10_000], thr)
    peak_batches = peak[0]
    predict_fraud(model, df, thr)
    stop.set()
    thread.join()
    return {
        "rows": len(df),
        "rss_after_imports_mb": rss_start / MB,
        "rss_after_data_mb": rss_data / MB,
        "rss_after_model_load_mb": rss_model / MB,
        "model_load_delta_mb": (rss_model - rss_data) / MB,
        "peak_rss_batches_10k_mb": peak_batches / MB,
        "peak_rss_single_batch_mb": peak[0] / MB,
        "inference_delta_batches_mb": (peak_batches - rss_model) / MB,
        "inference_delta_single_batch_mb": (peak[0] - rss_model) / MB,
    }


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2])))
