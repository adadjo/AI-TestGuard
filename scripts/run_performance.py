"""Tests de performance (Activité 8) : latence, P95, débit, taille, mémoire.

Protocole : voir PROTOCOL ci-dessous et le rapport (§9). Le matériel et les versions sont
enregistrés avec les résultats ; les mesures utilisent time.perf_counter_ns.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plot_utils as pu
import psutil

from app import config as cfg
from app.prediction import handle_prediction_request, predict_fraud
from app.preprocessing import deduplicate, load_data, preprocess_data, split_data
from app.synthetic import make_synthetic_transactions
from app.training import load_model
from app.utils import dump_json, environment_info

PROTOCOL = {
    "warmup_calls": 30,
    "single_calls_per_run": 1000,
    "independent_runs": 3,
    "batch_sizes": [1, 10, 100, 1000, 10_000, 50_000],
    "batch_repeats": {1: 30, 10: 30, 100: 30, 1000: 20, 10_000: 8, 50_000: 4},
    "timer": "time.perf_counter_ns",
    "gc": "ramasse-miettes actif (comportement de production) ; gc.collect() avant chaque série",
    "inputs": "transactions du jeu de test, une ligne distincte par appel",
    "scope_single": "(a) predict_proba brut ; (b) predict_fraud = validation + modèle + seuil ; "
    "(c) gestionnaire de requête = JSON + schéma strict + validation + modèle + réponse ; "
    "(d) = (c) + sérialisation JSON de la réponse. Sans HTTP ni réseau (aucune API déployée).",
}


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "inconnu"


def summarize(ms: np.ndarray) -> dict:
    return {
        "n": int(ms.size),
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
        "std_ms": float(ms.std(ddof=1)),
    }


def time_calls(fn, items, warmup: int) -> np.ndarray:
    for item in items[:warmup]:
        fn(item)
    gc.collect()
    out = np.empty(len(items))
    for i, item in enumerate(items):
        t0 = time.perf_counter_ns()
        fn(item)
        out[i] = (time.perf_counter_ns() - t0) / 1e6
    return out


def measure_model(name, model, thr, X_test, sample_csv, sample_csv_path):
    n = PROTOCOL["single_calls_per_run"]
    rows = [X_test.iloc[[i]] for i in range(n)]
    payloads = [
        json.dumps({k: float(v) for k, v in X_test.iloc[i][cfg.FEATURE_COLUMNS].items()})
        for i in range(n)
    ]
    layers = {
        "predict_proba brut": lambda r: model.predict_proba(r),
        "predict_fraud (service)": lambda r: predict_fraud(model, r, thr),
    }
    handler = lambda p: handle_prediction_request(p, model, thr)  # noqa: E731
    handler_json = lambda p: json.dumps(handle_prediction_request(p, model, thr)[1])  # noqa: E731
    res: dict = {"single": {}, "single_runs": {}}
    raw_samples = {}
    for label, fn in layers.items():
        runs = [
            time_calls(fn, rows, PROTOCOL["warmup_calls"])
            for _ in range(PROTOCOL["independent_runs"])
        ]
        res["single_runs"][label] = [summarize(r) for r in runs]
        res["single"][label] = summarize(np.concatenate(runs))
        raw_samples[label] = runs[0]
    for label, fn in (
        ("gestionnaire de requête", handler),
        ("gestionnaire + réponse JSON", handler_json),
    ):
        runs = [
            time_calls(fn, payloads, PROTOCOL["warmup_calls"])
            for _ in range(PROTOCOL["independent_runs"])
        ]
        res["single_runs"][label] = [summarize(r) for r in runs]
        res["single"][label] = summarize(np.concatenate(runs))
        raw_samples[label] = runs[0]

    # débit en lot
    pool = sample_csv
    res["batch"] = []
    for size in PROTOCOL["batch_sizes"]:
        reps = PROTOCOL["batch_repeats"][size]
        batch = pool.iloc[:size]
        predict_fraud(model, batch, thr)
        times_service, times_raw = [], []
        for _ in range(reps):
            gc.collect()
            t0 = time.perf_counter()
            predict_fraud(model, batch, thr)
            times_service.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            model.predict_proba(batch)
            times_raw.append(time.perf_counter() - t0)
        res["batch"].append(
            {
                "batch_size": size,
                "repeats": reps,
                "median_s": float(np.median(times_service)),
                "throughput_tx_s": float(size / np.median(times_service)),
                "throughput_raw_tx_s": float(size / np.median(times_raw)),
            }
        )
    # taille et chargement
    path = cfg.MODELS_DIR / f"{name}_v1.joblib"
    loads = []
    for _ in range(5):
        t0 = time.perf_counter()
        load_model(path)
        loads.append((time.perf_counter() - t0) * 1000)
    res["model_file"] = {
        "size_bytes": path.stat().st_size,
        "size_mb": path.stat().st_size / 1024**2,
        "load_median_ms": float(np.median(loads)),
    }
    # mémoire (processus neuf)
    probe = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("memory_probe.py")),
            str(path),
            str(sample_csv_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=cfg.PROJECT_ROOT,
    )
    res["memory"] = json.loads(probe.stdout.strip().splitlines()[-1])
    return res, raw_samples


def make_figure(samples: dict, batch: list, selected: str) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    ax = axes[0]
    colors = ["#7F8C8D", "#1F5FA8", "#D9822B", "#C0392B"]
    for (label, vals), color in zip(samples.items(), colors, strict=False):
        ax.hist(
            vals,
            bins=np.geomspace(max(vals.min(), 0.05), vals.max() * 1.05, 45),
            alpha=0.6,
            color=color,
            label=label,
        )
        ax.axvline(np.percentile(vals, 95), color=color, ls=":", lw=1.2)
    ax.axvline(20, color="black", ls="--", lw=1, label="Budget CA12 (20 ms)")
    ax.axvline(50, color="black", ls="-.", lw=1, label="Budget CA13 (50 ms)")
    ax.set_xscale("log")
    ax.set_xlabel("Latence d'un appel (ms, échelle log)")
    ax.set_ylabel("Nombre d'appels")
    ax.set_title(f"Latence unitaire — {pu.LABELS[selected]}")
    ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[1]
    sizes = [b["batch_size"] for b in batch]
    ax.plot(
        sizes,
        [b["throughput_tx_s"] for b in batch],
        marker="o",
        color=pu.COLORS["logreg"],
        label="predict_fraud (service)",
    )
    ax.plot(
        sizes,
        [b["throughput_raw_tx_s"] for b in batch],
        marker="s",
        color=pu.COLORS["grey"],
        label="predict_proba brut",
    )
    ax.axhline(1000, color=pu.COLORS["bad"], ls="--", lw=1, label="Critère CA14 (1 000 tx/s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Taille du lot")
    ax.set_ylabel("Débit (transactions / s)")
    ax.set_title("Débit selon la taille du lot")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return pu.save(fig, "fig_performance.png")


def main() -> None:
    results = json.loads((cfg.REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
    synthetic = results["source"]["type"] == "synthetic"
    raw = (
        make_synthetic_transactions(60_000, 0.005, seed=1)
        if synthetic
        else load_data(cfg.DATA_PATH)
    )
    df, _ = deduplicate(raw)
    X, y = preprocess_data(df)
    splits = split_data(X, y, random_state=cfg.RANDOM_STATE)
    pool = X.sample(n=min(100_000, len(X)), random_state=1)
    tmp_dir = tempfile.mkdtemp()
    sample_csv_path = str(Path(tmp_dir) / "sample.csv")
    pool.to_csv(sample_csv_path, index=False)
    env = {
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
        "os": platform.platform(),
        "threads_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        **environment_info(),
    }
    out = {"environment": env, "protocol": PROTOCOL, "models": {}}
    samples = {}
    for name in ("logreg", "tree"):
        print(f"performance : {name}", flush=True)
        model, thr, _ = load_model(cfg.MODELS_DIR / f"{name}_v1.joblib")
        out["models"][name], samples[name] = measure_model(
            name, model, thr, splits.X_test, pool, sample_csv_path
        )
    sel = results["selected_model"]["name"]
    m = out["models"][sel]
    batch_1000 = next(b for b in m["batch"] if b["batch_size"] == 1000)
    out["selected"] = sel
    out["criteria_feed"] = {
        "latency_p95_ms_service": float(
            np.median([r["p95_ms"] for r in m["single_runs"]["predict_fraud (service)"]])
        ),
        "latency_p95_ms_handler": float(
            np.median([r["p95_ms"] for r in m["single_runs"]["gestionnaire + réponse JSON"]])
        ),
        "throughput_tx_s": batch_1000["throughput_tx_s"],
        "model_size_mb": m["model_file"]["size_mb"],
        "rss_delta_mb": m["memory"]["inference_delta_single_batch_mb"],
    }
    out["figures"] = {"performance": make_figure(samples[sel], m["batch"], sel)}
    dump_json(out, cfg.REPORTS_DIR / "performance.json")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("criteria_feed:", {k: round(v, 3) for k, v in out["criteria_feed"].items()})


if __name__ == "__main__":
    main()
