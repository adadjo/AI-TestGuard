"""Surveillance après déploiement (Activité 10) : fenêtres de production simulées et alertes.

Schéma chronologique : le modèle est entraîné sur les 40 % premiers de la période ; la validation
(40-55 %) règle le seuil ; 6 fenêtres de production (45 % restants, dans l'ordre du temps) sont
rejouées. Des incidents sont INJECTÉS dans certaines fenêtres pour vérifier qu'ils sont détectés.
Deux références sont comparées :

* NAÏVE : la période de validation (la plus récente avant la mise en production) ;
* APPARIÉE : les transactions du JOUR 1 qui ont lieu aux MÊMES HEURES DU JOUR que la fenêtre ;
* GLISSANTE : la fenêtre de production PRÉCÉDENTE (telle qu'observée). Le jeu Kaggle est fortement
  non stationnaire (certaines composantes V dérivent d'un jour à l'autre) : une référence statique
  déclenche des alertes en permanence, une référence glissante détecte les ruptures.

Les alertes sont simulées (aucun Prometheus/Grafana déployé).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu

from app import config as cfg
from app.evaluation import evaluate_model, wilson_interval
from app.monitoring import drift_report, window_report
from app.preprocessing import deduplicate, load_data, preprocess_data
from app.robustness import apply_covariate_shift, reference_sigma
from app.synthetic import make_synthetic_transactions
from app.training import select_threshold, train_model
from app.utils import dump_json

N_WINDOWS = 6
DAY = 86_400.0
SCENARIOS = {
    1: ("Normale", "aucun incident"),
    2: ("Normale", "aucun incident"),
    3: ("Dérive de données", "Amount × 1,5 (inflation / nouveau marchand)"),
    4: ("Dérive de données", "décalage de +0,5 σ sur 8 composantes V"),
    5: ("Dérive de concept", "les fraudes ressemblent désormais à des transactions légitimes"),
    6: ("Incident de qualité", "2 % de lignes avec valeur infinie + 3 % de cellules manquantes"),
}


def summarize(rep: dict) -> dict:
    keys = (
        "max_feature_psi",
        "n_features_in_alert",
        "score_psi",
        "alert_rate",
        "alert_rate_z",
        "alerts",
    )
    return {k: rep[k] for k in keys}


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
    order = np.argsort(X["Time"].to_numpy(), kind="stable")
    X, y = X.iloc[order], y.iloc[order]
    n = len(X)
    i_train, i_ref = int(0.40 * n), int(0.55 * n)
    X_tr, y_tr = X.iloc[:i_train], y.iloc[:i_train]
    X_ref, y_ref = X.iloc[i_train:i_ref], y.iloc[i_train:i_ref]
    sel = results["selected_model"]
    model = train_model(X_tr, y_tr, sel["name"], sel["params"], cfg.RANDOM_STATE)
    all_scores = model.predict_proba(X)[:, 1]
    ref_scores = all_scores[i_train:i_ref]
    choice = select_threshold(y_ref, ref_scores, cfg.THRESHOLD_FPR_TARGET)
    thr = choice.threshold
    ref_alert_rate = float((ref_scores >= thr).mean())
    sigma = reference_sigma(X_tr)
    rng = np.random.default_rng(2026)
    subset = list(rng.choice(cfg.V_COLUMNS, size=8, replace=False))
    time_all = X["Time"].to_numpy()
    span = float(time_all.max())

    # plancher de bruit : PSI maximal entre deux moitiés ALÉATOIRES du jour 1 (aucune dérive réelle)
    day1 = X[X["Time"] < DAY]
    perm = np.random.default_rng(0).permutation(len(day1))
    half = len(day1) // 2
    noise_floor = max(
        r["psi"] for r in drift_report(day1.iloc[perm[:half]], day1.iloc[perm[half:]])
    )

    edges = np.linspace(i_ref, n, N_WINDOWS + 1).astype(int)
    rows = []
    prev_X, prev_s = None, None
    for w in range(1, N_WINDOWS + 1):
        Xw, yw = X.iloc[edges[w - 1] : edges[w]].copy(), y.iloc[edges[w - 1] : edges[w]].copy()
        t0, t1 = float(Xw["Time"].min()), float(Xw["Time"].max())
        invalid_rate = 0.0
        if w == 3:
            Xw["Amount"] = Xw["Amount"] * 1.5
        elif w == 4:
            Xw = apply_covariate_shift(Xw, sigma, shift_sigma=0.5, columns=subset)
        elif w == 5:
            fraud_idx = np.flatnonzero(yw.to_numpy() == 1)
            legit_idx = np.flatnonzero(yw.to_numpy() == 0)
            donors = rng.choice(legit_idx, size=len(fraud_idx), replace=False)
            cols = [Xw.columns.get_loc(c) for c in cfg.V_COLUMNS]
            Xw.iloc[fraud_idx, cols] = Xw.iloc[donors][cfg.V_COLUMNS].to_numpy()
        elif w == 6:
            bad = rng.choice(len(Xw), size=int(0.02 * len(Xw)), replace=False)
            holes = rng.random((len(Xw), len(cfg.MODEL_FEATURES))) < 0.03
            values = Xw[cfg.MODEL_FEATURES].to_numpy(copy=True)
            values[holes] = np.nan
            Xw[cfg.MODEL_FEATURES] = values
            Xw.iloc[bad, Xw.columns.get_loc("V3")] = np.inf
            invalid_rate = len(bad) / len(Xw)
            keep = np.ones(len(Xw), dtype=bool)
            keep[bad] = False  # rejetées à l'entrée par le contrat : jamais scorées
            Xw, yw = Xw[keep], yw[keep]
        scores = model.predict_proba(Xw)[:, 1]
        # référence appariée : mêmes heures du jour, jour précédent
        mask = (time_all >= t0 - DAY) & (time_all <= t1 - DAY)
        Xm, sm = X[mask], all_scores[mask]
        if (
            prev_X is None
        ):  # fenêtre 1 : dernière tranche de même taille de la période de validation
            j0 = max(0, i_ref - len(Xw))
            prev_X, prev_s = X.iloc[j0:i_ref], all_scores[j0:i_ref]
        trailing = window_report(
            prev_X, prev_s, Xw, scores, thr, float((prev_s >= thr).mean()), invalid_rate
        )
        trailing_ref_rate = float((prev_s >= thr).mean())
        prev_X, prev_s = Xw, scores
        naive = window_report(X_ref, ref_scores, Xw, scores, thr, ref_alert_rate, invalid_rate)
        matched = window_report(Xm, sm, Xw, scores, thr, float((sm >= thr).mean()), invalid_rate)
        m = evaluate_model(yw, (scores >= thr).astype(int), scores)
        rows.append(
            {
                "window": w,
                "scenario": SCENARIOS[w][0],
                "injected": SCENARIOS[w][1],
                "n": len(Xw),
                "hours_of_day": [round((t0 % DAY) / 3600, 1), round((t1 % DAY) / 3600, 1)],
                "matched_reference_rows": int(mask.sum()),
                "matched_reference_alert_rate": float((sm >= thr).mean()),
                "naive": summarize(naive),
                "matched": summarize(matched),
                "trailing": summarize(trailing),
                "alert_rate": trailing["alert_rate"],
                "trailing_reference_alert_rate": trailing_ref_rate,
                "n_fraud": m["n_positive"],
                "delayed_recall": m["recall"],
                "delayed_recall_ci": wilson_interval(m["tp"], m["n_positive"]),
                "delayed_precision": m["precision"],
                "delayed_fpr": m["fpr"],
                "delayed_ap": m["average_precision"],
            }
        )
        print(
            w,
            SCENARIOS[w][0],
            "naïve:",
            [a["name"] for a in naive["alerts"]],
            "| appariée:",
            [a["name"] for a in matched["alerts"]],
            "| glissante:",
            [a["name"] for a in trailing["alerts"]],
            f"| rappel différé={m['recall']:.2f}",
            flush=True,
        )

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2))
    xs = np.arange(1, N_WINDOWS + 1)
    ax = axes[0]
    ax.plot(
        xs,
        [r["naive"]["max_feature_psi"] for r in rows],
        marker="o",
        ls="--",
        color=pu.COLORS["grey"],
        label="PSI max, réf. naïve",
    )
    ax.plot(
        xs,
        [r["matched"]["max_feature_psi"] for r in rows],
        marker="o",
        color=pu.COLORS["logreg"],
        label="PSI max, réf. appariée",
    )
    ax.plot(
        xs,
        [r["matched"]["score_psi"] for r in rows],
        marker="s",
        color=pu.COLORS["tree"],
        label="PSI du score, appariée",
    )
    ax.axhline(cfg.PSI_WARN, color=pu.COLORS["warn"], ls="--", lw=1)
    ax.axhline(cfg.PSI_ALERT, color=pu.COLORS["bad"], ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title("Dérive des données (sans étiquette)", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    ax = axes[1]
    k = np.array([r["alert_rate"] for r in rows]) * 100
    base = np.array([r["trailing_reference_alert_rate"] for r in rows]) * 100
    sizes = np.array([r["n"] for r in rows])
    band = cfg.ALERT_RATE_Z * np.sqrt(base / 100 * (1 - base / 100) / sizes) * 100
    ax.fill_between(
        xs, base - band, base + band, color="grey", alpha=0.22, label="Fenêtre précédente ± 3 σ"
    )
    ax.plot(xs, base, color="black", lw=1)
    ax.plot(xs, k, marker="o", color=pu.COLORS["fraud"], label="Taux d'alertes observé")
    ax.set_title("Transactions signalées (%)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[2]
    rec = np.array([r["delayed_recall"] for r in rows])
    lo = rec - np.array([r["delayed_recall_ci"][0] for r in rows])
    hi = np.array([r["delayed_recall_ci"][1] for r in rows]) - rec
    ax.errorbar(xs, rec, yerr=[lo, hi], marker="o", capsize=3, color=pu.COLORS["ok"])
    ax.axhline(0.75, color=pu.COLORS["bad"], ls="--", lw=1, label="Critère CA01")
    ax.set_ylim(0, 1.05)
    ax.set_title("Rappel (étiquettes différées)", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    for a in axes:
        a.set_xlabel("Fenêtre de production")
        a.set_xticks(xs)
    fig.tight_layout()
    figure = pu.save(fig, "fig_monitoring.png")

    out = {
        "design": {
            "train_fraction": 0.40,
            "reference_fraction": 0.15,
            "n_windows": N_WINDOWS,
            "model": sel["name"],
            "threshold": thr,
            "reference_alert_rate": ref_alert_rate,
            "reference_rows": len(X_ref),
            "psi_warn": cfg.PSI_WARN,
            "psi_alert": cfg.PSI_ALERT,
            "alert_rate_z": cfg.ALERT_RATE_Z,
            "reference_val_recall": choice.recall,
            "reference_val_fpr": choice.fpr,
            "span_hours": span / 3600,
            "noise_floor_psi": noise_floor,
        },
        "windows": rows,
        "figures": {"monitoring": figure},
    }
    dump_json(out, cfg.REPORTS_DIR / "monitoring.json")


if __name__ == "__main__":
    main()
