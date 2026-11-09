"""Robustesse et sensibilité (Activité 5) : scénarios obligatoires, comportement attendu vs observé.

Les modèles sont rechargés depuis les artefacts sauvegardés (intégrité SHA-256 vérifiée), puis
évalués sur le jeu de test avec le seuil GELÉ : aucune décision de réglage n'en découle.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plot_utils as pu

from app import config as cfg
from app.monitoring import drift_report, population_stability_index
from app.prediction import handle_prediction_request, predict_fraud
from app.preprocessing import (
    DataValidationError,
    deduplicate,
    load_data,
    preprocess_data,
    split_data,
)
from app.robustness import (
    apply_covariate_shift,
    inject_missing,
    perturb_features,
    reference_sigma,
    sensitivity_report,
)
from app.synthetic import make_synthetic_transactions
from app.training import load_model
from app.utils import dump_json

EPS_LEVELS = [0.01, 0.05, 0.10, 0.25]
N_SEEDS = 5
MODERATE = 0.05
KEYS = (
    "mean_abs_dp",
    "p99_abs_dp",
    "max_abs_dp",
    "flip_rate",
    "n_flips",
    "delta_recall",
    "delta_precision",
    "delta_fpr",
    "delta_average_precision",
)


class _WarningCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def aggregate(reports: list[dict]) -> dict:
    return {
        k: {
            "mean": float(np.mean([r[k] for r in reports])),
            "std": float(np.std([r[k] for r in reports], ddof=1)) if len(reports) > 1 else 0.0,
        }
        for k in KEYS
    } | {"n_invalid": int(sum(r["n_invalid"] for r in reports))}


def attempt(model, X, thr):
    """Exécute une prédiction et décrit le comportement observé : traité ou rejeté."""
    counter = _WarningCounter()
    logger = logging.getLogger("aitestguard")
    logger.addHandler(counter)
    try:
        out = predict_fraud(model, X, thr)
        ok = bool(
            np.isfinite(out["fraud_probability"]).all()
            and out["fraud_probability"].between(0, 1).all()
        )
        return {"outcome": "traité", "valid_probabilities": ok, "n_warnings": len(counter.messages)}
    except DataValidationError as exc:
        return {"outcome": "rejeté", "error": "DataValidationError", "message": str(exc)[:110]}
    finally:
        logger.removeHandler(counter)


def invalid_batch_cases(X: pd.DataFrame) -> list[tuple[str, object]]:
    base = X.head(500).copy()

    def with_value(col, value):
        out = base.copy()
        out.loc[out.index[3], col] = value
        return out

    holes = base.copy()
    holes.loc[holes.index[:100], "V9"] = np.nan  # 20 % > 5 %
    return [
        ("Valeur +inf dans V1", with_value("V1", np.inf)),
        ("Valeur −inf dans Amount", with_value("Amount", -np.inf)),
        ("Type texte dans V3", base.assign(V3=base["V3"].astype(str))),
        ("Type booléen dans V4", base.assign(V4=True)),
        ("Colonne V5 absente", base.drop(columns=["V5"])),
        ("Colonne inattendue", base.assign(extra=1.0)),
        ("Colonne V6 dupliquée", pd.concat([base, base[["V6"]]], axis=1)),
        ("Amount négatif", with_value("Amount", -5.0)),
        ("Amount > 1 000 000", with_value("Amount", 1e9)),
        ("|V8| > 1 000", with_value("V8", 5e4)),
        ("Time négatif", with_value("Time", -10.0)),
        ("Taux de NaN de 20 % dans V9", holes),
        ("Jeu vide", base.iloc[0:0]),
        ("Entrée non DataFrame (liste)", [1, 2, 3]),
    ]


def handler_cases(payload: dict) -> list[tuple[str, object, int]]:
    text = json.dumps(payload)
    nan_text = text.replace(f'"V2": {payload["V2"]}', '"V2": NaN')
    inf_text = text.replace(f'"V2": {payload["V2"]}', '"V2": Infinity')
    big_text = text.replace(f'"V2": {payload["V2"]}', '"V2": 1e999')
    missing = {k: v for k, v in payload.items() if k != "V7"}
    return [
        ("Champ V7 manquant", missing, 422),
        ("Champ supplémentaire", {**payload, "intrus": 1.0}, 422),
        ("Type texte (V3)", {**payload, "V3": "abc"}, 422),
        ("Valeur nulle (V3)", {**payload, "V3": None}, 422),
        ("Booléen (V3)", {**payload, "V3": True}, 422),
        ("Constante NaN", nan_text, 400),
        ("Constante Infinity", inf_text, 400),
        ("Dépassement 1e999", big_text, 422),
        ("Amount négatif", {**payload, "Amount": -1.0}, 422),
        ("JSON invalide", text[:-8], 400),
        ("Corps vide", "", 400),
        ("Charge utile = liste", "[1, 2, 3]", 400),
        (
            "Charge utile trop grande",
            json.dumps({**payload, "pad": "x" * (cfg.MAX_PAYLOAD_BYTES + 1)}),
            413,
        ),
    ]


def run_model(name, model, thr, splits, sigma, source_scores):
    X, y = splits.X_test, splits.y_test.to_numpy()
    out: dict = {"threshold": thr}

    # A. balayage d'amplitude (V et Amount perturbés ensemble)
    out["noise"] = {}
    for eps in EPS_LEVELS:
        reps = [
            sensitivity_report(model, X, y, perturb_features(X, sigma, eps, eps, seed=100 + s), thr)
            for s in range(N_SEEDS)
        ]
        out["noise"][str(eps)] = aggregate(reps)
    # B. familles de perturbation à l'amplitude modérée
    out["families"] = {}
    for label, ev, ea in (("V1-V28 seulement", MODERATE, 0.0), ("Amount seulement", 0.0, MODERATE)):
        reps = [
            sensitivity_report(model, X, y, perturb_features(X, sigma, ev, ea, seed=200 + s), thr)
            for s in range(N_SEEDS)
        ]
        out["families"][label] = aggregate(reps)

    # C. valeurs manquantes
    out["missing"] = []
    for frac in (0.01, 0.03, 0.05, 0.10):
        holes = inject_missing(X, frac, seed=7)
        obs = attempt(model, holes, thr)
        row = {
            "fraction": frac,
            "realized_max_column_fraction": float(holes[cfg.MODEL_FEATURES].isna().mean().max()),
            **obs,
        }
        if obs["outcome"] == "traité":
            row.update(
                {
                    k: v
                    for k, v in sensitivity_report(model, X, y, holes, thr).items()
                    if k in ("flip_rate", "delta_recall", "delta_precision", "delta_fpr")
                }
            )
        out["missing"].append(row)

    # D. montants extrêmes
    rng = np.random.default_rng(11)
    scen = []
    x1 = X.copy()
    pick = rng.choice(len(X), size=len(X) // 100, replace=False)
    x1.iloc[pick, x1.columns.get_loc("Amount")] = x1["Amount"].to_numpy()[pick] * 10
    scen.append(("1 % des montants × 10", x1))
    x2 = X.copy()
    pick2 = rng.choice(len(X), size=20, replace=False)
    x2.iloc[pick2, x2.columns.get_loc("Amount")] = 50_000.0
    scen.append(("20 montants = 50 000 (> seuil d'alerte 30 000)", x2))
    x3 = X.head(500).copy()
    x3.iloc[0, x3.columns.get_loc("Amount")] = 1e9
    scen.append(("1 montant = 1 000 000 000 (hors contrat)", x3))
    out["extreme_amount"] = []
    for label, frame in scen:
        obs = attempt(model, frame, thr)
        row = {"scenario": label, **obs}
        if obs["outcome"] == "traité":
            ref = X if len(frame) == len(X) else X.head(len(frame))
            row.update(
                {
                    k: v
                    for k, v in sensitivity_report(model, ref, y[: len(frame)], frame, thr).items()
                    if k in ("flip_rate", "delta_recall", "delta_fpr", "max_abs_dp")
                }
            )
        out["extreme_amount"].append(row)

    # E. entrées invalides (lot) et requêtes malformées (gestionnaire)
    batch = []
    for label, frame in invalid_batch_cases(X):
        try:
            predict_fraud(model, frame, thr)
            batch.append({"case": label, "accepted_wrongly": True, "error": None})
        except DataValidationError as exc:
            batch.append({"case": label, "accepted_wrongly": False, "error": type(exc).__name__})
    payload = {k: float(v) for k, v in X.iloc[5][cfg.FEATURE_COLUMNS].items()}
    http = []
    for label, raw, expected in handler_cases(payload):
        status, body = handle_prediction_request(raw, model, thr)
        http.append(
            {
                "case": label,
                "expected_status": expected,
                "status": status,
                "error": body.get("error"),
                "accepted_wrongly": status == 200,
                "as_expected": status == expected,
            }
        )
    out["invalid_batch"], out["invalid_requests"] = batch, http
    out["invalid_inputs_accepted"] = int(sum(r["accepted_wrongly"] for r in batch + http))
    out["status_mismatches"] = int(sum(not r["as_expected"] for r in http))

    # F. décalage de distribution (référence PSI = validation)
    shifts = []
    rng2 = np.random.default_rng(5)
    subset = list(rng2.choice(cfg.V_COLUMNS, size=8, replace=False))
    for label, kwargs in (
        ("V1-V28 : +0,5 σ", {"shift_sigma": 0.5}),
        ("V1-V28 : +1 σ", {"shift_sigma": 1.0}),
        ("8 composantes V : +1 σ", {"shift_sigma": 1.0, "columns": subset}),
        ("Amount × 2", {"amount_factor": 2.0}),
        ("Amount × 5", {"amount_factor": 5.0}),
    ):
        shifted = apply_covariate_shift(X, sigma, **kwargs)
        rep = sensitivity_report(model, X, y, shifted, thr)
        drift = drift_report(splits.X_val, shifted)
        new_scores = predict_fraud(model, shifted, thr)["fraud_probability"].to_numpy()
        score_psi = population_stability_index(source_scores, new_scores)
        shifts.append(
            {
                "scenario": label,
                "max_feature_psi": max(d["psi"] for d in drift),
                "n_features_alert": sum(d["level"] == "alert" for d in drift),
                "score_psi": score_psi,
                "detected": bool(
                    any(d["level"] == "alert" for d in drift) or score_psi > cfg.PSI_ALERT
                ),
                "flip_rate": rep["flip_rate"],
                "delta_recall": rep["delta_recall"],
                "delta_precision": rep["delta_precision"],
                "delta_fpr": rep["delta_fpr"],
                "delta_average_precision": rep["delta_average_precision"],
            }
        )
    out["shift"] = shifts

    # G. invariance à Time (propriété de conception)
    changed = int(
        (
            model.predict_proba(X)[:, 1]
            != model.predict_proba(X.assign(Time=X["Time"] * 3 + 999))[:, 1]
        ).sum()
    )
    out["time_invariance_changed_rows"] = changed
    return out


def make_figures(res: dict, selected: str) -> dict:
    import matplotlib.pyplot as plt

    figs = {}
    eps = [str(e) for e in EPS_LEVELS]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.1))
    panels = [
        ("mean_abs_dp", "Variation moyenne |Δp|", True, None, 1),
        ("flip_rate", "Changement de classe (%)", False, 0.5, 100),
        ("delta_recall", "Variation du rappel (points)", False, -5.0, 100),
    ]
    for ax, (key, title, logy, crit, scale) in zip(axes, panels, strict=True):
        for name in res:
            m = [res[name]["noise"][e][key]["mean"] * scale for e in eps]
            s = [res[name]["noise"][e][key]["std"] * scale for e in eps]
            ax.errorbar(
                [100 * float(e) for e in eps],
                m,
                yerr=s,
                marker="o",
                ms=4,
                capsize=2,
                color=pu.COLORS[name],
                label=pu.LABELS[name],
            )
        if crit is not None:
            ax.axhline(crit, ls="--", color=pu.COLORS["bad"], lw=1, label="Critère")
        if logy:
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25], ["1", "5", "10", "25"])
        ax.set_xlabel("Amplitude du bruit (% de σ)")
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    figs["noise"] = pu.save(fig, "fig_robustness_noise.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    shifts = res[selected]["shift"]
    x = np.arange(len(shifts))
    ax.bar(
        x - 0.2,
        [100 * s["delta_recall"] for s in shifts],
        0.38,
        color=pu.COLORS["fraud"],
        label="Δ rappel (points)",
    )
    ax.bar(
        x + 0.2,
        [100 * s["delta_average_precision"] for s in shifts],
        0.38,
        color=pu.COLORS["legit"],
        label="Δ PR-AUC (points)",
    )
    for i, s in enumerate(shifts):
        ax.text(
            i,
            2.5,
            "alerte PSI ✔" if s["detected"] else "non détecté ✘",
            ha="center",
            fontsize=7,
            color=pu.COLORS["ok"] if s["detected"] else pu.COLORS["bad"],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [s["scenario"] for s in shifts], rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("Variation (points de %)")
    ax.set_title(f"Décalage de distribution — {pu.LABELS[selected]} (seuil gelé)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    figs["shift"] = pu.save(fig, "fig_robustness_shift.png")
    return figs


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
    sigma = reference_sigma(splits.X_train)

    res = {}
    for name in ("logreg", "tree"):
        model, thr, _ = load_model(cfg.MODELS_DIR / f"{name}_v1.joblib")
        source_scores = model.predict_proba(splits.X_val)[:, 1]
        print(f"robustesse : {name}", flush=True)
        res[name] = run_model(name, model, thr, splits, sigma, source_scores)

    selected = results["selected_model"]["name"]
    moderate = res[selected]["noise"][str(MODERATE)]
    feed = {
        "perturbation_flip_rate": moderate["flip_rate"]["mean"],
        "perturbation_recall_drop": max(0.0, -moderate["delta_recall"]["mean"]),
        "invalid_inputs_accepted": res[selected]["invalid_inputs_accepted"],
    }
    out = {
        "protocol": {
            "eps_levels": EPS_LEVELS,
            "n_seeds": N_SEEDS,
            "moderate": MODERATE,
            "sigma_reference": "écart-type des composantes V sur l'entraînement",
        },
        "models": res,
        "selected": selected,
        "criteria_feed": feed,
        "figures": make_figures(res, selected),
    }
    dump_json(out, cfg.REPORTS_DIR / "robustness.json")
    print("criteria_feed:", feed)


if __name__ == "__main__":
    main()
