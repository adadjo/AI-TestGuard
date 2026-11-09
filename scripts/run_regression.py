"""Non-régression (Activité 6.2) : compare des versions candidates à la référence FIGÉE.

La référence (baseline/baseline_full.json) a été créée par une action explicite
(scripts/make_baseline.py --kind full --confirm) ; ce script ne l'écrit ni ne la modifie jamais.
Les comparaisons se font sur le jeu de VALIDATION fixe (même graine, même empreinte).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from make_baseline import validation_set_hash

from app import config as cfg
from app.evaluation import evaluate_pipeline
from app.preprocessing import deduplicate, load_data, preprocess_data, split_data
from app.regression import (
    DECISION_ACCEPT,
    DECISION_INVESTIGATE,
    DECISION_REJECT,
    compare_to_baseline,
    environment_differences,
    load_baseline,
)
from app.synthetic import make_synthetic_transactions
from app.training import select_threshold, train_model
from app.utils import dump_json

# Les attentes ("expected") sont fixées A PRIORI pour les cas qui servent d'auto-test de la barrière.
CANDIDATES = [
    {
        "id": "C1",
        "version": "v1.0.1",
        "title": "Refactoring neutre : max_iter 2000 → 5000 (même optimum)",
        "expected": DECISION_ACCEPT,
    },
    {
        "id": "C2",
        "version": "v1.1.0",
        "title": "Régularisation 10 fois plus forte (C ÷ 10)",
        "expected": None,
    },
    {
        "id": "C3",
        "version": "v1.2.0",
        "title": "Time ajouté aux variables du modèle",
        "expected": None,
    },
    {
        "id": "C4",
        "version": "v1.3.0",
        "title": "Fuite accidentelle : entraînement sur train + validation",
        "expected": None,
    },
    {
        "id": "C5",
        "version": "v2.0.0",
        "title": "Remplacement par un arbre de décision de profondeur 3",
        "expected": DECISION_REJECT,
    },
]


def build_candidate(cand, base_name, base_params, splits):
    name, params, include_time = base_name, dict(base_params), False
    X_fit, y_fit = splits.X_train, splits.y_train
    if cand["id"] == "C1":
        params["max_iter"] = 5000
    elif cand["id"] == "C2":
        params["C"] = float(base_params.get("C", 1.0)) / 10
    elif cand["id"] == "C3":
        include_time = True
    elif cand["id"] == "C4":
        X_fit, y_fit = (
            pd.concat([splits.X_train, splits.X_val]),
            pd.concat([splits.y_train, splits.y_val]),
        )
    elif cand["id"] == "C5":
        name, params = "tree", {"max_depth": 3, "min_samples_leaf": 20, "class_weight": "balanced"}
    model = train_model(X_fit, y_fit, name, params, cfg.RANDOM_STATE, include_time=include_time)
    thr = select_threshold(
        splits.y_val, model.predict_proba(splits.X_val)[:, 1], cfg.THRESHOLD_FPR_TARGET
    ).threshold
    metrics, _ = evaluate_pipeline(model, splits.X_val, splits.y_val, thr)
    return name, params, thr, metrics


def main() -> None:
    results = json.loads((cfg.REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
    synthetic = results["source"]["type"] == "synthetic"
    path = (
        cfg.REPORTS_DIR / "baseline_proxy.json"
        if synthetic
        else cfg.BASELINE_DIR / "baseline_full.json"
    )
    baseline = load_baseline(path)
    raw = (
        make_synthetic_transactions(60_000, 0.005, seed=1)
        if synthetic
        else load_data(cfg.DATA_PATH)
    )
    df, _ = deduplicate(raw)
    X, y = preprocess_data(df)
    splits = split_data(X, y, random_state=baseline["seeds"]["split"])
    same_validation = (
        validation_set_hash(splits.X_val.index) == baseline["dataset"]["validation_set_sha256"]
    )

    rows = []
    for cand in CANDIDATES:
        name, params, thr, metrics = build_candidate(
            cand, baseline["model"]["name"], baseline["model"]["params"], splits
        )
        measured = {k: metrics[k] for k in cfg.REGRESSION_TOLERANCES}
        comparison = compare_to_baseline(
            measured, baseline["validation_metrics"], baseline["tolerances"]
        )
        rows.append(
            {
                **cand,
                "model": name,
                "params": params,
                "threshold": thr,
                "metrics": measured,
                "n_fraud_validation": metrics["n_positive"],
                "comparison": comparison,
                "matches_expectation": None
                if cand["expected"] is None
                else comparison["decision"] == cand["expected"],
            }
        )
        print(
            f"{cand['id']} {cand['version']}: {comparison['decision']}  {comparison['reasons'][:2]}"
        )

    # auto-test de la règle « gain suspect » (cas synthétique, sans entraînement)
    gain_case = dict(baseline["validation_metrics"])
    gain_case["average_precision"] += 0.08
    gain = compare_to_baseline(gain_case, baseline["validation_metrics"], baseline["tolerances"])
    selftest = [
        {
            "case": "C1 refactoring neutre",
            "expected": DECISION_ACCEPT,
            "obtained": rows[0]["comparison"]["decision"],
        },
        {
            "case": "C5 arbre de profondeur 3",
            "expected": DECISION_REJECT,
            "obtained": rows[4]["comparison"]["decision"],
        },
        {
            "case": "Gain de PR-AUC de +0,08 (suspect)",
            "expected": DECISION_INVESTIGATE,
            "obtained": gain["decision"],
        },
    ]
    failures = sum(t["expected"] != t["obtained"] for t in selftest)
    out = {
        "baseline": {
            k: baseline[k]
            for k in (
                "created_utc",
                "code_version",
                "git_commit",
                "model",
                "seeds",
                "dataset",
                "validation_metrics",
                "tolerances",
            )
        }
        | {"environment": baseline["environment"]},
        "same_validation_set": same_validation,
        "environment_differences": environment_differences(baseline["environment"]),
        "candidates": rows,
        "gate_selftest": selftest,
        "criteria_feed": {"regression_gate_failures": failures},
    }
    dump_json(out, cfg.REPORTS_DIR / "regression.json")
    print("auto-test de la barrière :", selftest, "-> échecs :", failures)


if __name__ == "__main__":
    main()
