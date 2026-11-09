"""Évaluation du modèle (Activités 3, 4 et 6.1) : modèles, seuil, erreurs, incertitudes.

Protocole (voir rapport §3-§7) : partition stratifiée 70/15/15 après dédoublonnage ; transformations
et modèle appris sur l'entraînement ; hyperparamètres (PR-AUC) et seuil (max rappel sous
FPR <= 0,08 %) réglés sur la validation ; jeu de test utilisé UNE fois pour l'évaluation finale.

Usage : python scripts/run_experiments.py [--synthetic | --allow-synthetic-fallback] [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plot_utils as pu
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve, roc_curve

from app import __version__
from app import config as cfg
from app.evaluation import (
    bootstrap_compare,
    check_criteria,
    cost_summary,
    evaluate_model,
    evaluate_pipeline,
    qualification_verdict,
    segment_report,
    wilson_interval,
)
from app.preprocessing import (
    dataset_fingerprint,
    deduplicate,
    load_data,
    preprocess_data,
    split_data,
    temporal_split,
)
from app.synthetic import make_synthetic_transactions
from app.training import (
    load_model,
    save_model,
    select_candidate,
    select_threshold,
    train_model,
    tune_model,
)
from app.utils import dump_json, environment_info, sha256_file

KAGGLE_MD5 = "e90efcb83d69faf99fcab8b0255024de"
PERF_CRITERIA = ["CA01", "CA02", "CA03", "CA04", "CA05", "CA06"]
CRITERIA_VALUES = {c["metric"]: c["value"] for c in cfg.ACCEPTANCE_CRITERIA}
AMOUNT_BANDS = [
    ("< 10 €", 0.0, 10.0),
    ("10 – 100 €", 10.0, 100.0),
    ("≥ 100 €", 100.0, float("inf")),
]
SEEDS = list(range(10))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_dataset(args):
    if args.synthetic or (not cfg.DATA_PATH.exists() and args.allow_synthetic_fallback):
        if not args.synthetic:
            log(
                "ATTENTION : data/creditcard.csv absent -> proxy synthétique (résultats non réels)."
            )
        raw = make_synthetic_transactions(60_000, 0.005, seed=1)
        return raw, {"type": "synthetic", "label": "PROXY SYNTHÉTIQUE (pas le jeu Kaggle)"}
    if not cfg.DATA_PATH.exists():
        sys.exit("data/creditcard.csv absent : voir data/README.md (ou utiliser --synthetic).")
    raw = load_data(cfg.DATA_PATH)
    md5 = hashlib.md5(cfg.DATA_PATH.read_bytes()).hexdigest()
    return raw, {
        "type": "kaggle",
        "label": "Credit Card Fraud Detection (Kaggle, mlg-ulb / ULB)",
        "file_sha256": sha256_file(cfg.DATA_PATH),
        "file_md5": md5,
        "md5_matches_published": md5 == KAGGLE_MD5,
    }


def _stats(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def error_analysis(X_test, y_test, proba, threshold) -> dict:
    y = np.asarray(y_test)
    pred = (proba >= threshold).astype(int)
    amount = X_test["Amount"].to_numpy()
    masks = {
        "TP": (y == 1) & (pred == 1),
        "FN": (y == 1) & (pred == 0),
        "FP": (y == 0) & (pred == 1),
        "TN": (y == 0) & (pred == 0),
    }
    fn_p = proba[masks["FN"]]
    lt = float(_logit(threshold))
    near = (lt - _logit(fn_p)) <= 2.0 if fn_p.size else np.array([], dtype=bool)
    detail = lambda m: sorted(  # noqa: E731
        ({"proba": float(p), "amount": float(a)} for p, a in zip(proba[m], amount[m], strict=True)),
        key=lambda r: -r["proba"],
    )
    return {
        "amount": {k: _stats(amount[m]) for k, m in masks.items()},
        "proba": {k: _stats(proba[m]) for k, m in masks.items()},
        "tn_proba_quantiles": {
            q: float(np.quantile(proba[masks["TN"]], q)) for q in (0.5, 0.99, 0.999)
        },
        "fn_share_below_p001": float((fn_p < 0.01).mean()) if fn_p.size else None,
        "fn_near_threshold": int(near.sum()),
        "fn_far_from_threshold": int((~near).sum()),
        "fn_list": detail(masks["FN"])[:40],
        "fp_list": detail(masks["FP"])[:40],
        "fraud_amount_total": float(amount[y == 1].sum()),
        "fraud_amount_detected": float(amount[masks["TP"]].sum()),
        "fraud_amount_missed": float(amount[masks["FN"]].sum()),
        "legit_amount_blocked": float(amount[masks["FP"]].sum()),
    }


def fit_model(name, splits):
    t0 = time.perf_counter()
    params, pipe, tuning = tune_model(
        name, splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    tune_seconds = time.perf_counter() - t0
    p_val = pipe.predict_proba(splits.X_val)[:, 1]
    choice = select_threshold(splits.y_val, p_val)
    thr = choice.threshold
    m_train, p_train = evaluate_pipeline(pipe, splits.X_train, splits.y_train, thr)
    m_val, _ = evaluate_pipeline(pipe, splits.X_val, splits.y_val, thr)
    m_test, p_test = evaluate_pipeline(pipe, splits.X_test, splits.y_test, thr)
    ci = {
        "recall": wilson_interval(m_test["tp"], m_test["n_positive"]),
        "precision": wilson_interval(m_test["tp"], m_test["tp"] + m_test["fp"]),
        "fpr": wilson_interval(m_test["fp"], m_test["n_negative"]),
        "fnr": wilson_interval(m_test["fn"], m_test["n_positive"]),
    }
    per_day = lambda k: k / m_test["n"] * cfg.TRANSACTIONS_PER_DAY  # noqa: E731
    out = {
        "params": params,
        "threshold": thr,
        "threshold_selection": {
            "max_fpr_target": cfg.THRESHOLD_FPR_TARGET,
            "val_fpr": choice.fpr,
            "val_recall": choice.recall,
            "val_precision": choice.precision,
        },
        "tuning": tuning,
        "train": m_train,
        "val": m_val,
        "test": m_test,
        "ci_test": ci,
        "tune_seconds": tune_seconds,
        "alerts_per_day": per_day(m_test["n_alerts"]),
        "false_alerts_per_day": per_day(m_test["fp"]),
        "cost_test": cost_summary(
            splits.y_test, (p_test >= thr).astype(int), splits.X_test["Amount"]
        ),
    }
    return out, pipe, {"val": p_val, "train": p_train, "test": p_test}


def criteria_for(model_out, bootstrap_models) -> list[dict]:
    t = model_out["test"]
    measured = {
        "recall": t["recall"],
        "fpr": t["fpr"],
        "precision": t["precision"],
        "average_precision": t["average_precision"],
        "roc_auc": t["roc_auc"],
        "generalization_gap_ap": model_out["train"]["average_precision"] - t["average_precision"],
    }
    intervals = {
        "recall": model_out["ci_test"]["recall"],
        "fpr": model_out["ci_test"]["fpr"],
        "precision": model_out["ci_test"]["precision"],
        "average_precision": (
            bootstrap_models["average_precision"]["low"],
            bootstrap_models["average_precision"]["high"],
        ),
        "roc_auc": (bootstrap_models["roc_auc"]["low"], bootstrap_models["roc_auc"]["high"]),
    }
    crit = [c for c in cfg.ACCEPTANCE_CRITERIA if c["id"] in PERF_CRITERIA]
    return check_criteria(measured, crit, intervals)


def seed_study(name, params, X, y) -> dict:
    rows = []
    for seed in SEEDS:
        sp = split_data(X, y, random_state=seed)
        pipe = train_model(sp.X_train, sp.y_train, name, params, seed)
        thr = select_threshold(sp.y_val, pipe.predict_proba(sp.X_val)[:, 1]).threshold
        m, _ = evaluate_pipeline(pipe, sp.X_test, sp.y_test, thr)
        rows.append(
            {
                "seed": seed,
                "threshold": thr,
                "n_fraud_test": m["n_positive"],
                **{
                    k: m[k]
                    for k in (
                        "recall",
                        "precision",
                        "fpr",
                        "average_precision",
                        "roc_auc",
                        "fp",
                        "fn",
                        "tp",
                    )
                },
            }
        )
    keys = ("recall", "precision", "fpr", "average_precision", "roc_auc")
    summary = {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "std": float(np.std([r[k] for r in rows], ddof=1)),
            "min": float(np.min([r[k] for r in rows])),
            "max": float(np.max([r[k] for r in rows])),
        }
        for k in keys
    }
    passes = {
        "CA01": float(np.mean([r["recall"] >= CRITERIA_VALUES["recall"] for r in rows])),
        "CA02": float(np.mean([r["fpr"] <= CRITERIA_VALUES["fpr"] for r in rows])),
        "CA03": float(np.mean([r["precision"] >= CRITERIA_VALUES["precision"] for r in rows])),
        "CA04": float(
            np.mean([r["average_precision"] >= CRITERIA_VALUES["average_precision"] for r in rows])
        ),
        "CA05": float(np.mean([r["roc_auc"] >= CRITERIA_VALUES["roc_auc"] for r in rows])),
    }
    return {"rows": rows, "summary": summary, "pass_rates": passes}


def temporal_study(name, params, X, y, include_time=False) -> dict:
    sp = temporal_split(X, y, 0.6, 0.2)
    pipe = train_model(
        sp.X_train, sp.y_train, name, params, cfg.RANDOM_STATE, include_time=include_time
    )
    thr = select_threshold(sp.y_val, pipe.predict_proba(sp.X_val)[:, 1]).threshold
    m, _ = evaluate_pipeline(pipe, sp.X_test, sp.y_test, thr)
    return {
        "sizes": sp.sizes(),
        "threshold": thr,
        **{
            k: m[k]
            for k in (
                "recall",
                "precision",
                "fpr",
                "average_precision",
                "roc_auc",
                "fp",
                "fn",
                "tp",
            )
        },
    }


def time_ablation(name, params, splits) -> dict:
    out = {}
    for label, inc in (("sans_time", False), ("avec_time", True)):
        pipe = train_model(
            splits.X_train, splits.y_train, name, params, cfg.RANDOM_STATE, include_time=inc
        )
        pv = pipe.predict_proba(splits.X_val)[:, 1]
        thr = select_threshold(splits.y_val, pv).threshold
        m, _ = evaluate_pipeline(pipe, splits.X_test, splits.y_test, thr)
        mv, _ = evaluate_pipeline(pipe, splits.X_val, splits.y_val, thr)
        out[label] = {
            "val_average_precision": mv["average_precision"],
            "test_average_precision": m["average_precision"],
            "test_recall": m["recall"],
            "test_fpr": m["fpr"],
            "test_roc_auc": m["roc_auc"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--allow-synthetic-fallback", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    t_start = time.perf_counter()

    raw, source = load_dataset(args)
    log(f"Source : {source['label']} — {len(raw)} lignes")
    df, n_dup = deduplicate(raw)
    fraud_dup = int(raw["Class"].sum() - df["Class"].sum())
    X, y = preprocess_data(df)
    splits = split_data(X, y, random_state=cfg.RANDOM_STATE)
    span_days = float(raw["Time"].max() / 86400.0)
    dataset = {
        "raw": dataset_fingerprint(raw, cfg.DATA_PATH if source["type"] == "kaggle" else None),
        "after_dedup": dataset_fingerprint(df),
        "duplicates_removed": n_dup,
        "duplicates_among_fraud": fraud_dup,
        "split": splits.sizes(),
        "time_span_days": span_days,
        "tx_per_day_observed": len(raw) / span_days if span_days else None,
        "amount_max": float(raw["Amount"].max()),
        "fraud_by_time_quartile": [
            int(v)
            for v in raw.groupby(
                np.digitize(raw["Time"], np.quantile(raw["Time"], [0.25, 0.5, 0.75]))
            )["Class"].sum()
        ],
    }
    log(
        f"Dédoublonnage : {n_dup} lignes retirées ({fraud_dup} fraudes) -> {len(df)} lignes, {int(y.sum())} fraudes"
    )

    # --- classifieur trivial (piège de l'accuracy)
    dummy = DummyClassifier(strategy="most_frequent").fit(splits.X_train, splits.y_train)
    dummy_metrics = evaluate_model(splits.y_test, dummy.predict(splits.X_test))

    # --- modèles
    models, pipes, probas = {}, {}, {}
    for name in ("logreg", "tree"):
        log(f"Réglage + évaluation : {name}")
        models[name], pipes[name], probas[name] = fit_model(name, splits)
        t = models[name]["test"]
        log(
            f"  {name}: params={models[name]['params']} seuil={models[name]['threshold']:.6f} | test: "
            f"rappel={t['recall']:.3f} préc={t['precision']:.3f} FPR={t['fpr']:.5f} AP={t['average_precision']:.3f} AUC={t['roc_auc']:.3f}"
        )

    # --- bootstrap apparié sur le test
    log(f"Bootstrap stratifié ({args.n_boot} rééchantillons)")
    boot = bootstrap_compare(
        splits.y_test, {n: probas[n]["test"] for n in models}, n_boot=args.n_boot
    )
    criteria = {n: criteria_for(models[n], boot["models"][n]) for n in models}

    # --- sélection du modèle (VALIDATION uniquement) : règle R3 ; R1 conservée pour comparaison
    choice = select_candidate({n: models[n]["val"] for n in models})
    no_candidate = choice["pick"] is None
    selected_name = choice["r1_pick"] if no_candidate else choice["pick"]
    sel = models[selected_name]
    selected = {
        "name": selected_name,
        "params": sel["params"],
        "threshold": sel["threshold"],
        "rule": "R3 : parmi les modèles satisfaisant CA01-CA05 sur la VALIDATION, meilleure PR-AUC de validation",
        "r1_pick": choice["r1_pick"],
        "qualified_on_validation": choice["qualified_on_validation"],
        "no_candidate_qualified": no_candidate,
        "note": "R1 (meilleure PR-AUC de validation seule) était la règle initiale ; elle a désigné "
        f"« {choice['r1_pick']} ». R3 a été adoptée APRÈS avoir constaté que ce choix ne satisfait pas "
        "les critères d'acceptation (voir rapport §4.6) ; le test n'intervient pas dans la règle.",
    }
    log(f"Modèle retenu : {selected_name}")

    # --- sauvegarde / rechargement / reproductibilité
    paths = {}
    for n in models:
        path = cfg.MODELS_DIR / f"{n}_v1.joblib"
        save_model(
            pipes[n],
            models[n]["threshold"],
            path,
            {
                "model": n,
                "params": models[n]["params"],
                "source": source["label"],
                "train_rows": len(splits.X_train),
            },
        )
        paths[n] = path
    reloaded, thr_loaded, meta_loaded = load_model(paths[selected_name])
    p_orig = probas[selected_name]["test"]
    p_reload = reloaded.predict_proba(splits.X_test)[:, 1]
    again = train_model(
        splits.X_train, splits.y_train, selected_name, sel["params"], cfg.RANDOM_STATE
    )
    p_again = again.predict_proba(splits.X_test)[:, 1]
    code = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, '.'); import repro_real, json; "
        f"print(repro_real.digest('{selected_name}', {sel['params']!r}, {args.synthetic or source['type'] == 'synthetic'}))"
    )
    digests = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=cfg.PROJECT_ROOT,
        ).stdout.strip()
        for _ in range(2)
    ]
    repro = {
        "repro_max_abs_diff": float(np.max(np.abs(p_orig - p_again))),
        "reload_max_abs_diff": float(np.max(np.abs(p_orig - p_reload))),
        "threshold_identical_after_reload": thr_loaded == sel["threshold"],
        "model_sha256": meta_loaded["sha256"],
        "model_size_bytes": meta_loaded["size_bytes"],
        "cross_process_digests": digests,
        "cross_process_identical": digests[0] == digests[1] and len(digests[0]) == 64,
    }
    log(
        f"Reproductibilité : {repro['repro_max_abs_diff']:.2e} | rechargement : {repro['reload_max_abs_diff']:.2e} | inter-processus : {repro['cross_process_identical']}"
    )

    # --- analyse des erreurs, points de fonctionnement, coût
    errors = {
        n: error_analysis(splits.X_test, splits.y_test, probas[n]["test"], models[n]["threshold"])
        for n in models
    }
    p_val_sel, p_test_sel = probas[selected_name]["val"], probas[selected_name]["test"]
    budgets = [0.0003, 0.0005, cfg.THRESHOLD_FPR_TARGET, cfg.BUSINESS_MAX_FPR, 0.002, 0.005]
    operating_points = []
    for b in budgets:
        ch = select_threshold(splits.y_val, p_val_sel, b)
        mt = evaluate_model(splits.y_test, (p_test_sel >= ch.threshold).astype(int), p_test_sel)
        operating_points.append(
            {
                "max_fpr_target": b,
                "threshold": ch.threshold,
                "val_recall": ch.recall,
                "val_fpr": ch.fpr,
                **{f"test_{k}": mt[k] for k in ("recall", "precision", "fpr", "tp", "fp", "fn")},
            }
        )
    cost_curve = []
    for b in np.geomspace(0.0001, 0.02, 40):
        ch = select_threshold(splits.y_val, p_val_sel, float(b))
        pred = (p_val_sel >= ch.threshold).astype(int)
        mv = evaluate_model(splits.y_val, pred)
        cost_curve.append(
            {
                "max_fpr": float(b),
                "threshold": ch.threshold,
                "recall": mv["recall"],
                "fpr": mv["fpr"],
                "fp": mv["fp"],
                "fn": mv["fn"],
                **cost_summary(splits.y_val, pred, splits.X_val["Amount"]),
            }
        )

    # --- segments par montant (analyse opérationnelle, PAS une analyse d'équité)
    amount_test = splits.X_test["Amount"].to_numpy()
    band = np.array(
        [next(lbl for lbl, lo, hi in AMOUNT_BANDS if lo <= a < hi) for a in amount_test]
    )
    segments = segment_report(splits.y_test, (p_test_sel >= sel["threshold"]).astype(int), band)

    # --- stabilité : graines, partition temporelle, ablation de Time
    seeds = {}
    for n in models:
        log(f"Sensibilité aux graines : {n}")
        seeds[n] = seed_study(n, models[n]["params"], X, y)
    log("Partition temporelle et ablation de Time")
    temporal = {n: temporal_study(n, models[n]["params"], X, y) for n in models}
    ablation = {n: time_ablation(n, models[n]["params"], splits) for n in models}

    # --- figures
    log("Figures")
    y_test = splits.y_test.to_numpy()
    figs = {"architecture": pu.fig_architecture()}
    figs["accuracy_trap"] = pu.fig_accuracy_trap(
        dummy_metrics, models[selected_name]["test"], selected_name
    )
    curves_roc, ops_roc, aucs, curves_pr, ops_pr, aps = {}, {}, {}, {}, {}, {}
    for n in models:
        fpr, tpr, _ = roc_curve(y_test, probas[n]["test"])
        prec, rec, _ = precision_recall_curve(y_test, probas[n]["test"])
        t = models[n]["test"]
        curves_roc[n], ops_roc[n], aucs[n] = (fpr, tpr), (t["fpr"], t["recall"]), t["roc_auc"]
        curves_pr[n], ops_pr[n], aps[n] = (
            (rec, prec),
            (t["recall"], t["precision"]),
            t["average_precision"],
        )
    figs["roc"] = pu.fig_roc(curves_roc, ops_roc, aucs, cfg.BUSINESS_MAX_FPR)
    figs["pr"] = pu.fig_pr(curves_pr, ops_pr, aps, float(y_test.mean()))
    figs["confusion"] = pu.fig_confusion({n: models[n]["test"] for n in models})
    fpr_v, tpr_v, _ = roc_curve(splits.y_val, p_val_sel)
    figs["threshold"] = pu.fig_threshold(
        (fpr_v, tpr_v), cost_curve, cfg.THRESHOLD_FPR_TARGET, cfg.BUSINESS_MAX_FPR, 0.0
    )
    figs["scores"] = pu.fig_scores(
        p_test_sel[y_test == 0], p_test_sel[y_test == 1], sel["threshold"], selected_name
    )
    figs["overfit"] = pu.fig_overfit(
        {
            n: {
                "train": models[n]["train"]["average_precision"],
                "val": models[n]["val"]["average_precision"],
                "test": models[n]["test"]["average_precision"],
            }
            for n in models
        }
    )
    figs["seeds"] = pu.fig_seeds(
        {n: seeds[n]["rows"] for n in models},
        {
            "recall": CRITERIA_VALUES["recall"],
            "fpr": CRITERIA_VALUES["fpr"],
            "precision": CRITERIA_VALUES["precision"],
            "average_precision": CRITERIA_VALUES["average_precision"],
        },
    )

    results = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "app_version": __version__,
        "environment": environment_info(),
        "source": source,
        "dataset": dataset,
        "accuracy_trap": {"dummy": dummy_metrics},
        "models": models,
        "selected_model": selected,
        "bootstrap": boot,
        "criteria": criteria,
        "verdict_by_model": {n: qualification_verdict(criteria[n]) for n in models},
        "error_analysis": errors,
        "operating_points": operating_points,
        "cost_curve": cost_curve,
        "segments": {"bands": [b[0] for b in AMOUNT_BANDS], "rows": segments},
        "seeds": seeds,
        "temporal": temporal,
        "time_ablation": ablation,
        "reproducibility": repro,
        "figures": figs,
        "runtime_seconds": time.perf_counter() - t_start,
    }
    dump_json(results, cfg.REPORTS_DIR / "results.json")
    log(f"Terminé en {results['runtime_seconds']:.0f} s -> reports/results.json")


if __name__ == "__main__":
    main()
