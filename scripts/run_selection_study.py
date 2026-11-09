"""Étude du PROTOCOLE de sélection du modèle (méta-analyse sur 10 partitions indépendantes).

Question : quelle règle de sélection, calculée sur la VALIDATION uniquement, désigne un modèle qui
satisfait ensuite les critères d'acceptation sur le TEST ?

  R1 = meilleure PR-AUC de validation (règle initiale, indépendante du seuil) ;
  R2 = meilleur rappel de validation au seuil réglé sur la contrainte de FPR (égalité : PR-AUC),
       c'est-à-dire la grandeur que les critères métier CA01/CA02 imposent réellement.

Les jeux de test ne servent ici qu'à juger les RÈGLES (sur 10 partitions), jamais à choisir un modèle.
Hyperparamètres : ceux réglés sur la partition de référence (graine 42), figés pour toutes les graines.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg
from app.evaluation import evaluate_pipeline
from app.preprocessing import deduplicate, load_data, preprocess_data, split_data
from app.synthetic import make_synthetic_transactions
from app.training import select_candidate, select_threshold, train_model
from app.utils import dump_json

CRIT = {c["metric"]: c["value"] for c in cfg.ACCEPTANCE_CRITERIA}
SEEDS = list(range(10))


def passes(m: dict) -> bool:
    return bool(
        m["recall"] >= CRIT["recall"]
        and m["fpr"] <= CRIT["fpr"]
        and m["precision"] >= CRIT["precision"]
        and m["average_precision"] >= CRIT["average_precision"]
        and m["roc_auc"] >= CRIT["roc_auc"]
    )


def pick_r1(val: dict) -> str:
    return max(val, key=lambda n: val[n]["average_precision"])


def pick_r2(val: dict) -> str:
    return max(val, key=lambda n: (round(val[n]["recall"], 12), val[n]["average_precision"]))


def evaluate_seed(seed: int, X, y, params: dict) -> dict:
    sp = split_data(X, y, random_state=seed)
    row: dict = {"seed": seed}
    val, test = {}, {}
    for name in ("logreg", "tree"):
        model = train_model(sp.X_train, sp.y_train, name, params[name], seed)
        thr = select_threshold(sp.y_val, model.predict_proba(sp.X_val)[:, 1]).threshold
        mv, _ = evaluate_pipeline(model, sp.X_val, sp.y_val, thr)
        mt, _ = evaluate_pipeline(model, sp.X_test, sp.y_test, thr)
        val[name], test[name] = mv, mt
        row[name] = {
            "val_average_precision": mv["average_precision"],
            "val_recall": mv["recall"],
            "val_fpr": mv["fpr"],
            "test_recall": mt["recall"],
            "test_fpr": mt["fpr"],
            "test_precision": mt["precision"],
            "test_average_precision": mt["average_precision"],
            "test_roc_auc": mt["roc_auc"],
            "passes_ca01_ca05": passes(mt),
        }
    row["R1_pick"], row["R2_pick"] = pick_r1(val), pick_r2(val)
    row["R3_pick"] = select_candidate(val)["pick"]
    for rule in ("R1", "R2", "R3"):
        pick = row[f"{rule}_pick"]
        row[f"{rule}_pick_passes"] = bool(pick and row[pick]["passes_ca01_ca05"])
    return row


def main() -> None:
    results = json.loads((cfg.REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
    params = {n: results["models"][n]["params"] for n in ("logreg", "tree")}
    synthetic = results["source"]["type"] == "synthetic"
    raw = (
        make_synthetic_transactions(60_000, 0.005, seed=1)
        if synthetic
        else load_data(cfg.DATA_PATH)
    )
    df, _ = deduplicate(raw)
    X, y = preprocess_data(df)
    rows = []
    for seed in [cfg.RANDOM_STATE, *SEEDS]:
        t0 = time.perf_counter()
        rows.append(evaluate_seed(seed, X, y, params))
        r = rows[-1]
        print(
            f"graine {seed:>2}: R1→{r['R1_pick']:6s} ({'ok' if r['R1_pick_passes'] else 'ÉCHEC'}) | "
            f"R2→{r['R2_pick']:6s} ({'ok' if r['R2_pick_passes'] else 'ÉCHEC'}) | "
            f"R3→{r['R3_pick']!s:6s} ({'ok' if r['R3_pick_passes'] else 'ÉCHEC'}) | {time.perf_counter() - t0:.0f} s",
            flush=True,
        )
    study = [
        r for r in rows if r["seed"] in SEEDS
    ]  # la partition de référence (42) est rapportée à part
    n = len(study)
    summary = {
        rule: {
            "picks": {
                m: sum(r[f"{rule}_pick"] == m for r in study) for m in ("logreg", "tree", None)
            },
            "pick_passes": sum(r[f"{rule}_pick_passes"] for r in study),
            "n": n,
        }
        for rule in ("R1", "R2", "R3")
    }
    summary["model_passes"] = {
        m: sum(r[m]["passes_ca01_ca05"] for r in study) for m in ("logreg", "tree")
    }
    out = {
        "rules": {
            "R1": "meilleure PR-AUC de validation",
            "R2": "meilleur rappel de validation au seuil FPR ≤ cible (égalité : PR-AUC)",
        },
        "params": params,
        "rows": rows,
        "reference_seed": cfg.RANDOM_STATE,
        "summary": summary,
    }
    dump_json(out, cfg.REPORTS_DIR / "selection_study.json")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
