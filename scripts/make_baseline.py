"""Crée une RÉFÉRENCE (baseline) de non-régression. N'écrase jamais sans --overwrite.

Exemples :
    python scripts/make_baseline.py --kind ci --confirm      # données synthétiques (CI)
    python scripts/make_baseline.py --kind full --confirm    # jeu Kaggle (nécessite data/)

La référence est une décision explicite : elle n'est JAMAIS recalculée automatiquement.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from app import config as cfg
from app.preprocessing import dataset_fingerprint, deduplicate, load_data
from app.regression import build_baseline_record, write_baseline
from app.synthetic import make_synthetic_transactions
from app.training import run_training_pipeline

CI_DATASET = {"n_rows": 20_000, "fraud_rate": 0.01, "seed": 2024}
CI_MODEL = {"name": "logreg", "params": {}, "max_fpr": 0.005}


def validation_set_hash(index) -> str:
    return hashlib.sha256(",".join(map(str, sorted(index))).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["ci", "full", "proxy"], required=True)
    parser.add_argument("--confirm", action="store_true", help="confirme la décision explicite")
    parser.add_argument("--overwrite", action="store_true", help="remplace une référence existante")
    args = parser.parse_args()
    if not args.confirm:
        sys.exit("Refus : ajouter --confirm (la référence est une décision explicite).")

    seed = cfg.RANDOM_STATE
    if args.kind == "ci":
        df = make_synthetic_transactions(**CI_DATASET)
        dataset = {
            **dataset_fingerprint(df),
            "source": "synthétique",
            "synthetic_params": CI_DATASET,
        }
        name, params, max_fpr = CI_MODEL["name"], CI_MODEL["params"], CI_MODEL["max_fpr"]
        target = cfg.BASELINE_DIR / "baseline_ci.json"
    elif args.kind == "proxy":
        # référence JETABLE sur le proxy synthétique (démonstration sans le jeu Kaggle) : écrite dans reports/
        selected = json.loads((cfg.REPORTS_DIR / "results.json").read_text(encoding="utf-8"))[
            "selected_model"
        ]
        df, removed = deduplicate(make_synthetic_transactions(60_000, 0.005, seed=1))
        dataset = {
            **dataset_fingerprint(df),
            "source": "synthétique",
            "duplicates_removed": removed,
        }
        name, params, max_fpr = selected["name"], selected["params"], cfg.THRESHOLD_FPR_TARGET
        target = cfg.REPORTS_DIR / "baseline_proxy.json"
    else:
        results = json.loads((cfg.REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
        selected = results["selected_model"]
        df, removed = deduplicate(load_data())
        dataset = {
            **dataset_fingerprint(df, cfg.DATA_PATH),
            "source": "Kaggle mlg-ulb/creditcardfraud",
            "duplicates_removed": removed,
        }
        name, params = selected["name"], selected["params"]
        max_fpr = cfg.THRESHOLD_FPR_TARGET
        target = cfg.BASELINE_DIR / "baseline_full.json"

    run = run_training_pipeline(df, name, params, max_fpr=max_fpr, random_state=seed)
    dataset["split_sizes"] = run.splits.sizes()
    dataset["validation_set_sha256"] = validation_set_hash(run.splits.X_val.index)
    record = build_baseline_record(
        name,
        params,
        run.threshold,
        {"split": seed, "model": seed},
        dataset,
        run.val_metrics,
        datetime.now(UTC).isoformat(timespec="seconds"),
        max_fpr=max_fpr,
    )
    path = write_baseline(record, target, overwrite=args.overwrite)
    print(f"Référence écrite : {path}")
    print({k: round(v, 4) for k, v in record["validation_metrics"].items()})


if __name__ == "__main__":
    main()
