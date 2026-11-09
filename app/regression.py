"""Non-régression : référence (baseline) figée et comparaison à une nouvelle version.

Règle essentielle : la référence n'est JAMAIS recalculée automatiquement. Elle est écrite par
une action explicite (``scripts/make_baseline.py --confirm``), refuse l'écrasement, et les
tests échouent si elle est absente, au lieu de la recréer silencieusement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app import __version__
from app import config as cfg
from app.utils import dump_json, environment_info, git_commit

REQUIRED_BASELINE_FIELDS = (
    "schema_version",
    "created_utc",
    "code_version",
    "git_commit",
    "model",
    "seeds",
    "environment",
    "dataset",
    "validation_metrics",
    "tolerances",
)
DECISION_ACCEPT = "ACCEPTER"
DECISION_INVESTIGATE = "INVESTIGUER"
DECISION_REJECT = "REJETER"
_RANK = {"ok": 0, "warn": 1, "fail": 2}


def build_baseline_record(
    model_name: str,
    params: dict,
    threshold: float,
    seeds: dict[str, int],
    dataset: dict[str, Any],
    validation_metrics: dict[str, float],
    created_utc: str,
    tolerances: dict | None = None,
    max_fpr: float = cfg.THRESHOLD_FPR_TARGET,
) -> dict[str, Any]:
    """Assemble l'enregistrement de référence (code, paramètres, versions, graines, seuil...)."""
    return {
        "schema_version": 1,
        "created_utc": created_utc,
        "code_version": __version__,
        "git_commit": git_commit(),
        "model": {
            "name": model_name,
            "params": params,
            "threshold": float(threshold),
            "max_fpr": float(max_fpr),
            "features": cfg.MODEL_FEATURES,
        },
        "seeds": seeds,
        "environment": environment_info(),
        "dataset": dataset,
        "validation_metrics": {k: validation_metrics[k] for k in cfg.REGRESSION_TOLERANCES},
        "tolerances": tolerances or cfg.REGRESSION_TOLERANCES,
    }


def validate_baseline_record(record: dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_BASELINE_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Référence incomplète, champs absents : {missing}")
    absent_metrics = [m for m in cfg.REGRESSION_TOLERANCES if m not in record["validation_metrics"]]
    if absent_metrics:
        raise ValueError(f"Métriques absentes de la référence : {absent_metrics}")


def write_baseline(record: dict[str, Any], path: str | Path, overwrite: bool = False) -> Path:
    """Écrit la référence ; refuse d'écraser un fichier existant sans ``overwrite=True``."""
    validate_baseline_record(record)
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"La référence {target.name} existe déjà : elle ne doit pas être recalculée "
            "automatiquement (utiliser overwrite=True après décision explicite)."
        )
    return dump_json(record, target)


def load_baseline(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"Référence absente : {target.name}. Elle n'est jamais créée automatiquement ; "
            "exécuter scripts/make_baseline.py --confirm après décision explicite."
        )
    record = json.loads(target.read_text(encoding="utf-8"))
    validate_baseline_record(record)
    return record


def environment_differences(baseline_env: dict, current_env: dict | None = None) -> dict:
    """Versions de bibliothèques qui diffèrent de celles de la référence (information)."""
    current = (current_env or environment_info())["libraries"]
    ref = baseline_env.get("libraries", {})
    return {
        k: {"baseline": ref.get(k), "current": current.get(k)}
        for k in ref
        if ref.get(k) != current.get(k)
    }


def compare_to_baseline(
    candidate: dict[str, float],
    baseline: dict[str, float],
    tolerances: dict | None = None,
    suspicious_gain: float = cfg.SUSPICIOUS_GAIN,
) -> dict[str, Any]:
    """Décide ACCEPTER / INVESTIGUER / REJETER en comparant aux tolérances de la référence.

    * une dégradation > ``fail`` sur une métrique => REJETER ;
    * une dégradation > ``warn`` (ou un gain de PR-AUC > ``suspicious_gain``, signe possible de
      fuite de données) => INVESTIGUER ;
    * sinon ACCEPTER.
    """
    tol = tolerances or cfg.REGRESSION_TOLERANCES
    details: dict[str, dict[str, Any]] = {}
    worst = "ok"
    reasons: list[str] = []
    for metric, limits in tol.items():
        if metric not in candidate or metric not in baseline:
            raise KeyError(f"Métrique manquante pour la comparaison : {metric}")
        cand, ref = float(candidate[metric]), float(baseline[metric])
        if math.isnan(cand) or math.isnan(ref):
            raise ValueError(f"Valeur NaN pour la métrique {metric}")
        delta = cand - ref
        degradation = -delta if limits["higher_is_better"] else delta
        status = "ok"
        if degradation > limits["fail"]:
            status = "fail"
        elif degradation > limits["warn"]:
            status = "warn"
        if metric == "average_precision" and delta > suspicious_gain:
            status = "warn"
            reasons.append(
                f"Gain anormal de {metric} (+{delta:.3f}) : vérifier l'absence de fuite de données."
            )
        if status != "ok" and degradation > 0:
            reasons.append(f"{metric} : dégradation de {degradation:.4f} ({status}).")
        details[metric] = {
            "baseline": ref,
            "candidate": cand,
            "delta": delta,
            "degradation": degradation,
            "status": status,
        }
        if _RANK[status] > _RANK[worst]:
            worst = status
    decision = {"ok": DECISION_ACCEPT, "warn": DECISION_INVESTIGATE, "fail": DECISION_REJECT}[worst]
    return {"decision": decision, "details": details, "reasons": reasons}
