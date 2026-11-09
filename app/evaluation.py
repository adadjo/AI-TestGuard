"""Évaluation d'un classifieur de fraude sur données fortement déséquilibrées.

Fonctions pures et testées : métriques, intervalles d'incertitude (Wilson, bootstrap
stratifié), table de compromis selon le seuil, analyse de coût, vérification des critères
d'acceptation et verdict de qualification.
"""

from __future__ import annotations

import math
import operator
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from app import config as cfg
from app.prediction import predict_fraud

_OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}
_METRIC_FUNCTIONS = {"average_precision": average_precision_score, "roc_auc": roc_auc_score}


def _as_binary_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} doit être un vecteur à une dimension.")
    if arr.size == 0:
        raise ValueError(f"{name} est vide.")
    if arr.dtype == bool:
        arr = arr.astype(int)
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{name} ne doit contenir que des 0 et des 1.")
    return arr.astype(int)


def evaluate_model(y_true: Any, y_pred: Any, y_proba: Any = None) -> dict[str, Any]:
    """Calcule les métriques de classification binaire (classe positive = fraude).

    Conventions : une métrique indéfinie (ex. rappel sans aucune fraude réelle) vaut NaN et
    non 0, pour ne pas masquer un problème de données ; la précision vaut 0 si aucune alerte.
    """
    yt = _as_binary_array(y_true, "y_true")
    yp = _as_binary_array(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError("y_true et y_pred doivent avoir la même longueur.")

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(yt, yp, labels=[0, 1]).ravel())
    n = tn + fp + fn + tp
    n_pos, n_neg = tp + fn, tn + fp
    nan = math.nan

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_pos if n_pos else nan
    if n_pos == 0:
        f1 = nan
    elif precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0

    metrics: dict[str, Any] = {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_alerts": tp + fp,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / n_neg if n_neg else nan,
        "fnr": fn / n_pos if n_pos else nan,
        "specificity": tn / n_neg if n_neg else nan,
        "mcc": mcc,
        "roc_auc": nan,
        "average_precision": nan,
    }
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype="float64")
        if proba.shape != yt.shape:
            raise ValueError("y_proba et y_true doivent avoir la même longueur.")
        if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
            raise ValueError("y_proba doit être finie et comprise entre 0 et 1.")
        if n_pos and n_neg:
            metrics["roc_auc"] = float(roc_auc_score(yt, proba))
            metrics["average_precision"] = float(average_precision_score(yt, proba))
    return metrics


def evaluate_pipeline(model: Any, X: Any, y: Any, threshold: float) -> tuple[dict, np.ndarray]:
    """Évalue un modèle via la fonction de service ``predict_fraud`` (contrat inclus)."""
    scored = predict_fraud(model, X, threshold=threshold)
    proba = scored["fraud_probability"].to_numpy()
    pred = scored["is_fraud"].to_numpy()
    return evaluate_model(np.asarray(y), pred, proba), proba


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Intervalle de confiance de Wilson (95 % par défaut) pour une proportion k/n."""
    if n <= 0:
        return (math.nan, math.nan)
    if not 0 <= k <= n:
        raise ValueError("k doit être compris entre 0 et n.")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # bornées par l'estimation : évite un écart d'arrondi (~1e-18) aux extrémités k=0 et k=n
    return (min(max(0.0, center - half), p), max(min(1.0, center + half), p))


def bootstrap_compare(
    y_true: Any,
    probas: dict[str, Any],
    metrics: tuple[str, ...] = ("average_precision", "roc_auc"),
    n_boot: int = 1000,
    seed: int = cfg.RANDOM_STATE,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap STRATIFIÉ (effectifs de classes conservés) de métriques de classement.

    Les mêmes rééchantillons sont appliqués à tous les modèles : les différences par paires
    sont donc des différences appariées (IC de la différence, probabilité que A > B).
    """
    yt = _as_binary_array(y_true, "y_true")
    scores = {name: np.asarray(p, dtype="float64") for name, p in probas.items()}
    pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Le bootstrap exige les deux classes.")
    rng = np.random.default_rng(seed)
    draws = {(m, name): np.empty(n_boot) for m in metrics for name in scores}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for m in metrics:
            fn = _METRIC_FUNCTIONS[m]
            for name, p in scores.items():
                draws[(m, name)][b] = fn(yt[idx], p[idx])

    def summarize(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        return {
            "low": float(low),
            "high": float(high),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    out: dict[str, Any] = {"n_boot": n_boot, "models": {}, "differences": {}}
    for name in scores:
        out["models"][name] = {m: summarize(draws[(m, name)]) for m in metrics}
    names = list(scores)
    for i, a in enumerate(names):
        for b_name in names[i + 1 :]:
            entry = {}
            for m in metrics:
                diff = draws[(m, a)] - draws[(m, b_name)]
                entry[m] = {**summarize(diff), "prob_a_better": float((diff > 0).mean())}
            out["differences"][f"{a}_minus_{b_name}"] = entry
    return out


def threshold_table(
    y_true: Any, y_proba: Any, thresholds: Any, amounts: Any = None
) -> list[dict[str, Any]]:
    """Compromis FP/FN pour une liste de seuils (à calculer sur la VALIDATION)."""
    yt = _as_binary_array(y_true, "y_true")
    proba = np.asarray(y_proba, dtype="float64")
    amt = None if amounts is None else np.asarray(amounts, dtype="float64")
    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        m = evaluate_model(yt, pred, proba)
        row = {
            "threshold": float(thr),
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
            "precision": m["precision"],
            "recall": m["recall"],
            "fpr": m["fpr"],
        }
        if amt is not None:
            row.update(cost_summary(yt, pred, amt))
        rows.append(row)
    return rows


def cost_summary(
    y_true: Any, y_pred: Any, amounts: Any, cost_false_alert: float = cfg.COST_FALSE_ALERT_EUR
) -> dict[str, float]:
    """Vue économique : fraude évitée, fraude manquée, coût des fausses alertes."""
    yt = _as_binary_array(y_true, "y_true")
    yp = _as_binary_array(y_pred, "y_pred")
    amt = np.asarray(amounts, dtype="float64")
    total = float(amt[yt == 1].sum())
    detected = float(amt[(yt == 1) & (yp == 1)].sum())
    n_fp = int(((yt == 0) & (yp == 1)).sum())
    false_alert_cost = n_fp * cost_false_alert
    return {
        "fraud_amount_total": total,
        "fraud_amount_detected": detected,
        "fraud_amount_missed": total - detected,
        "false_alert_cost": false_alert_cost,
        "total_cost": (total - detected) + false_alert_cost,
    }


def check_criteria(
    measured: dict[str, Any],
    criteria: list[dict] | None = None,
    intervals: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Compare les mesures aux critères d'acceptation ; signale l'incertitude statistique."""
    criteria = cfg.ACCEPTANCE_CRITERIA if criteria is None else criteria
    intervals = intervals or {}
    rows = []
    for criterion in criteria:
        value = measured.get(criterion["metric"])
        if value is None or (isinstance(value, float) and math.isnan(value)):
            passed, status = None, "NON MESURÉ"
        else:
            passed = bool(_OPS[criterion["op"]](value, criterion["value"]))
            status = "CONFORME" if passed else "NON CONFORME"
        interval = intervals.get(criterion["metric"])
        straddles = None
        if interval is not None and passed is not None:
            straddles = bool(interval[0] < criterion["value"] < interval[1])
        rows.append(
            {
                **criterion,
                "measured": value,
                "passed": passed,
                "status": status,
                "ci": interval,
                "ci_straddles_threshold": straddles,
            }
        )
    return rows


PERFORMANCE_CRITERIA_IDS = ("CA01", "CA02", "CA03", "CA04", "CA05")


def meets_performance_criteria(metrics: dict[str, Any]) -> bool:
    """Vrai si un jeu de métriques satisfait CA01-CA05 (rappel, FPR, précision, PR-AUC, ROC-AUC).

    Appliquée aux métriques de VALIDATION, elle sert à qualifier un candidat avant de le choisir.
    """
    selected = [c for c in cfg.ACCEPTANCE_CRITERIA if c["id"] in PERFORMANCE_CRITERIA_IDS]
    return all(row["status"] == "CONFORME" for row in check_criteria(metrics, selected))


def qualification_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Règle de décision : critique en échec => REJETÉ ; majeur => À INVESTIGUER ; sinon APTE."""
    failed = {
        sev: [r["id"] for r in rows if r["severity"] == sev and r["status"] != "CONFORME"]
        for sev in ("critique", "majeur", "mineur")
    }
    if failed["critique"]:
        verdict = "REJETÉ"
    elif failed["majeur"]:
        verdict = "À INVESTIGUER"
    else:
        verdict = "APTE À UN PILOTE (sous conditions)"
    uncertain = [r["id"] for r in rows if r.get("ci_straddles_threshold")]
    return {"verdict": verdict, "failed": failed, "uncertain_criteria": uncertain}


def segment_report(
    y_true: Any, y_pred: Any, groups: Any, min_positives: int = 30
) -> list[dict[str, Any]]:
    """Métriques par segment avec intervalles de Wilson et indicateur de taille d'échantillon.

    Sert à comparer rappel / FPR / précision entre groupes. Un segment avec moins de
    ``min_positives`` fraudes est marqué ``sufficient_sample = False`` : aucune conclusion
    d'équité ne doit en être tirée.
    """
    yt = _as_binary_array(y_true, "y_true")
    yp = _as_binary_array(y_pred, "y_pred")
    labels = np.asarray(groups)
    if labels.shape != yt.shape or yt.shape != yp.shape:
        raise ValueError("y_true, y_pred et groups doivent avoir la même longueur.")
    rows = []
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        m = evaluate_model(yt[mask], yp[mask])
        rows.append(
            {
                "group": str(label),
                "n": m["n"],
                "n_positive": m["n_positive"],
                "n_negative": m["n_negative"],
                "recall": m["recall"],
                "recall_ci": wilson_interval(m["tp"], m["n_positive"]),
                "fpr": m["fpr"],
                "fpr_ci": wilson_interval(m["fp"], m["n_negative"]),
                "precision": m["precision"],
                "precision_ci": wilson_interval(m["tp"], m["tp"] + m["fp"]),
                "sufficient_sample": m["n_positive"] >= min_positives,
            }
        )
    return rows
