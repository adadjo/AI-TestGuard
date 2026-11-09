"""Test de mutation artisanal : des défauts RÉALISTES sont injectés un par un dans une copie du
code ; la suite de tests doit échouer (« mutant tué »). Un mutant qui survit signale un test
manquant ou trop faible. Sert aussi de preuve d'« échec volontaire correctement détecté » (CI).

Usage : python scripts/run_mutation.py [--evidence M05]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg
from app.utils import dump_json

IGNORE = shutil.ignore_patterns(
    "data",
    "models",
    "reports",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "htmlcov",
    "*.pyc",
    ".coverage",
    "baseline_full.json",
)
PYTEST = [
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "--no-header",
    "-p",
    "no:cacheprovider",
    "-m",
    "not slow and not full_data",
]

MUTANTS = [
    (
        "M01",
        "app/preprocessing.py",
        "if (amount < 0).any():",
        "if (amount < -1).any():",
        "Borne basse de Amount décalée (−1 au lieu de 0)",
        "Contrat de données",
        1,
    ),
    (
        "M02",
        "app/preprocessing.py",
        "    if infinite:",
        "    if False and infinite:",
        "Contrôle des valeurs infinies désactivé",
        "Contrat de données",
        1,
    ),
    (
        "M03",
        "app/preprocessing.py",
        "    if absent:",
        "    if False and absent:",
        "Contrôle des colonnes obligatoires désactivé",
        "Contrat de données",
        1,
    ),
    (
        "M04",
        "app/preprocessing.py",
        "stratify=y, random_state=random_state",
        "random_state=random_state",
        "Partition sans stratification (premier découpage)",
        "Partition",
        1,
    ),
    (
        "M05",
        "app/training.py",
        "choice = select_threshold(splits.y_val, val_proba, max_fpr)",
        "choice = select_threshold(splits.y_test, pipeline.predict_proba(splits.X_test)[:, 1], max_fpr)",
        "FUITE : seuil réglé sur le jeu de TEST",
        "Fuite de données",
        1,
    ),
    (
        "M06",
        "app/training.py",
        "pipeline = train_model(splits.X_train, splits.y_train, model_name, params, random_state)",
        "pipeline = train_model(pd.concat([splits.X_train, splits.X_val]), pd.concat([splits.y_train, splits.y_val]), model_name, params, random_state)",
        "FUITE : entraînement sur train + validation",
        "Fuite de données",
        1,
    ),
    (
        "M07",
        "app/training.py",
        "feasible = (fpr <= max_fpr) & np.isfinite(thresholds)",
        "feasible = (fpr <= 2 * max_fpr) & np.isfinite(thresholds)",
        "Contrainte de FPR doublée dans la sélection du seuil",
        "Seuil de décision",
        1,
    ),
    (
        "M08",
        "app/prediction.py",
        '"is_fraud": (proba >= thr).astype("int8")',
        '"is_fraud": (proba > thr).astype("int8")',
        "Règle de décision « p ≥ seuil » devenue « p > seuil »",
        "Prédiction",
        1,
    ),
    (
        "M09",
        "app/prediction.py",
        "or (proba < 0).any() or (proba > 1).any():",
        "or (proba < 0).any():",
        "Contrôle de sortie « p ≤ 1 » supprimé",
        "Prédiction",
        1,
    ),
    (
        "M10",
        "app/evaluation.py",
        "precision = tp / (tp + fp) if (tp + fp) else 0.0",
        "precision = tp / (tp + fn) if (tp + fn) else 0.0",
        "Formule de la précision erronée (FN au lieu de FP)",
        "Métriques",
        1,
    ),
    (
        "M11",
        "app/evaluation.py",
        '"fpr": fp / n_neg if n_neg else nan,',
        '"fpr": fp / n if n else nan,',
        "FPR calculé sur l'ensemble au lieu des négatifs",
        "Métriques",
        1,
    ),
    (
        "M12",
        "app/regression.py",
        'if degradation > limits["fail"]:',
        'if degradation > limits["fail"] * 10:',
        "Tolérance de rejet multipliée par 10",
        "Non-régression",
        1,
    ),
    (
        "M13",
        "app/regression.py",
        "if target.exists() and not overwrite:",
        "if False:",
        "Écrasement de la référence autorisé sans décision",
        "Non-régression",
        1,
    ),
    (
        "M14",
        "app/training.py",
        'if meta.get("sha256") != sha256_file(target):',
        "if False:",
        "Vérification d'intégrité SHA-256 désactivée",
        "Sécurité",
        1,
    ),
    (
        "M15",
        "app/prediction.py",
        "        if n_unexpected:",
        "        if False:",
        "Champs supplémentaires acceptés par le gestionnaire",
        "Sécurité",
        1,
    ),
    (
        "M16",
        "app/training.py",
        "keep_empty_features=True",
        "keep_empty_features=False",
        "Colonne entièrement manquante supprimée (dimensions changées)",
        "Prétraitement",
        0,
    ),
    (
        "M17",
        "app/monitoring.py",
        "np.log(actual / expected)",
        "np.log(expected / actual)",
        "Signe du PSI inversé",
        "Surveillance",
        1,
    ),
    (
        "M18",
        "app/robustness.py",
        "scale = eps_v * sigma_ref[cfg.V_COLUMNS].to_numpy()",
        "scale = sigma_ref[cfg.V_COLUMNS].to_numpy()",
        "Amplitude du bruit ignorée (toujours 1 σ)",
        "Robustesse",
        1,
    ),
    (
        "M19",
        "app/training.py",
        "return np.log1p(np.clip(x, 0.0, None))",
        "return np.log1p(x)",
        "Plancher à 0 retiré avant log1p (défense en profondeur)",
        "Prétraitement",
        1,
    ),
    (
        "M20",
        "app/preprocessing.py",
        "if (amount > cfg.AMOUNT_HARD_LIMIT).any():",
        "if (amount > cfg.AMOUNT_HARD_LIMIT * 1e6).any():",
        "Plafond de Amount rendu inopérant",
        "Contrat de données",
        1,
    ),
    (
        "M21",
        "app/preprocessing.py",
        "if v / n_rows > cfg.MAX_MISSING_FRACTION]",
        "if v / n_rows >= cfg.MAX_MISSING_FRACTION]",
        "Limite de valeurs manquantes : > devenu ≥ (cas limite)",
        "Contrat de données",
        1,
    ),
    (
        "M22",
        "app/evaluation.py",
        "half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom",
        "half = z * math.sqrt(p * (1 - p) / n) / denom",
        "Intervalle de Wilson remplacé par celui de Wald",
        "Métriques",
        1,
    ),
    (
        "M23",
        "app/monitoring.py",
        "if abs(z_score) > cfg.ALERT_RATE_Z and relative > cfg.ALERT_RATE_RELATIVE_CHANGE:",
        "if False:",
        "Alerte sur le taux de transactions signalées désactivée",
        "Surveillance",
        1,
    ),
    (
        "M24",
        "app/preprocessing.py",
        'out = df.drop_duplicates(keep="first")',
        "out = df.drop_duplicates(keep=False)",
        "Dédoublonnage supprimant aussi l'original",
        "Partition",
        1,
    ),
    (
        "M25",
        "app/prediction.py",
        "if size > max_bytes:",
        "if size > max_bytes * 1000:",
        "Limite de taille de requête rendue inopérante",
        "Sécurité",
        1,
    ),
]


def run_pytest(cwd: Path, full_output: bool = False) -> subprocess.CompletedProcess:
    cmd = PYTEST if full_output else [*PYTEST, "-x"]
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": "/tmp",
        },
    )


def mutate(tmp: Path, file: str, old: str, new: str, count: int) -> None:
    path = tmp / file
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"motif introuvable dans {file}: {old!r}")
    path.write_text(text.replace(old, new, count if count else -1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--evidence", default="M05", help="mutant dont la sortie complète sert de preuve d'échec"
    )
    ap.add_argument("--output", default="mutation.json")
    ap.add_argument("--only", nargs="*", help="identifiants à exécuter (défaut : tous)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reprend après une interruption (mutants déjà évalués conservés)",
    )
    args = ap.parse_args()
    root = cfg.PROJECT_ROOT
    results, evidence_text = [], ""
    target = cfg.REPORTS_DIR / args.output
    done: set[str] = set()
    if args.resume and target.exists():
        import json

        results = [
            r
            for r in json.loads(target.read_text(encoding="utf-8"))["mutants"]
            if r["killed"] is not None
        ]
        done = {r["id"] for r in results}
        print(f"Reprise : {len(done)} mutants déjà évalués", flush=True)
    # contrôle à vide : sur la copie NON mutée, toute la suite doit passer (sinon « tué » ne prouve rien)
    ctl = Path(tempfile.mkdtemp())
    shutil.copytree(root, ctl / "repo", ignore=IGNORE)
    control = run_pytest(ctl / "repo", full_output=True)
    shutil.rmtree(ctl, ignore_errors=True)
    if control.returncode != 0:
        sys.exit(
            f"Contrôle à vide en échec : la suite doit passer avant mutation.\n{control.stdout[-1500:]}"
        )
    print("Contrôle à vide : suite verte sur le code non muté", flush=True)
    for mid, file, old, new, desc, category, count in MUTANTS:
        if (args.only and mid not in args.only) or mid in done:
            continue
        tmp = Path(tempfile.mkdtemp())
        shutil.copytree(root, tmp / "repo", ignore=IGNORE)
        work = tmp / "repo"
        t0 = time.perf_counter()
        try:
            mutate(work, file, old, new, count)
            proc = run_pytest(work)
            failed = re.findall(r"FAILED (tests/\S+)", proc.stdout)
            error = re.findall(r"ERROR (tests/\S+)", proc.stdout)
            killed = proc.returncode != 0
            first = (failed or error or [None])[0]
            status = "tué" if killed else "SURVIVANT"
            if mid == args.evidence:
                full = run_pytest(work, full_output=True)
                evidence_text = (
                    f"$ pytest -q -m 'not slow and not full_data'   # mutant {mid} : {desc}\n"
                    f"{full.stdout[-3500:]}\n[code de sortie : {full.returncode}]\n"
                )
        except Exception as exc:  # motif introuvable, délai dépassé...
            killed, first, status = None, None, f"ERREUR : {exc}"
        results.append(
            {
                "id": mid,
                "file": file,
                "description": desc,
                "category": category,
                "killed": killed,
                "first_failing_test": first,
                "seconds": time.perf_counter() - t0,
            }
        )
        print(f"{mid} {status:10s} {desc[:62]:62s} -> {first}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        dump_json({"control_passed": True, "mutants": results}, target)
        if mid == args.evidence and evidence_text:
            (cfg.REPORTS_DIR / "ci_evidence" / "pytest_intentional_failure.txt").write_text(
                evidence_text, encoding="utf-8"
            )

    evaluated = [r for r in results if r["killed"] is not None]
    killed_n = sum(bool(r["killed"]) for r in evaluated)
    out = {
        "mutants": results,
        "total": len(evaluated),
        "killed": killed_n,
        "survived": [r["id"] for r in evaluated if not r["killed"]],
        "mutation_score": killed_n / len(evaluated) if evaluated else None,
    }
    dump_json(out, target)
    if evidence_text:
        (cfg.REPORTS_DIR / "ci_evidence" / "pytest_intentional_failure.txt").write_text(
            evidence_text, encoding="utf-8"
        )
    print(f"Score de mutation : {killed_n}/{len(evaluated)}  survivants : {out['survived']}")


if __name__ == "__main__":
    main()
