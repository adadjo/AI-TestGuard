"""Exécute toute la chaîne de qualification, dans l'ordre (≈ 15 min avec le jeu Kaggle complet).

python scripts/run_all.py                              # jeu réel (data/creditcard.csv)
python scripts/run_all.py --allow-synthetic-fallback   # proxy synthétique si le jeu est absent
python scripts/run_all.py --with-mutation              # ajoute le test de mutation (≈ 4 min)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"


def step(label: str, *cmd: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)
    print(f"--- {label} : {time.perf_counter() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-synthetic-fallback", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--with-mutation", action="store_true")
    args = ap.parse_args()
    flags = [
        f
        for f, on in (
            ("--allow-synthetic-fallback", args.allow_synthetic_fallback),
            ("--synthetic", args.synthetic),
        )
        if on
    ]
    step("Évaluation des modèles (Activités 3, 4, 6.1)", str(S / "run_experiments.py"), *flags)
    step("Étude de la règle de sélection", str(S / "run_selection_study.py"))
    synthetic_mode = args.synthetic or (
        args.allow_synthetic_fallback and not (ROOT / "data" / "creditcard.csv").exists()
    )
    if synthetic_mode:
        step(
            "Référence jetable sur le proxy synthétique",
            str(S / "make_baseline.py"),
            "--kind",
            "proxy",
            "--confirm",
            "--overwrite",
        )
    elif not (ROOT / "baseline" / "baseline_full.json").exists():
        step(
            "Référence de non-régression (décision explicite)",
            str(S / "make_baseline.py"),
            "--kind",
            "full",
            "--confirm",
        )
    step("Robustesse (Activité 5)", str(S / "run_robustness.py"))
    step("Non-régression (Activité 6.2)", str(S / "run_regression.py"))
    step("Performance (Activité 8)", str(S / "run_performance.py"))
    step("Sécurité opérationnelle (Activité 8)", str(S / "run_security.py"))
    step("Surveillance (Activité 10)", str(S / "run_monitoring.py"))
    if args.with_mutation:
        step("Test de mutation / échec volontaire (Activité 9)", str(S / "run_mutation.py"))
    step("Qualification (lint, tests, couverture, critères)", str(S / "qualify.py"))


if __name__ == "__main__":
    main()
