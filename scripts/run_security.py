"""Sécurité opérationnelle (Activité 8) : secrets, données, journaux, intégrité des artefacts."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg
from app.prediction import handle_prediction_request
from app.security import find_large_files, missing_gitignore_entries, scan_for_secrets
from app.training import ModelIntegrityError, load_model
from app.utils import dump_json

SENTINELS = {"amount": 123456.789, "v1": 31.41592653, "field": "champ_secret_abc"}


def main() -> None:
    checks = []

    findings = scan_for_secrets(cfg.PROJECT_ROOT)
    checks.append(
        {
            "id": "S1",
            "check": "Aucun secret dans le dépôt (motifs : clés AWS, jetons GitHub/Slack, "
            "clés privées, clé Kaggle, affectations password/token/api_key)",
            "result": len(findings) == 0,
            "detail": f"{len(findings)} détection(s)",
        }
    )
    missing = missing_gitignore_entries(cfg.PROJECT_ROOT)
    checks.append(
        {
            "id": "S2",
            "check": ".gitignore protège données, modèles, .env et kaggle.json",
            "result": not missing,
            "detail": f"entrées manquantes : {missing}",
        }
    )
    large = find_large_files(cfg.PROJECT_ROOT)
    checks.append(
        {
            "id": "S3",
            "check": "Aucun fichier > 5 Mo hors data/ et models/",
            "result": not large,
            "detail": f"{len(large)} fichier(s)",
        }
    )
    tracked_csv = None
    if shutil.which("git") and (cfg.PROJECT_ROOT / ".git").exists():
        files = subprocess.run(
            ["git", "ls-files", "data"], cwd=cfg.PROJECT_ROOT, capture_output=True, text=True
        ).stdout.split()
        tracked_csv = [f for f in files if f.endswith(".csv")]
    checks.append(
        {
            "id": "S4",
            "check": "Le jeu Kaggle (creditcard.csv) n'est pas suivi par Git",
            "result": not tracked_csv,
            "detail": "dépôt Git non initialisé ici : contrôlé par .gitignore (S2)"
            if tracked_csv is None
            else f"{len(tracked_csv)} CSV suivi(s)",
        }
    )

    # journaux : aucune valeur d'entrée ne doit apparaître
    model, thr, _ = load_model(cfg.MODELS_DIR / "logreg_v1.joblib")
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.DEBUG)
    logger = logging.getLogger("aitestguard")
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    payload = dict.fromkeys(cfg.FEATURE_COLUMNS, 0.1)
    payload.update({"Amount": SENTINELS["amount"], "V1": SENTINELS["v1"], "Time": 1000.0})
    handle_prediction_request(payload, model, thr)
    handle_prediction_request({**payload, "V2": "texte"}, model, thr)
    handle_prediction_request({**payload, SENTINELS["field"]: 1.0}, model, thr)
    handle_prediction_request({**payload, "Amount": -SENTINELS["amount"]}, model, thr)
    handle_prediction_request(json.dumps(payload)[:-4], model, thr)
    logger.removeHandler(handler)
    logger.setLevel(old_level)
    text = "\n".join(records)
    leaks = [k for k, v in SENTINELS.items() if str(v)[:7] in text]
    checks.append(
        {
            "id": "S5",
            "check": "Les journaux ne contiennent aucune valeur d'entrée (5 requêtes : valide, "
            "type invalide, champ inconnu, montant négatif, JSON tronqué)",
            "result": not leaks,
            "detail": f"{len(records)} lignes de journal examinées ; fuites : {leaks}",
        }
    )

    # intégrité des artefacts
    tmp = Path(tempfile.mkdtemp())
    src = cfg.MODELS_DIR / "logreg_v1.joblib"
    shutil.copy(src, tmp / "m.joblib")
    shutil.copy(src.with_suffix(".json"), tmp / "m.json")
    content = bytearray((tmp / "m.joblib").read_bytes())
    content[len(content) // 2] ^= 0xFF
    (tmp / "m.joblib").write_bytes(bytes(content))
    try:
        load_model(tmp / "m.joblib")
        tamper_detected = False
    except ModelIntegrityError:
        tamper_detected = True
    (tmp / "m.json").unlink()
    shutil.copy(src, tmp / "n.joblib")
    try:
        load_model(tmp / "n.joblib")
        missing_hash_refused = False
    except ModelIntegrityError:
        missing_hash_refused = True
    shutil.rmtree(tmp, ignore_errors=True)
    checks.append(
        {
            "id": "S6",
            "check": "Un modèle altéré (1 octet) est refusé AVANT désérialisation (SHA-256)",
            "result": tamper_detected,
            "detail": "ModelIntegrityError levée",
        }
    )
    checks.append(
        {
            "id": "S7",
            "check": "Un modèle sans fichier d'empreinte est refusé",
            "result": missing_hash_refused,
            "detail": "ModelIntegrityError levée",
        }
    )

    out = {
        "checks": checks,
        "findings": findings,
        "criteria_feed": {"secrets_found": len(findings), "sensitive_values_in_logs": len(leaks)},
        "residual_risks": [
            "joblib/pickle : l'empreinte SHA-256 protège contre l'altération accidentelle ou le remplacement du seul "
            "fichier ; un attaquant qui remplace aussi le fichier d'empreinte la contourne (prévoir signature/HMAC "
            "et un magasin d'artefacts de confiance).",
            "Aucun audit des dépendances (pip-audit) n'a pu être exécuté : accès réseau restreint ; à ajouter en CI.",
            "Pas d'API HTTP déployée : authentification, TLS, limitation de débit et journalisation d'accès non évalués.",
        ],
    }
    dump_json(out, cfg.REPORTS_DIR / "security.json")
    for c in checks:
        print(("OK  " if c["result"] else "ÉCHEC"), c["id"], c["check"][:80])


if __name__ == "__main__":
    main()
