"""Génère docs/plan_de_test.md à partir de la source unique : docs/traceability.json + app/config.py.

Le plan n'est jamais saisi à la main : exigences, tests liés et critères d'acceptation proviennent des
mêmes fichiers que ceux contrôlés par `tests/test_traceability.py`. Les statuts proviennent de
reports/qualification.json lorsqu'il existe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg

ROOT = cfg.PROJECT_ROOT


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += [
        "| " + " | ".join(str(c).replace("\n", " ").replace("|", "\\|") for c in r) + " |"
        for r in rows
    ]
    return "\n".join(out)


def main() -> None:
    trace = json.loads((ROOT / "docs" / "traceability.json").read_text(encoding="utf-8"))
    qpath = ROOT / "reports" / "qualification.json"
    status = (
        json.loads(qpath.read_text(encoding="utf-8"))["traceability_status"]
        if qpath.exists()
        else {}
    )
    rows = []
    for r in trace:
        st = status.get(r["id"])
        rows.append(
            [
                r["id"],
                r["requirement"],
                r["type"],
                r["expected"],
                r["risk"],
                r["priority"],
                f"{st['n_tests']} tests / {st['n_cases']} exéc. — "
                + ("✔" if st["all_passed"] else "✘")
                if st
                else "—",
            ]
        )
    tests_md = "\n".join(
        f"- **{r['id']}** : " + ", ".join(f"`{t}`" for t in r["tests"]) for r in trace
    )
    crit = [
        [c["id"], c["label"], f"{c['op']} {c['value']}", c["severity"], c["rationale"]]
        for c in cfg.ACCEPTANCE_CRITERIA
    ]
    text = f"""# Plan de test — AI-TestGuard

*Document généré par `scripts/make_test_plan.py` à partir de `docs/traceability.json` et `app/config.py`.*

## 1. Périmètre

Système sous test : chargement et validation des données, prétraitement sans état, pipeline scikit-learn
(imputation, `log1p(Amount)`, mise à l'échelle, modèle), réglage et seuil de décision, sauvegarde/rechargement
avec empreinte SHA-256, prédiction (`predict_fraud`, gestionnaire de requête), évaluation, non-régression,
robustesse, surveillance, sécurité opérationnelle et intégration continue.

## 2. Stratégie

Pyramide de tests : unitaires et propriétés (Hypothesis) → intégration (bout en bout, fuites, sauvegarde) →
évaluation statistique du modèle (métriques, intervalles de confiance, graines, partition temporelle) →
robustesse et non-régression → performance et sécurité → CI. Les tests rapides s'exécutent à chaque Pull Request ;
les tests lents (`-m slow`) et ceux qui exigent le jeu Kaggle (`-m full_data`) sont lancés manuellement ou sur calendrier.

## 3. Exigences, risques et traçabilité

{md_table(["ID", "Exigence", "Type de test", "Résultat attendu", "Risque maîtrisé", "Priorité", "Tests liés / statut"], rows)}

## 4. Cas de test par exigence

{tests_md}

## 5. Critères d'acceptation et de rejet

Définis à partir d'exigences métier et **figés avant toute exécution sur le jeu de test** (`app/config.py`).
Règle de décision : un critère **critique** en échec (ou non mesuré) ⇒ REJETÉ ; un critère **majeur** en échec ⇒ À INVESTIGUER ;
sinon APTE À UN PILOTE (sous conditions), les échecs mineurs devenant des réserves.

{md_table(["ID", "Critère", "Seuil", "Gravité", "Justification"], crit)}

## 6. Outils et preuves à conserver

| Outil | Usage | Preuve |
|---|---|---|
| Pytest, pytest-cov | exécution, couverture lignes + branches | `reports/junit.xml`, `reports/coverage.json`, `reports/ci_evidence/pytest_success.txt` |
| Hypothesis | tests de propriétés (mode déterministe) | tests `test_property_*` |
| Ruff | analyse statique et format | `reports/ci_evidence/ruff_check.txt` |
| scikit-learn, joblib | pipeline, sérialisation, SHA-256 | `models/*.json`, `reports/results.json` |
| `scripts/run_mutation.py` | les tests détectent-ils de vrais défauts ? | `reports/mutation.json`, `reports/ci_evidence/pytest_intentional_failure.txt` |
| GitHub Actions | exécution automatique, artefacts | `.github/workflows/tests.yml`, `experiments.yml` |

## 7. Hypothèses de cadrage à valider avec le métier

Volume ≈ 142 400 transactions/jour ; coût d'une fausse alerte 5 € ; plafond de friction 1 fausse alerte pour
1 000 transactions normales ; budget de scoring 20 ms dans une autorisation de paiement.
"""
    (ROOT / "docs" / "plan_de_test.md").write_text(text, encoding="utf-8")
    print("docs/plan_de_test.md écrit :", len(text), "caractères")


if __name__ == "__main__":
    main()
