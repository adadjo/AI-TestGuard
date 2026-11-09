# AI-TestGuard

**Test logiciel et assurance qualité d'un modèle de détection de fraude bancaire** — Évaluation formative n° 1 (Tests logiciels).

Le projet ne cherche pas le meilleur score prédictif : il démontre, avec des **tests reproductibles et des résultats mesurables**,
dans quelles conditions un modèle de détection de fraude (jeu Kaggle *Credit Card Fraud Detection*) peut être considéré comme
acceptable pour une phase pilote, et ce qui manque avant une mise en production. Le rapport complet de qualification est
fourni à part (`Rapport_AI-TestGuard.docx`) ; tous ses chiffres sont produits par les scripts de ce dépôt et archivés dans `reports/`.

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt                         # versions figées (Python >= 3.11)

pytest -m "not slow and not full_data" --cov=app        # tests rapides = ceux de la CI (≈ 10 s)
ruff check . && ruff format --check .                   # qualité du code

# Données : voir data/README.md (jeu Kaggle non versionné)
pytest                                                  # suite complète (+ tests lents et jeu réel)
python scripts/run_all.py                               # toutes les expériences et preuves (≈ 15 min)
python scripts/run_all.py --synthetic                   # démonstration sans le jeu Kaggle (proxy étiqueté, résultats NON réels)
```

## Organisation

| Dossier | Contenu |
|---|---|
| `app/` | code testé : contrat de données et partitions (`preprocessing`), pipeline, seuil, sauvegarde SHA-256 (`training`), prédiction et gestionnaire de requête (`prediction`), métriques et critères (`evaluation`), non-régression, robustesse, surveillance, sécurité |
| `tests/` | 260 tests : unitaires, propriétés Hypothesis, intégration, robustesse, non-régression, sécurité, CI, traçabilité, jeu réel |
| `scripts/` | expériences et preuves : `run_all.py`, `run_experiments.py`, `run_robustness.py`, `run_regression.py`, `run_performance.py`, `run_monitoring.py`, `run_mutation.py`, `qualify.py`… |
| `baseline/` | références de non-régression **figées** (jamais recalculées automatiquement) |
| `docs/` | `plan_de_test.md` (généré) et `traceability.json` (source unique des exigences T01–T23) |
| `reports/` | résultats générés (`*.json`), figures, preuves d'exécution (`ci_evidence/`) |
| `.github/workflows/` | `tests.yml` (chaque Pull Request) et `experiments.yml` (manuel / hebdomadaire) |

## Principes de qualité appliqués

- **Pas de fuite de données** : partition stratifiée avant tout `fit` ; imputation et mise à l'échelle dans un `Pipeline` ajusté sur l'entraînement ;
  hyperparamètres et seuil réglés sur la validation ; jeu de test utilisé une seule fois (provenance des index vérifiée par test).
- **Critères d'acceptation fixés a priori** (`app/config.py`) et décision mécanique : critique en échec ⇒ REJETÉ, majeur ⇒ À INVESTIGUER, sinon APTE À UN PILOTE.
- **Incertitude explicite** : intervalles de Wilson et bootstrap ; un critère conforme dont l'intervalle chevauche le seuil est signalé.
- **Contrat de données strict** : rejeter ou corriger selon une règle documentée, jamais deviner ; les messages d'erreur ne citent jamais de valeurs.
- **Artefacts vérifiés** : le modèle sérialisé est accompagné d'une empreinte SHA-256 contrôlée *avant* la désérialisation.
- **Les tests vérifient des comportements** : un test de mutation (`scripts/run_mutation.py`) injecte 25 défauts réalistes et vérifie que la suite échoue.

## Intégration continue

`tests.yml` exécute à chaque Pull Request : Ruff (lint + format), puis Pytest sur Python 3.11 et 3.12 avec couverture (seuil 85 %),
rapport JUnit et artefacts publiés même en cas d'échec ; aucun `continue-on-error` (vérifié par `tests/test_ci_config.py`).
Protection de la branche `main` : voir §10.4 du rapport. Les expériences coûteuses (`experiments.yml`) sont lancées manuellement ou chaque lundi.

## Limites à connaître

Deux jours de transactions de 2013, 71 fraudes dans le jeu de test, variables anonymisées, aucune variable de groupe (donc **aucune conclusion
d'équité**), pas d'API déployée. La décision du rapport est « apte à un pilote en mode ombre », **pas** une autorisation de déploiement.
