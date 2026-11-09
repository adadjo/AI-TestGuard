"""Configuration centrale : contrat de données, graines et critères d'acceptation.

Les critères d'acceptation (``ACCEPTANCE_CRITERIA``) et les tolérances de non-régression
ont été définis à partir des exigences métier AVANT toute exécution sur le jeu de test
(voir ``docs/plan_de_test.md``). Ils ne doivent jamais être modifiés après observation
des résultats : toute évolution passe par une Pull Request relue et justifiée par une
exigence, pas par un score.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_DIR = PROJECT_ROOT / "baseline"
MODELS_DIR = PROJECT_ROOT / "models"

# ------------------------------------------------------------------ reproductibilité
RANDOM_STATE = 42

# ----------------------------------------------------------------- contrat de données
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *V_COLUMNS, "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
# Time est exclu du modèle : temps écoulé depuis le début de l'enregistrement, donc non
# transférable en production et source possible de fuite temporelle (cf. rapport, §4.5).
MODEL_FEATURES = [*V_COLUMNS, "Amount"]

MAX_FILE_BYTES = 500 * 1024**2  # un fichier plus gros est rejeté avant lecture
MAX_PAYLOAD_BYTES = 16 * 1024  # taille maximale d'une requête de prédiction unitaire
MAX_MISSING_FRACTION = 0.05  # par colonne ; au-delà : rejet (sinon imputation médiane)
AMOUNT_ALERT_THRESHOLD = 30_000.0  # > maximum observé (25 691,16) : traité avec alerte
AMOUNT_HARD_LIMIT = 1_000_000.0  # au-delà : rejet (hors contrat)
V_ABS_HARD_LIMIT = 1_000.0  # composantes PCA : |Vi| plus grand = erreur d'unité/encodage

# ------------------------------------------------------------------------ décision
BUSINESS_MAX_FPR = 0.001  # plafond métier : 1 fausse alerte / 1 000 transactions légitimes
THRESHOLD_FPR_TARGET = 0.0008  # cible sur validation = 80 % du plafond (marge de sécurité)
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# Grilles d'hyperparamètres (réglées sur la validation uniquement)
GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    "tree": {
        "max_depth": [4, 6, 8, 12],
        "min_samples_leaf": [5, 20],
        "class_weight": [None, "balanced"],
    },
}

# Hypothèses économiques (à valider avec le métier ; utilisées pour l'analyse de coût)
COST_FALSE_ALERT_EUR = 5.0
TRANSACTIONS_PER_DAY = 142_500  # 284 807 transactions sur ~2 jours dans le jeu Kaggle

# Surveillance
PSI_WARN = 0.10
PSI_ALERT = 0.25
ALERT_RATE_RELATIVE_CHANGE = 0.5  # variation relative minimale du taux d'alertes
ALERT_RATE_Z = 3.0  # et écart > 3 écarts-types (loi binomiale) : évite les fausses alertes

# ------------------------------------------------------------ non-régression
# degradation = baisse (ou hausse pour fpr) tolérée par rapport à la référence figée.
REGRESSION_TOLERANCES = {
    "average_precision": {"warn": 0.010, "fail": 0.020, "higher_is_better": True},
    "roc_auc": {"warn": 0.005, "fail": 0.010, "higher_is_better": True},
    "recall": {"warn": 0.015, "fail": 0.030, "higher_is_better": True},
    "precision": {"warn": 0.030, "fail": 0.050, "higher_is_better": True},
    "fpr": {"warn": 0.0001, "fail": 0.0002, "higher_is_better": False},
}
SUSPICIOUS_GAIN = 0.05  # un gain de PR-AUC > 0,05 d'un coup est suspect (fuite ?) -> investiguer

# ------------------------------------------------- critères d'acceptation (figés a priori)
# severity : critique (un échec => REJET), majeur (un échec => INVESTIGATION), mineur (réserve)
ACCEPTANCE_CRITERIA = [
    {
        "id": "CA01",
        "label": "Rappel au seuil figé (jeu de test)",
        "metric": "recall",
        "op": ">=",
        "value": 0.75,
        "severity": "critique",
        "rationale": "Au moins 3 fraudes sur 4 interceptées ; en deçà, la perte résiduelle "
        "rend le modèle peu utile face à des règles expertes.",
    },
    {
        "id": "CA02",
        "label": "Taux de faux positifs (jeu de test)",
        "metric": "fpr",
        "op": "<=",
        "value": 0.001,
        "severity": "critique",
        "rationale": "Au plus 1 transaction légitime sur 1 000 bloquée à tort "
        "(≈ 140 fausses alertes/jour pour 142 500 transactions/jour).",
    },
    {
        "id": "CA03",
        "label": "Précision au seuil figé (jeu de test)",
        "metric": "precision",
        "op": ">=",
        "value": 0.50,
        "severity": "majeur",
        "rationale": "Au moins une alerte sur deux doit être une vraie fraude pour que les "
        "analystes gardent confiance dans les alertes.",
    },
    {
        "id": "CA04",
        "label": "PR-AUC / Average Precision (jeu de test)",
        "metric": "average_precision",
        "op": ">=",
        "value": 0.65,
        "severity": "majeur",
        "rationale": "Qualité de classement indépendante du seuil ; ≈ 380 fois la prévalence "
        "(0,17 %), seuil d'un modèle réellement utile.",
    },
    {
        "id": "CA05",
        "label": "ROC-AUC (jeu de test)",
        "metric": "roc_auc",
        "op": ">=",
        "value": 0.95,
        "severity": "majeur",
        "rationale": "Pouvoir de séparation global minimal attendu d'un détecteur de fraude "
        "sur ce type de données.",
    },
    {
        "id": "CA06",
        "label": "Écart de généralisation PR-AUC (entraînement − test)",
        "metric": "generalization_gap_ap",
        "op": "<=",
        "value": 0.15,
        "severity": "majeur",
        "rationale": "Un écart supérieur à 0,15 signale un sur-apprentissage.",
    },
    {
        "id": "CA07",
        "label": "Reproductibilité : écart max entre deux entraînements",
        "metric": "repro_max_abs_diff",
        "op": "<=",
        "value": 1e-9,
        "severity": "critique",
        "rationale": "Mêmes données + mêmes graines = mêmes probabilités (tolérance numérique).",
    },
    {
        "id": "CA08",
        "label": "Cohérence avant/après sauvegarde-rechargement",
        "metric": "reload_max_abs_diff",
        "op": "<=",
        "value": 1e-12,
        "severity": "critique",
        "rationale": "L'artefact déployé doit se comporter exactement comme celui qui a été testé.",
    },
    {
        "id": "CA09",
        "label": "Robustesse : taux de changement de classe (perturbation modérée)",
        "metric": "perturbation_flip_rate",
        "op": "<=",
        "value": 0.005,
        "severity": "majeur",
        "rationale": "Moins de 0,5 % des décisions peuvent changer sous un bruit réaliste "
        "(0,05 σ sur V1-V28 ; ±5 % sur Amount).",
    },
    {
        "id": "CA10",
        "label": "Robustesse : baisse de rappel (perturbation modérée)",
        "metric": "perturbation_recall_drop",
        "op": "<=",
        "value": 0.05,
        "severity": "majeur",
        "rationale": "Le bruit ne doit pas faire perdre plus de 5 points de rappel.",
    },
    {
        "id": "CA11",
        "label": "Entrées invalides acceptées à tort",
        "metric": "invalid_inputs_accepted",
        "op": "<=",
        "value": 0,
        "severity": "critique",
        "rationale": "Toute entrée hors contrat doit être rejetée explicitement.",
    },
    {
        "id": "CA12",
        "label": "Latence P95, prédiction isolée (fonction de service, ms)",
        "metric": "latency_p95_ms_service",
        "op": "<=",
        "value": 20.0,
        "severity": "majeur",
        "rationale": "Budget alloué au scoring dans une autorisation de paiement "
        "(réponse globale visée < 100-200 ms).",
    },
    {
        "id": "CA13",
        "label": "Latence P95, gestionnaire de requête (JSON + validation + score, ms)",
        "metric": "latency_p95_ms_handler",
        "op": "<=",
        "value": 50.0,
        "severity": "majeur",
        "rationale": "Marge pour la couche HTTP et le réseau dans le budget global.",
    },
    {
        "id": "CA14",
        "label": "Débit en lot de 1 000 transactions (tx/s)",
        "metric": "throughput_tx_s",
        "op": ">=",
        "value": 1000.0,
        "severity": "mineur",
        "rationale": "≥ 10 fois le pic estimé (~100 tx/s) pour le traitement par lots.",
    },
    {
        "id": "CA15",
        "label": "Taille du modèle sérialisé (Mo)",
        "metric": "model_size_mb",
        "op": "<=",
        "value": 10.0,
        "severity": "mineur",
        "rationale": "Chargement rapide et empreinte réduite en conteneur.",
    },
    {
        "id": "CA16",
        "label": "Surcoût mémoire pendant l'inférence en lot (Mo)",
        "metric": "rss_delta_mb",
        "op": "<=",
        "value": 500.0,
        "severity": "mineur",
        "rationale": "Compatible avec une limite de conteneur de 1 Go.",
    },
    {
        "id": "CA17",
        "label": "Secrets détectés dans le dépôt",
        "metric": "secrets_found",
        "op": "<=",
        "value": 0,
        "severity": "critique",
        "rationale": "Aucun identifiant, clé ou jeton versionné.",
    },
    {
        "id": "CA18",
        "label": "Valeurs sensibles retrouvées dans les journaux",
        "metric": "sensitive_values_in_logs",
        "op": "<=",
        "value": 0,
        "severity": "critique",
        "rationale": "Les journaux ne contiennent ni valeurs d'entrée ni données de transaction.",
    },
    {
        "id": "CA19",
        "label": "Tests automatisés en échec",
        "metric": "tests_failed",
        "op": "<=",
        "value": 0,
        "severity": "critique",
        "rationale": "Aucune régression connue tolérée.",
    },
    {
        "id": "CA20",
        "label": "Couverture de code (%)",
        "metric": "coverage_pct",
        "op": ">=",
        "value": 85.0,
        "severity": "majeur",
        "rationale": "Seuil usuel d'une suite de tests crédible sur le code critique.",
    },
    {
        "id": "CA21",
        "label": "Erreurs de style/qualité Ruff",
        "metric": "ruff_errors",
        "op": "<=",
        "value": 0,
        "severity": "majeur",
        "rationale": "Code homogène et analysé statiquement.",
    },
    {
        "id": "CA22",
        "label": "Barrière de non-régression : scénarios de référence mal classés",
        "metric": "regression_gate_failures",
        "op": "<=",
        "value": 0,
        "severity": "majeur",
        "rationale": "Le contrôle doit accepter un refactoring neutre et rejeter une dégradation.",
    },
]
