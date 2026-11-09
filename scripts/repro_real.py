"""Empreinte des probabilités de test d'un entraînement complet (jeu réel ou proxy synthétique).

Deux exécutions dans deux processus distincts doivent donner la même empreinte : preuve de
reproductibilité de bout en bout (chargement, dédoublonnage, partition, entraînement).
Usage : python scripts/repro_real.py logreg '{"C": 1.0}' [--synthetic]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import config as cfg
from app.preprocessing import deduplicate, load_data, preprocess_data, split_data
from app.synthetic import make_synthetic_transactions
from app.training import train_model


def digest(name: str, params: dict, synthetic: bool = False) -> str:
    raw = (
        make_synthetic_transactions(60_000, 0.005, seed=1)
        if synthetic
        else load_data(cfg.DATA_PATH)
    )
    df, _ = deduplicate(raw)
    X, y = preprocess_data(df)
    s = split_data(X, y, random_state=cfg.RANDOM_STATE)
    model = train_model(s.X_train, s.y_train, name, params, cfg.RANDOM_STATE)
    proba = np.round(model.predict_proba(s.X_test)[:, 1], 10)
    return hashlib.sha256(proba.tobytes()).hexdigest()


if __name__ == "__main__":
    print(digest(sys.argv[1], json.loads(sys.argv[2]), "--synthetic" in sys.argv))
