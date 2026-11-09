"""Empreinte des prédictions d'un entraînement complet : utilisée pour prouver la
reproductibilité ENTRE PROCESSUS (deux exécutions indépendantes doivent donner la même)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.preprocessing import preprocess_data, split_data
from app.synthetic import make_synthetic_transactions
from app.training import train_model


def prediction_digest(model_name: str = "logreg", seed: int = 42) -> str:
    df = make_synthetic_transactions(4000, 0.02, seed=11)
    X, y = preprocess_data(df)
    s = split_data(X, y, random_state=seed)
    model = train_model(s.X_train, s.y_train, model_name, random_state=seed)
    proba = np.round(model.predict_proba(s.X_test)[:, 1], 10)
    return hashlib.sha256(proba.tobytes()).hexdigest()


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "logreg"
    print(prediction_digest(name))
