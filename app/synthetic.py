"""Transactions SYNTHÉTIQUES au schéma du jeu Kaggle (tests, CI, démonstrations).

Ces données ne prouvent AUCUNE performance : elles servent à tester la plomberie (schéma,
dimensions, sérialisation, robustesse) rapidement et de façon déterministe, sans versionner
le jeu de données réel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config as cfg

_V_STD = np.array(
    [
        1.96,
        1.65,
        1.52,
        1.42,
        1.38,
        1.33,
        1.24,
        1.19,
        1.10,
        1.09,
        1.02,
        1.00,
        1.00,
        0.96,
        0.92,
        0.88,
        0.85,
        0.84,
        0.81,
        0.77,
        0.73,
        0.73,
        0.62,
        0.61,
        0.52,
        0.48,
        0.40,
        0.33,
    ]
)
# décalages (en écarts-types) appliqués à la classe frauduleuse sur quelques composantes
_FRAUD_SHIFT = {
    1: -1.5,
    2: 1.3,
    3: -2.5,
    4: 2.0,
    7: -1.5,
    9: -1.0,
    10: -2.5,
    11: 2.0,
    12: -3.0,
    14: -3.5,
    16: -2.0,
    17: -2.5,
    18: -1.0,
}


def make_synthetic_transactions(
    n_rows: int = 6000, fraud_rate: float = 0.02, seed: int = 0
) -> pd.DataFrame:
    """Jeu déterministe (colonnes Time, V1-V28, Amount, Class), triées par Time."""
    if n_rows < 50:
        raise ValueError("n_rows doit être >= 50.")
    rng = np.random.default_rng(seed)
    n_fraud = max(10, round(n_rows * fraud_rate))
    y = np.zeros(n_rows, dtype="int64")
    y[rng.choice(n_rows, n_fraud, replace=False)] = 1

    V = rng.standard_normal((n_rows, 28)) * _V_STD
    shift = np.zeros(28)
    for idx, value in _FRAUD_SHIFT.items():
        shift[idx - 1] = value
    fraud = y == 1
    camouflaged = fraud & (rng.random(n_rows) < 0.3)  # fraudes proches des transactions normales
    strength = np.where(camouflaged, 0.25, 1.0)[:, None]
    V[fraud] = V[fraud] * 1.4 + (strength * shift * _V_STD)[fraud]

    amount = np.where(
        fraud,
        rng.lognormal(2.4, 1.8, n_rows),
        rng.lognormal(3.1, 1.5, n_rows),
    )
    amount = np.clip(np.round(amount, 2), 0.0, 25_000.0)
    time_ = np.sort(rng.uniform(0, 172_792, n_rows)).round(0)

    df = pd.DataFrame(V, columns=cfg.V_COLUMNS)
    df.insert(0, "Time", time_)
    df["Amount"] = amount
    df[cfg.TARGET_COLUMN] = y
    return df
