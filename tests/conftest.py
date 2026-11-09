"""Fixtures partagées : données synthétiques déterministes, partitions, modèles entraînés.

Les tests rapides n'utilisent JAMAIS le jeu Kaggle (non versionné) : ils s'appuient sur un
générateur synthétique déterministe de même schéma (voir ``app/synthetic.py``).
"""

from __future__ import annotations

import pytest

from app import config as cfg
from app.preprocessing import preprocess_data, split_data
from app.synthetic import make_synthetic_transactions
from app.training import select_threshold, train_model

FIXTURE_MAX_FPR = 0.01  # jeux de test très petits : une cible de 0,08 % serait dégénérée


def pytest_collection_modifyitems(config, items):
    if cfg.DATA_PATH.exists():
        return
    skip = pytest.mark.skip(reason="data/creditcard.csv absent (jeu Kaggle non versionné)")
    for item in items:
        if "full_data" in item.keywords:
            item.add_marker(skip)


class StubModel:
    """Faux modèle renvoyant des probabilités imposées (tests du contrat de sortie)."""

    classes_ = [0, 1]

    def __init__(self, proba=None, raw=None):
        self._proba = proba
        self._raw = raw

    def predict_proba(self, X):
        if self._raw is not None:
            return self._raw
        import numpy as np

        p = np.asarray(self._proba, dtype="float64")
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="session")
def small_df():
    return make_synthetic_transactions(6000, 0.02, seed=7)


@pytest.fixture()
def valid_df(small_df):
    """300 premières transactions : copie fraîche pour chaque test (mutable sans risque)."""
    return small_df.head(300).copy()


@pytest.fixture(scope="session")
def xy(small_df):
    return preprocess_data(small_df)


@pytest.fixture(scope="session")
def splits(xy):
    X, y = xy
    return split_data(X, y, random_state=cfg.RANDOM_STATE)


@pytest.fixture(scope="session")
def trained_lr(splits):
    return train_model(splits.X_train, splits.y_train, "logreg")


@pytest.fixture(scope="session")
def trained_tree(splits):
    return train_model(splits.X_train, splits.y_train, "tree")


@pytest.fixture(scope="session")
def lr_threshold(trained_lr, splits):
    proba = trained_lr.predict_proba(splits.X_val)[:, 1]
    return select_threshold(splits.y_val, proba, max_fpr=FIXTURE_MAX_FPR).threshold


@pytest.fixture()
def valid_payload(small_df):
    row = small_df.iloc[10][cfg.FEATURE_COLUMNS]
    return {key: float(value) for key, value in row.items()}
