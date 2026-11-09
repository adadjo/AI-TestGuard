"""Tests sur le jeu Kaggle réel (ignorés si data/creditcard.csv est absent)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app import config as cfg
from app.preprocessing import deduplicate, load_data, preprocess_data, validate_data
from app.training import run_training_pipeline

pytestmark = pytest.mark.full_data

KAGGLE_MD5 = "e90efcb83d69faf99fcab8b0255024de"  # empreinte publiée pour creditcard.csv (Zenodo)


@pytest.fixture(scope="module")
def real_df():
    return load_data(cfg.DATA_PATH)


def test_real_file_is_the_original_kaggle_file():
    md5 = hashlib.md5(cfg.DATA_PATH.read_bytes()).hexdigest()
    assert md5 == KAGGLE_MD5


def test_real_dataset_matches_the_published_description(real_df):
    assert real_df.shape == (284_807, 31)
    assert int(real_df["Class"].sum()) == 492
    assert real_df["Class"].mean() == pytest.approx(0.00173, abs=1e-5)
    assert list(real_df.columns) == cfg.REQUIRED_COLUMNS


def test_real_dataset_satisfies_the_data_contract(real_df):
    report = validate_data(real_df)
    assert report.n_missing_cells == 0 and not report.has_warnings
    assert real_df["Amount"].min() >= 0
    assert real_df["Amount"].max() < cfg.AMOUNT_ALERT_THRESHOLD
    assert np.isfinite(real_df.to_numpy()).all()


def test_real_dataset_contains_known_exact_duplicates(real_df):
    clean, removed = deduplicate(real_df)
    assert removed == 1081
    assert len(clean) == 283_726 and int(clean["Class"].sum()) == 473


@pytest.mark.slow
def test_end_to_end_run_on_the_real_dataset_is_sane(real_df):
    clean, _ = deduplicate(real_df)
    X, _ = preprocess_data(clean)
    run = run_training_pipeline(clean, "logreg")
    assert run.test_metrics["roc_auc"] > 0.9
    assert run.test_metrics["n"] == pytest.approx(0.15 * len(X), abs=2)
