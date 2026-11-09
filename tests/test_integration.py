"""Tests d'intégration : pipeline de bout en bout, fuites de données, sauvegarde (T05, T09, T12, T13)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app import config as cfg
from app.evaluation import evaluate_pipeline
from app.preprocessing import (
    DataLoadError,
    DataValidationError,
    dataset_fingerprint,
    deduplicate,
    load_data,
    preprocess_data,
    split_data,
    temporal_split,
)
from app.training import (
    ModelIntegrityError,
    ModelLoadError,
    TrainingError,
    build_pipeline,
    load_model,
    run_training_pipeline,
    save_model,
    select_threshold,
    train_model,
    tune_model,
)
from app.utils import dump_json, sha256_file
from tests.conftest import FIXTURE_MAX_FPR

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ partitions (T05, T09)
def test_split_is_disjoint_complete_and_stratified(xy, splits):
    X, y = xy
    train, val, test = (set(s.index) for s in (splits.X_train, splits.X_val, splits.X_test))
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(X.index)
    overall = y.mean()
    for part in (splits.y_train, splits.y_val, splits.y_test):
        assert abs(part.mean() - overall) < 0.005
        assert part.sum() >= 10
    sizes = [len(splits.X_train), len(splits.X_val), len(splits.X_test)]
    assert sizes == pytest.approx([0.70 * len(X), 0.15 * len(X), 0.15 * len(X)], abs=2)


def test_split_is_reproducible_and_depends_on_the_seed(xy):
    X, y = xy
    a, b = split_data(X, y, random_state=1), split_data(X, y, random_state=1)
    c = split_data(X, y, random_state=2)
    assert a.X_test.index.equals(b.X_test.index)
    assert not a.X_test.index.equals(c.X_test.index)


@pytest.mark.parametrize(
    "kwargs",
    [{"val_size": 0.0}, {"test_size": 1.0}, {"val_size": 0.6, "test_size": 0.5}],
    ids=["val-0", "test-1", "sum>1"],
)
def test_split_rejects_invalid_proportions(xy, kwargs):
    with pytest.raises(ValueError, match="]0, 1"):
        split_data(*xy, **kwargs)


def test_split_rejects_misaligned_inputs_and_tiny_minority_class(xy):
    X, y = xy
    with pytest.raises(ValueError, match="alignés"):
        split_data(X, y.iloc[:-1])
    few = y.copy()
    few[:] = 0
    few.iloc[:3] = 1
    with pytest.raises(ValueError, match="Au moins 6"):
        split_data(X, few)


def test_temporal_split_is_chronological(xy):
    X, y = xy
    s = temporal_split(X, y)
    assert s.X_train["Time"].max() <= s.X_val["Time"].min()
    assert s.X_val["Time"].max() <= s.X_test["Time"].min()
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == len(X)
    with pytest.raises(ValueError):
        temporal_split(X, y, train_frac=0.9, val_frac=0.2)


def test_deduplicate_removes_exact_duplicates_so_none_cross_partitions(small_df):
    duplicated = pd.concat([small_df, small_df.head(100)], ignore_index=True)
    clean, removed = deduplicate(duplicated)
    assert removed == 100 and len(clean) == len(small_df)
    X, y = preprocess_data(clean)
    s = split_data(X, y)
    merged = s.X_test.merge(s.X_train.drop_duplicates(), how="inner")
    assert merged.empty


# ----------------------------------------------------- bout en bout et provenance (T05, T09)
def test_end_to_end_run_uses_each_partition_only_for_its_role(small_df):
    run = run_training_pipeline(small_df, "logreg", max_fpr=FIXTURE_MAX_FPR)
    s = run.splits
    fit, thr, test = (
        set(run.provenance[k]) for k in ("fit_index", "threshold_index", "test_index")
    )
    assert fit == set(s.X_train.index)
    assert thr == set(s.X_val.index)
    assert not (fit & test) and not (thr & test)

    proba_val = run.pipeline.predict_proba(s.X_val)[:, 1]
    assert select_threshold(s.y_val, proba_val, FIXTURE_MAX_FPR).threshold == run.threshold
    assert run.val_metrics["n"] == len(s.X_val) and run.test_metrics["n"] == len(s.X_test)
    assert run.train_metrics["n"] == len(s.X_train)


def test_pipeline_fit_never_sees_validation_or_test_rows(small_df, monkeypatch):
    seen: list[set] = []
    original = Pipeline.fit

    def spy(self, X, y=None, **params):
        seen.append(set(X.index))
        return original(self, X, y, **params)

    monkeypatch.setattr(Pipeline, "fit", spy)
    run = run_training_pipeline(small_df, "logreg", max_fpr=FIXTURE_MAX_FPR)
    train_index = set(run.splits.X_train.index)
    assert seen, "le pipeline doit avoir été ajusté"
    assert all(index <= train_index for index in seen)


def test_tuning_uses_validation_rows_only_and_picks_a_grid_value(small_df):
    grid = {"C": [0.1, 1.0], "class_weight": [None, "balanced"]}
    run = run_training_pipeline(small_df, "logreg", tune=True, grid=grid, max_fpr=FIXTURE_MAX_FPR)
    assert run.provenance["tuning_index"].equals(run.splits.X_val.index)
    assert len(run.tuning) == 4
    assert run.params["C"] in grid["C"] and run.params["class_weight"] in grid["class_weight"]
    best = max(r["val_average_precision"] for r in run.tuning)
    chosen = next(
        r
        for r in run.tuning
        if r["C"] == run.params["C"] and r["class_weight"] == run.params["class_weight"]
    )
    assert chosen["val_average_precision"] == best


def test_tune_model_is_deterministic(splits):
    grid = {"max_depth": [3, 5], "min_samples_leaf": [10]}
    a = tune_model("tree", splits.X_train, splits.y_train, splits.X_val, splits.y_val, grid)
    b = tune_model("tree", splits.X_train, splits.y_train, splits.X_val, splits.y_val, grid)
    assert a[0] == b[0] and a[2] == b[2]


@pytest.mark.parametrize("name", ["logreg", "tree"])
def test_both_models_train_and_evaluate(name, splits):
    model = train_model(splits.X_train, splits.y_train, name)
    metrics, proba = evaluate_pipeline(model, splits.X_val, splits.y_val, 0.5)
    assert metrics["roc_auc"] > 0.8
    assert proba.shape == (len(splits.X_val),)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        build_pipeline("deep-magic")


def test_train_model_rejects_unusable_training_data(splits):
    with pytest.raises(TrainingError, match="deux classes"):
        train_model(splits.X_train, splits.y_train * 0)
    with pytest.raises(TrainingError, match="même longueur"):
        train_model(splits.X_train, splits.y_train.iloc[:-1])
    with pytest.raises(DataValidationError, match="absentes"):
        train_model(splits.X_train.drop(columns=["V5"]), splits.y_train)


# ----------------------------------------------------- dimensions de bout en bout (T13)
def test_dimensions_are_consistent_from_input_to_prediction(trained_lr, splits):
    n = len(splits.X_test)
    transformed = trained_lr.named_steps["preprocess"].transform(splits.X_test)
    assert transformed.shape == (n, len(cfg.MODEL_FEATURES))
    assert trained_lr.named_steps["model"].coef_.shape == (1, len(cfg.MODEL_FEATURES))
    assert trained_lr.predict_proba(splits.X_test).shape == (n, 2)


# -------------------------------------------------- sauvegarde / rechargement (T12)
def test_saved_and_reloaded_pipeline_predicts_identically(
    tmp_path, trained_lr, splits, lr_threshold
):
    path = tmp_path / "model.joblib"
    meta = save_model(trained_lr, lr_threshold, path, {"note": "test"})
    reloaded, threshold, loaded_meta = load_model(path)

    before = trained_lr.predict_proba(splits.X_test)[:, 1]
    after = reloaded.predict_proba(splits.X_test)[:, 1]
    ca08 = next(c for c in cfg.ACCEPTANCE_CRITERIA if c["id"] == "CA08")
    assert np.max(np.abs(before - after)) <= ca08["value"]
    assert np.array_equal(before, after)
    assert threshold == lr_threshold
    assert meta["sha256"] == loaded_meta["sha256"] == sha256_file(path)
    assert {"sklearn_version", "code_version", "features", "created_utc"} <= set(loaded_meta)


def test_reload_detects_a_tampered_artifact_before_deserialising(
    tmp_path, trained_lr, lr_threshold
):
    path = tmp_path / "model.joblib"
    save_model(trained_lr, lr_threshold, path)
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 0xFF
    path.write_bytes(bytes(content))
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        load_model(path)


def test_reload_refuses_an_artifact_without_hash_file(tmp_path, trained_lr, lr_threshold):
    path = tmp_path / "model.joblib"
    save_model(trained_lr, lr_threshold, path)
    path.with_suffix(".json").unlink()
    with pytest.raises(ModelIntegrityError, match="absent"):
        load_model(path)


def test_reload_missing_or_malformed_artifact_is_reported(tmp_path):
    with pytest.raises(ModelLoadError, match="introuvable"):
        load_model(tmp_path / "absent.joblib")
    path = tmp_path / "odd.joblib"
    joblib.dump({"unexpected": 1}, path)
    dump_json({"sha256": sha256_file(path)}, path.with_suffix(".json"))
    with pytest.raises(ModelLoadError, match="inattendu"):
        load_model(path)


# --------------------------------------------------------------------- chargement de fichiers
def test_load_data_round_trip(tmp_path, small_df):
    path = tmp_path / "ok.csv"
    small_df.head(120).to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded.shape == (120, len(cfg.REQUIRED_COLUMNS))


def test_load_data_reports_every_file_problem_explicitly(tmp_path, small_df):
    with pytest.raises(DataLoadError, match="introuvable"):
        load_data(tmp_path / "absent.csv")
    with pytest.raises(DataLoadError, match="pas un fichier"):
        load_data(tmp_path)
    wrong_ext = tmp_path / "data.txt"
    wrong_ext.write_text("a,b\n1,2\n")
    with pytest.raises(DataLoadError, match="Extension"):
        load_data(wrong_ext)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataLoadError, match="vide"):
        load_data(empty)
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("a,b\n1,2\n3,4,5,6,7\n")
    with pytest.raises(DataLoadError, match="illisible"):
        load_data(malformed)
    big = tmp_path / "big.csv"
    small_df.head(50).to_csv(big, index=False)
    with pytest.raises(DataLoadError, match="volumineux"):
        load_data(big, max_bytes=100)
    wrong_schema = tmp_path / "schema.csv"
    wrong_schema.write_text("a,b\n1,2\n")
    with pytest.raises(DataValidationError, match="absentes"):
        load_data(wrong_schema)


def test_dataset_fingerprint_is_stable_and_sensitive(small_df):
    a, b = dataset_fingerprint(small_df), dataset_fingerprint(small_df.copy())
    assert a == b and a["n_rows"] == len(small_df) and a["n_fraud"] == int(small_df["Class"].sum())
    changed = small_df.copy()
    changed.loc[changed.index[0], "Amount"] += 1.0
    assert dataset_fingerprint(changed)["content_sha256"] != a["content_sha256"]


# ------------------------------------------------------ reproductibilité entre processus
@pytest.mark.slow
@pytest.mark.parametrize("name", ["logreg", "tree"])
def test_training_is_reproducible_across_independent_processes(name):
    digests = [
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "repro_check.py"), name],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert digests[0] == digests[1] and len(digests[0]) == 64
