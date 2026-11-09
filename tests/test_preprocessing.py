"""Tests unitaires : contrat de données, prétraitement, absence de fuite (T01, T02, T03, T08)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler

from app import config as cfg
from app.preprocessing import DataValidationError, preprocess_data, validate_data
from app.robustness import inject_missing
from app.training import build_preprocessor


def test_standard_scaler():
    """Exemple de l'énoncé : comportement d'un scaler ajusté sur trois observations."""
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


# --------------------------------------------------------------- T01 / T02 : schéma, types
def test_validate_accepts_a_valid_frame(valid_df):
    report = validate_data(valid_df)
    assert report.n_rows == len(valid_df)
    assert report.n_missing_cells == 0
    assert not report.has_warnings


@pytest.mark.parametrize("column", ["Time", "V1", "V14", "V28", "Amount", "Class"])
def test_validate_rejects_missing_required_column(valid_df, column):
    with pytest.raises(DataValidationError, match="absentes"):
        validate_data(valid_df.drop(columns=[column]))


def test_validate_does_not_require_target_for_inference(valid_df):
    report = validate_data(valid_df.drop(columns=["Class"]), require_target=False)
    assert report.n_rows == len(valid_df)


def test_validate_rejects_duplicated_column(valid_df):
    duplicated = pd.concat([valid_df, valid_df[["Amount"]]], axis=1)
    with pytest.raises(DataValidationError, match="dupliquées"):
        validate_data(duplicated)


def test_validate_rejects_unexpected_column(valid_df):
    valid_df["Unexpected"] = 1.0
    with pytest.raises(DataValidationError, match="inattendue"):
        validate_data(valid_df)


def test_validate_rejects_empty_dataframe(valid_df):
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(valid_df.iloc[0:0])


@pytest.mark.parametrize("bad", [None, [1, 2, 3], {"Amount": [1.0]}, "texte", np.zeros((3, 31))])
def test_validate_rejects_non_dataframe_inputs(bad):
    with pytest.raises(DataValidationError, match="DataFrame"):
        validate_data(bad)


def test_validate_rejects_string_typed_numeric_column(valid_df):
    valid_df["Amount"] = valid_df["Amount"].astype(str)
    with pytest.raises(DataValidationError, match="numériques"):
        validate_data(valid_df)


def test_validate_rejects_object_dtype_even_if_values_look_numeric(valid_df):
    """Pas de coercition silencieuse : un type incorrect est rejeté, pas « réparé »."""
    valid_df["V1"] = valid_df["V1"].astype(object)
    with pytest.raises(DataValidationError, match="numériques"):
        validate_data(valid_df)


def test_validate_rejects_boolean_feature(valid_df):
    valid_df["V2"] = True
    with pytest.raises(DataValidationError, match="numériques"):
        validate_data(valid_df)


# ------------------------------------------------------- T08 : manquantes, infinies, cible
def test_validate_accepts_few_missing_values_with_warning(valid_df):
    valid_df.loc[valid_df.index[:5], "V3"] = np.nan  # 5/300 = 1,7 % <= 5 %
    report = validate_data(valid_df)
    assert report.n_missing_cells == 5
    assert report.missing_by_column == {"V3": 5}
    assert report.has_warnings


def test_validate_rejects_too_many_missing_values(valid_df):
    valid_df.loc[valid_df.index[:30], "V3"] = np.nan  # 10 % > 5 %
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_data(valid_df)


def test_validate_rejects_any_missing_value_when_channel_forbids_them(valid_df):
    valid_df.loc[valid_df.index[0], "V1"] = np.nan
    with pytest.raises(DataValidationError, match="non autorisées"):
        validate_data(valid_df, allow_missing=False)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_validate_rejects_infinite_values(valid_df, bad_value):
    valid_df.loc[valid_df.index[3], "V5"] = bad_value
    with pytest.raises(DataValidationError, match="infinies"):
        validate_data(valid_df)


@pytest.mark.parametrize("bad_target", [2, -1, 0.5, np.nan])
def test_validate_rejects_target_values_other_than_0_and_1(valid_df, bad_target):
    valid_df["Class"] = valid_df["Class"].astype("float64")
    valid_df.loc[valid_df.index[0], "Class"] = bad_target
    with pytest.raises(DataValidationError, match="hors de"):
        validate_data(valid_df)


def test_validate_accepts_float_and_boolean_targets(valid_df):
    as_float = valid_df.assign(Class=valid_df["Class"].astype("float64"))
    as_bool = valid_df.assign(Class=valid_df["Class"].astype(bool))
    assert validate_data(as_float).n_rows == len(valid_df)
    assert validate_data(as_bool).n_rows == len(valid_df)


def test_validate_rejects_non_numeric_target(valid_df):
    valid_df["Class"] = valid_df["Class"].map({0: "normale", 1: "fraude"})
    with pytest.raises(DataValidationError, match="numérique"):
        validate_data(valid_df)


# ------------------------------------------------------------------- valeurs limites
def test_validate_single_observation_is_valid_and_keeps_dimensions(valid_df):
    single = valid_df.iloc[[0]]
    assert validate_data(single).n_rows == 1
    X, y = preprocess_data(single)
    assert X.shape == (1, len(cfg.FEATURE_COLUMNS))
    assert y.shape == (1,)


def test_validate_amount_boundaries(valid_df):
    idx = valid_df.index[0]
    valid_df.loc[idx, "Amount"] = 0.0  # borne basse incluse
    assert validate_data(valid_df).n_rows == len(valid_df)
    valid_df.loc[idx, "Amount"] = cfg.AMOUNT_HARD_LIMIT  # borne haute incluse
    assert validate_data(valid_df).n_rows == len(valid_df)
    valid_df.loc[idx, "Amount"] = -0.01
    with pytest.raises(DataValidationError, match="négatifs"):
        validate_data(valid_df)
    valid_df.loc[idx, "Amount"] = cfg.AMOUNT_HARD_LIMIT + 1
    with pytest.raises(DataValidationError, match="hors contrat"):
        validate_data(valid_df)


def test_validate_flags_amount_above_alert_threshold_without_rejecting(valid_df):
    valid_df.loc[valid_df.index[0], "Amount"] = cfg.AMOUNT_ALERT_THRESHOLD + 1
    report = validate_data(valid_df)
    assert report.has_warnings
    assert "seuil d'alerte" in report.warnings[0]


def test_validate_rejects_negative_time_and_implausible_pca_component(valid_df):
    negative_time = valid_df.copy()
    negative_time.loc[negative_time.index[0], "Time"] = -1.0
    with pytest.raises(DataValidationError, match="Time"):
        validate_data(negative_time)
    implausible = valid_df.copy()
    implausible.loc[implausible.index[0], "V9"] = cfg.V_ABS_HARD_LIMIT * 2
    with pytest.raises(DataValidationError, match="Vi"):
        validate_data(implausible)


def test_validation_messages_never_echo_input_values(valid_df):
    valid_df.loc[valid_df.index[0], "Amount"] = -987654.321
    with pytest.raises(DataValidationError) as excinfo:
        validate_data(valid_df)
    assert "987654" not in str(excinfo.value)


# ------------------------------------------------------------------------ preprocess_data
def test_preprocess_data_shapes_order_dtype_and_index(valid_df):
    X, y = preprocess_data(valid_df)
    assert list(X.columns) == cfg.FEATURE_COLUMNS
    assert X.shape == (len(valid_df), len(cfg.FEATURE_COLUMNS))
    assert (X.dtypes == "float64").all()
    assert y.shape == (len(valid_df),)
    assert X.index.equals(valid_df.index) and y.index.equals(valid_df.index)


def test_preprocess_data_does_not_modify_its_input(valid_df):
    snapshot = valid_df.copy(deep=True)
    preprocess_data(valid_df)
    pd.testing.assert_frame_equal(valid_df, snapshot)


def test_preprocess_data_without_target_returns_none(valid_df):
    X, y = preprocess_data(valid_df.drop(columns=["Class"]), require_target=False)
    assert y is None
    assert X.shape[1] == len(cfg.FEATURE_COLUMNS)


# --------------------------------------------------- T03 : sortie du prétraitement finie
def test_preprocessor_output_is_finite_with_expected_dimensions(splits):
    pre = build_preprocessor().fit(splits.X_train)
    out = pre.transform(splits.X_val)
    assert np.isfinite(out).all()
    assert out.shape == (len(splits.X_val), len(cfg.MODEL_FEATURES))
    assert list(pre.get_feature_names_out()) == ["Amount", *cfg.V_COLUMNS]


def test_preprocessor_imputes_missing_values_so_output_stays_finite(splits):
    pre = build_preprocessor().fit(splits.X_train)
    holes = inject_missing(splits.X_val, 0.02, seed=3)
    assert holes.isna().to_numpy().any()
    assert np.isfinite(pre.transform(holes)).all()


def test_preprocessor_keeps_dimensions_when_a_training_column_is_entirely_missing(splits):
    broken = splits.X_train.copy()
    broken["V7"] = np.nan
    pre = build_preprocessor().fit(broken)
    out = pre.transform(splits.X_val)
    assert out.shape[1] == len(cfg.MODEL_FEATURES)
    assert np.isfinite(out).all()


def test_time_is_excluded_from_model_features_unless_requested(splits):
    assert "Time" not in cfg.MODEL_FEATURES
    default = build_preprocessor().fit(splits.X_train)
    with_time = build_preprocessor(include_time=True).fit(splits.X_train)
    assert "Time" not in default.get_feature_names_out()
    assert "Time" in with_time.get_feature_names_out()


def test_amount_transform_clips_negative_values_before_the_logarithm(splits):
    """Défense en profondeur : même si la validation était contournée, log1p(Amount < 0) ne doit
    produire ni NaN ni avertissement numérique (régression révélée par le test de mutation, M19)."""
    pre = build_preprocessor().fit(splits.X_train)
    negative = splits.X_val.head(20).assign(Amount=-5.0)
    zero = splits.X_val.head(20).assign(Amount=0.0)
    out = pre.transform(negative)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out, pre.transform(zero))


def test_missing_fraction_limit_is_inclusive_at_exactly_five_percent(small_df):
    """Cas limite (mutant M21) : 5 % exactement est accepté, 6 % est rejeté."""
    df = small_df.head(100).copy()
    df.loc[df.index[:5], "V3"] = np.nan
    assert validate_data(df).missing_by_column == {"V3": 5}
    df.loc[df.index[:6], "V3"] = np.nan
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_data(df)


# ------------------------------------------------------------- T09 : pas de fuite (unitaire)
def test_scaler_is_fitted_on_training_data_only(splits):
    pre = build_preprocessor().fit(splits.X_train)
    scaler = pre.named_transformers_["pca"].named_steps["scale"]
    train_mean = splits.X_train[cfg.V_COLUMNS].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_mean, atol=1e-9)

    learned = scaler.mean_.copy()
    pre.transform(splits.X_test)  # transformer le test ne doit rien réapprendre
    np.testing.assert_array_equal(scaler.mean_, learned)

    everything = pd.concat([splits.X_train, splits.X_val, splits.X_test])
    assert not np.allclose(scaler.mean_, everything[cfg.V_COLUMNS].mean().to_numpy(), atol=1e-12)


# ------------------------------------------------------------------- propriété (Hypothesis)
@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(1, 30),
    scale=st.floats(0.0, 40.0),
    amount=st.floats(0.0, 1e5),
)
def test_property_valid_inputs_always_give_finite_preprocessed_features(seed, n, scale, amount):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0.0, scale, (n, 28)), columns=cfg.V_COLUMNS)
    df.insert(0, "Time", rng.uniform(0, 172_792, n))
    df["Amount"] = amount
    df["Class"] = rng.integers(0, 2, n)
    X, y = preprocess_data(df)
    assert np.isfinite(X.to_numpy()).all()
    assert X.shape == (n, len(cfg.FEATURE_COLUMNS))
    assert set(y.unique()) <= {0, 1}
