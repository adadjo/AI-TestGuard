"""Tests de sécurité opérationnelle (T19) : secrets, données, journaux, .gitignore."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from app.preprocessing import validate_data
from app.security import find_large_files, missing_gitignore_entries, scan_for_secrets

ROOT = Path(__file__).resolve().parents[1]


def test_no_secret_is_committed_in_the_repository():
    assert scan_for_secrets(ROOT) == []


def test_secret_scanner_is_not_vacuous(tmp_path):
    """Le détecteur doit réellement trouver des secrets plantés (sinon le test précédent ne prouve rien)."""
    fake_aws = "AKIA" + "ABCDEFGHIJKLMNOP"
    (tmp_path / "config.py").write_text(
        f'KEY = "{fake_aws}"\n' + "pass" + 'word = "hunter2hunter2"\n', encoding="utf-8"
    )
    (tmp_path / "id_rsa.txt").write_text("-----BEGIN " + "PRIVATE KEY-----\n", encoding="utf-8")
    rules = {finding["rule"] for finding in scan_for_secrets(tmp_path)}
    assert {"aws_access_key", "generic_assignment", "private_key"} <= rules


def test_secret_scanner_ignores_data_and_model_directories(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dump.txt").write_text("AKIA" + "ABCDEFGHIJKLMNOP", encoding="utf-8")
    assert scan_for_secrets(tmp_path) == []


def test_gitignore_protects_data_models_and_credentials():
    assert missing_gitignore_entries(ROOT) == []


def test_gitignore_check_reports_what_is_missing(tmp_path):
    assert "data/*.csv" in missing_gitignore_entries(tmp_path)  # pas de .gitignore du tout
    (tmp_path / ".gitignore").write_text("*.joblib\n.env\n", encoding="utf-8")
    missing = missing_gitignore_entries(tmp_path)
    assert "*.joblib" not in missing and "kaggle.json" in missing


def test_no_large_file_is_versionable():
    assert find_large_files(ROOT) == []


def test_kaggle_csv_is_not_tracked_by_git():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        return
    tracked = subprocess.run(
        ["git", "ls-files", "data"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert not [f for f in tracked if f.endswith(".csv") and not f.startswith("data/sample_")]


def test_validation_warnings_log_counts_not_values(caplog, valid_df):
    valid_df.loc[valid_df.index[:3], "V8"] = np.nan
    valid_df.loc[valid_df.index[5], "Amount"] = 98765.4321
    with caplog.at_level(logging.WARNING, logger="aitestguard"):
        validate_data(valid_df)
    assert caplog.records
    assert "98765" not in caplog.text
