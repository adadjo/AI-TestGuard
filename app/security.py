"""Contrôles de sécurité opérationnelle du dépôt : secrets, fichiers volumineux, .gitignore."""

from __future__ import annotations

import re
from pathlib import Path

SECRET_PATTERNS = {
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "github_token": r"gh[pousr]_[A-Za-z0-9]{36,}",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "slack_token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "kaggle_json_key": r"(?i)\"key\"\s*:\s*\"[0-9a-f]{32}\"",
    "generic_assignment": (
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}
EXCLUDED_DIRS = {
    ".git",
    "data",
    "models",
    "reports",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
    ".hypothesis",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
}
SCANNED_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".toml", ".txt", ".ini", ".cfg", ".sh"}
REQUIRED_GITIGNORE_ENTRIES = ("data/*.csv", "*.joblib", ".env", "kaggle.json", "__pycache__/")


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        yield path


def scan_for_secrets(root: str | Path, max_bytes: int = 2_000_000) -> list[dict]:
    """Recherche des motifs de secrets dans les fichiers texte versionnables (hors données)."""
    root = Path(root)
    compiled = {name: re.compile(pattern) for name, pattern in SECRET_PATTERNS.items()}
    findings: list[dict] = []
    for path in _iter_files(root):
        if path.suffix.lower() not in SCANNED_SUFFIXES and path.name != ".env":
            continue
        if path.stat().st_size > max_bytes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            for rule, regex in compiled.items():
                if regex.search(line):
                    findings.append(
                        {"file": str(path.relative_to(root)), "line": number, "rule": rule}
                    )
    return findings


def find_large_files(root: str | Path, max_bytes: int = 5_000_000) -> list[dict]:
    """Fichiers volumineux hors dossiers de données/modèles (à ne pas versionner sans raison)."""
    root = Path(root)
    return [
        {"file": str(p.relative_to(root)), "size_bytes": p.stat().st_size}
        for p in _iter_files(root)
        if p.stat().st_size > max_bytes
    ]


def missing_gitignore_entries(root: str | Path) -> list[str]:
    """Entrées obligatoires absentes du .gitignore (données, modèles, secrets)."""
    gitignore = Path(root) / ".gitignore"
    if not gitignore.is_file():
        return list(REQUIRED_GITIGNORE_ENTRIES)
    lines = {ln.strip() for ln in gitignore.read_text(encoding="utf-8").splitlines()}
    return [entry for entry in REQUIRED_GITIGNORE_ENTRIES if entry not in lines]
