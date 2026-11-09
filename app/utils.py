"""Utilitaires transverses : empreintes, versions d'environnement, sérialisation JSON."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata as md
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "matplotlib",
    "pytest",
    "hypothesis",
    "psutil",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Empreinte SHA-256 d'un fichier (lecture par blocs)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict[str, Any]:
    """Versions de Python et des bibliothèques : à enregistrer avec chaque référence."""
    versions: dict[str, str | None] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": versions,
    }


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Hash du commit courant, ou None si le code n'est pas versionné."""
    root = Path(cwd) if cwd else Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value if out.returncode == 0 and value else None


def safe_names(names: Any, limit: int = 5) -> str:
    """Représentation prudente de noms fournis par l'utilisateur (pas d'écho brut)."""
    items = [repr(str(n))[:60] for n in list(names)[:limit]]
    suffix = ", ..." if len(list(names)) > limit else ""
    return "[" + ", ".join(items) + suffix + "]"


def to_jsonable(obj: Any) -> Any:
    """Convertit récursivement types NumPy/Path/dataclasses en types JSON (NaN -> null)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def dump_json(obj: Any, path: str | Path) -> Path:
    """Écrit un JSON lisible (UTF-8, indenté) ; crée le dossier parent si besoin."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(obj), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target
