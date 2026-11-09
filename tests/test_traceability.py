"""Méta-tests : la matrice de traçabilité (docs/traceability.json) reste cohérente avec les tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRACE = json.loads((ROOT / "docs" / "traceability.json").read_text(encoding="utf-8"))
REQUIRED_FIELDS = {"id", "requirement", "type", "expected", "risk", "priority", "tests"}


def _defined_tests() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found[f"tests/{path.name}"] = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
    return found


def test_the_seven_mandatory_requirements_and_at_least_five_more_are_present():
    ids = [r["id"] for r in TRACE]
    assert ids[:7] == [f"T0{i}" for i in range(1, 8)]
    assert len(ids) - 7 >= 5
    assert len(ids) == len(set(ids))


def test_mandatory_requirements_keep_the_wording_of_the_assignment():
    expected = {
        "T01": ("Les colonnes nécessaires sont présentes", "Données"),
        "T02": ("Les données respectent les types attendus", "Unitaire"),
        "T03": ("Le prétraitement ne produit pas de valeurs non finies", "Unitaire"),
        "T04": ("Le modèle produit une probabilité valide", "Fonctionnel ML"),
        "T05": ("Les performances sont évaluées sur des données indépendantes", "ML"),
        "T06": ("Les résultats sont reproductibles", "Non-régression"),
        "T07": ("Les tests s'exécutent sur GitHub", "CI"),
    }
    by_id = {r["id"]: r for r in TRACE}
    for key, (text, kind) in expected.items():
        assert by_id[key]["requirement"] == text and by_id[key]["type"] == kind


@pytest.mark.parametrize("requirement", TRACE, ids=lambda r: r["id"])
def test_each_requirement_is_complete_and_linked_to_existing_tests(requirement):
    assert set(requirement) >= REQUIRED_FIELDS
    assert requirement["risk"] and requirement["expected"]
    defined = _defined_tests()
    assert requirement["tests"], "une exigence sans test n'est pas vérifiée"
    for reference in requirement["tests"]:
        file, _, name = reference.partition("::")
        assert file in defined, f"fichier de test inconnu : {file}"
        assert name in defined[file], f"test introuvable : {reference}"
