"""Consolidation de la qualification : lint, tests, couverture, critères d'acceptation, verdict.

Exécute Ruff et Pytest (preuves archivées dans reports/ci_evidence/), rassemble les mesures de tous
les scripts d'expérimentation et applique la règle de décision DÉFINIE A PRIORI (app/config.py).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg
from app.evaluation import check_criteria, qualification_verdict
from app.utils import dump_json

EVIDENCE = cfg.REPORTS_DIR / "ci_evidence"


def run(cmd: list[str], out_file: Path) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, capture_output=True, text=True)
    out_file.write_text(
        f"$ {' '.join(cmd)}\n{proc.stdout}{proc.stderr}\n[code de sortie : {proc.returncode}]\n",
        encoding="utf-8",
    )
    return proc


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    cases = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    by_file = defaultdict(lambda: {"tests": 0, "passed": 0, "failed": 0, "skipped": 0})
    for case in suite.iter("testcase"):
        file = case.get("classname", "").replace(".", "/") + ".py"
        base = case.get("name", "").split("[")[0]
        status = (
            "failed"
            if case.find("failure") is not None or case.find("error") is not None
            else ("skipped" if case.find("skipped") is not None else "passed")
        )
        cases[f"{file}::{base}"][status] += 1
        by_file[file]["tests"] += 1
        by_file[file][status] += 1
    return {
        "tests": int(suite.get("tests")),
        "failures": int(suite.get("failures")),
        "errors": int(suite.get("errors")),
        "skipped": int(suite.get("skipped")),
        "time": float(suite.get("time")),
        "by_file": dict(by_file),
        "cases": {k: dict(v) for k, v in cases.items()},
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ruff = run(["ruff", "check", "."], EVIDENCE / "ruff_check.txt")
    ruff_fmt = run(["ruff", "format", "--check", "."], EVIDENCE / "ruff_format.txt")
    n_ruff = (
        0
        if ruff.returncode == 0
        else int(re.search(r"Found (\d+) error", ruff.stdout + ruff.stderr).group(1))
    )

    fast = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "not slow and not full_data",
            "--cov=app",
            "--cov-report=term-missing",
            f"--cov-report=json:{cfg.REPORTS_DIR / 'coverage.json'}",
            f"--junitxml={cfg.REPORTS_DIR / 'junit.xml'}",
            "-p",
            "no:cacheprovider",
            "--cov-fail-under=85",
        ],
        EVIDENCE / "pytest_success.txt",
    )
    slow = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "slow or full_data",
            f"--junitxml={cfg.REPORTS_DIR / 'junit_slow.xml'}",
            "-p",
            "no:cacheprovider",
        ],
        EVIDENCE / "pytest_slow_fulldata.txt",
    )
    j_fast, j_slow = (
        parse_junit(cfg.REPORTS_DIR / "junit.xml"),
        parse_junit(cfg.REPORTS_DIR / "junit_slow.xml"),
    )
    cov = json.loads((cfg.REPORTS_DIR / "coverage.json").read_text(encoding="utf-8"))
    coverage = {
        "percent": cov["totals"]["percent_covered"],
        "statements": cov["totals"]["num_statements"],
        "missing": cov["totals"]["missing_lines"],
        "files": {
            f: {
                "percent": d["summary"]["percent_covered"],
                "statements": d["summary"]["num_statements"],
                "missing": d["summary"]["missing_lines"],
            }
            for f, d in cov["files"].items()
        },
    }
    cases = {**j_fast["cases"], **j_slow["cases"]}
    trace = json.loads(
        (cfg.PROJECT_ROOT / "docs" / "traceability.json").read_text(encoding="utf-8")
    )
    trace_status = {}
    for req in trace:
        statuses = [cases.get(t) for t in req["tests"]]
        executed = [s for s in statuses if s]
        trace_status[req["id"]] = {
            "n_tests": len(req["tests"]),
            "n_executed": len(executed),
            "n_cases": sum(s["passed"] + s["failed"] + s["skipped"] for s in executed),
            "all_passed": bool(executed)
            and len(executed) == len(statuses)
            and all(s["failed"] == 0 and s["passed"] > 0 for s in executed),
        }
    tests_total = j_fast["tests"] + j_slow["tests"]
    tests_failed = j_fast["failures"] + j_fast["errors"] + j_slow["failures"] + j_slow["errors"]

    res = json.loads((cfg.REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
    rob = json.loads((cfg.REPORTS_DIR / "robustness.json").read_text(encoding="utf-8"))
    perf = json.loads((cfg.REPORTS_DIR / "performance.json").read_text(encoding="utf-8"))
    sec = json.loads((cfg.REPORTS_DIR / "security.json").read_text(encoding="utf-8"))
    reg = json.loads((cfg.REPORTS_DIR / "regression.json").read_text(encoding="utf-8"))
    name = res["selected_model"]["name"]
    m = res["models"][name]
    t, boot = m["test"], res["bootstrap"]["models"][name]
    measured = {
        "recall": t["recall"],
        "fpr": t["fpr"],
        "precision": t["precision"],
        "average_precision": t["average_precision"],
        "roc_auc": t["roc_auc"],
        "generalization_gap_ap": m["train"]["average_precision"] - t["average_precision"],
        "repro_max_abs_diff": res["reproducibility"]["repro_max_abs_diff"],
        "reload_max_abs_diff": res["reproducibility"]["reload_max_abs_diff"],
        **rob["criteria_feed"],
        **perf["criteria_feed"],
        **sec["criteria_feed"],
        **reg["criteria_feed"],
        "tests_failed": tests_failed,
        "coverage_pct": coverage["percent"],
        "ruff_errors": n_ruff + (0 if ruff_fmt.returncode == 0 else 1),
    }
    intervals = {
        "recall": m["ci_test"]["recall"],
        "fpr": m["ci_test"]["fpr"],
        "precision": m["ci_test"]["precision"],
        "average_precision": (boot["average_precision"]["low"], boot["average_precision"]["high"]),
        "roc_auc": (boot["roc_auc"]["low"], boot["roc_auc"]["high"]),
    }
    rows = check_criteria(measured, intervals=intervals)
    verdict = qualification_verdict(rows)
    out = {
        "candidate": name,
        "measured": measured,
        "criteria": rows,
        "verdict": verdict,
        "tests": {
            "fast": {k: v for k, v in j_fast.items() if k != "cases"},
            "slow_and_full_data": {k: v for k, v in j_slow.items() if k != "cases"},
            "total": tests_total,
            "failed": tests_failed,
            "fast_exit_code": fast.returncode,
            "slow_exit_code": slow.returncode,
        },
        "coverage": coverage,
        "ruff": {
            "check_exit": ruff.returncode,
            "format_exit": ruff_fmt.returncode,
            "errors": n_ruff,
        },
        "traceability_status": trace_status,
        "cases": cases,
        "seconds": time.perf_counter() - t0,
    }
    dump_json(out, cfg.REPORTS_DIR / "qualification.json")
    print(
        f"Tests : {tests_total} (échecs : {tests_failed}) | couverture {coverage['percent']:.1f} % | ruff : {n_ruff}"
    )
    print(
        f"Verdict : {verdict['verdict']} | échecs : {verdict['failed']} | incertains : {verdict['uncertain_criteria']}"
    )


if __name__ == "__main__":
    main()
