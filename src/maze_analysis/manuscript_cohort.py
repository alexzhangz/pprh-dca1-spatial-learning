"""Manuscript definitions collected from manuscript_cohort.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import json

from pathlib import Path

from typing import Any, Mapping

DEFAULT_MANUSCRIPT_COHORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "manuscript_cohort_v3_20260804.json"
)

PROJECT_KEYS = ("figure_8_maze", "t_maze")

GROUP_KEYS = ("control", "inhibition", "pprh_recording")

EXPECTED_DAYS = (1, 2, 3, 4, 5)

def load_manuscript_cohort(path: str | Path = DEFAULT_MANUSCRIPT_COHORT_PATH) -> dict[str, Any]:
    """Load and validate one manuscript-only cohort contract."""

    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("status") != "current":
        raise ValueError(f"manuscript cohort contract is not current: {resolved}")
    if tuple(int(day) for day in contract.get("days", ())) != EXPECTED_DAYS:
        raise ValueError("manuscript cohort contract must cover exactly D1-D5")
    if not str(contract.get("version", "")).strip():
        raise ValueError("manuscript cohort contract is missing a version")

    included: set[str] = set()
    for project in PROJECT_KEYS:
        if project not in contract:
            raise ValueError(f"manuscript cohort contract is missing {project}")
        project_contract = contract[project]
        exclusions = set(
            str(mouse)
            for mouse in project_contract.get("whole_mouse_exclusions", {})
        )
        for group in GROUP_KEYS:
            mice = tuple(str(mouse) for mouse in project_contract.get(group, ()))
            if not mice:
                raise ValueError(f"{project}/{group} must contain at least one mouse")
            if len(mice) != len(set(mice)):
                raise ValueError(f"{project}/{group} contains duplicate mouse IDs")
            overlap = included.intersection(mice)
            if overlap:
                raise ValueError(f"mouse IDs are reused across contract groups: {sorted(overlap)}")
            if exclusions.intersection(mice):
                raise ValueError(
                    f"{project}/{group} includes excluded mice: {sorted(exclusions.intersection(mice))}"
                )
            included.update(mice)
    rules = contract.get("rules", {})
    required_true = (
        "whole_mouse_exclusions_are_task_wide",
        "posthoc_outcome_selected_exclusions_present",
        "preserve_prior_cohort_releases",
    )
    missing_rules = [rule for rule in required_true if not rules.get(rule)]
    if missing_rules:
        raise ValueError(f"manuscript cohort contract lacks required rules: {missing_rules}")
    contract["_contract_path"] = str(resolved)
    return contract

def group_mice(
    contract: Mapping[str, Any],
    project: str,
    *,
    include_recording: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return the manuscript group-to-mouse mapping for one project."""

    project_contract = contract[project]
    result = {
        "C": tuple(str(mouse) for mouse in project_contract["control"]),
        "I": tuple(str(mouse) for mouse in project_contract["inhibition"]),
    }
    if include_recording:
        result["P"] = tuple(
            str(mouse) for mouse in project_contract["pprh_recording"]
        )
    return result

def cohort_count_label(groups: Mapping[str, tuple[str, ...]]) -> str:
    """Return an explicit group-count label such as ``C5_I7``."""

    return "_".join(f"{group}{len(tuple(mice))}" for group, mice in groups.items())
