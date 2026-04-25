from __future__ import annotations
"""Mode selection rules for multi-mode RCPSP.

Context-free rules choose a mode without considering current schedule state.
Context-aware rules examine current resource availability.

Each base rule is implemented as a scoring function returning a list of scores
(one per mode, lower = better). Composite rules pair a primary with a
tie-breaker, producing (primary, tiebreak) tuples for lexicographic selection.

References:
  - Lova, Tormos & Barber (2006): systematic study of mode selection rules
  - Van Peteghem & Vanhoucke (2011): resource scarceness characteristics
"""

import hashlib
import random
from mrcpsp import Project, Activity
from sgs import _find_earliest_feasible_start


# ---------------------------------------------------------------------------
# Base scoring functions — each returns list[numeric], lower = better
# ---------------------------------------------------------------------------

def _shortest_duration_scores(activity: Activity, **_) -> list:
    """Score = duration (lower = shorter = better). Shortest Feasible Mode
    (SFM) in Lova, Tormos & Barber (2006); Sum-of-Durations (SOD) mode
    characteristic in Van Peteghem & Vanhoucke (2011), who attribute the
    idea to Boctor (1993)."""
    return [mode.duration for mode in activity.modes]


def _min_resource_scores(activity: Activity, **_) -> list:
    """Score = total renewable demand Σ_r k_{j,r,m} (lower = better).
    Standard context-free mode score; related to — but simpler than —
    Lova, Tormos & Barber (2006) LRP (Least Resource Proportion), which
    normalizes each demand by its resource's capacity."""
    return [sum(mode.renewable_demands) for mode in activity.modes]


def _ltru_scores(activity: Activity, **_) -> list:
    """Least Total Resource Usage: d_{j,m} × Σ_r k_{j,r,m} over renewable
    resources. Lower = better. Buddhakulsomsiri & Kim (2007), attributed
    there to Boctor (1993) with the duration multiplier added by B&K."""
    return [mode.duration * sum(mode.renewable_demands)
            for mode in activity.modes]


def _longest_duration_scores(activity: Activity, **_) -> list:
    """Score = -duration (longer = better). Lova, Tormos & Barber (2006)
    use longest_duration as the EFFT tie-breaker: when multiple modes share
    the earliest feasible finish time, pick the longest-duration one. Their
    stated rationale is that a longer duration at the same finish time
    implies an earlier start, which lets other activities in the serial
    scheduling process be executed earlier."""
    return [-mode.duration for mode in activity.modes]


def _earliest_start_scores(activity: Activity, project: Project = None,
                           resource_profile=None, earliest_possible: int = 0,
                           **_) -> list:
    """Score = earliest feasible start time given the current renewable
    profile (lower = better). Context-aware analogue of the Shortest
    Feasible Mode (SFM) idea in Lova, Tormos & Barber (2006), scoring on
    earliest start rather than on shortest duration."""
    scores = []
    for mode in activity.modes:
        t = _find_earliest_feasible_start(
            mode.duration, mode.renewable_demands,
            project.renewable_capacities, resource_profile, earliest_possible
        )
        scores.append(t)
    return scores


def _earliest_finish_scores(activity: Activity, project: Project = None,
                            resource_profile=None, earliest_possible: int = 0,
                            **_) -> list:
    """Score = earliest feasible finish time (lower = better). This is the
    primary score of EFFT (Earliest Feasible Finish Time) in Lova, Tormos
    & Barber (2006), their recommended mode selection rule; paired with
    longest_duration as tie-breaker it reproduces their full EFFT rule."""
    scores = []
    for mode in activity.modes:
        t = _find_earliest_feasible_start(
            mode.duration, mode.renewable_demands,
            project.renewable_capacities, resource_profile, earliest_possible
        )
        scores.append(t + mode.duration)
    return scores


def _resource_fitting_scores(activity: Activity, project: Project = None,
                             resource_profile=None, earliest_possible: int = 0,
                             **_) -> list:
    """Score = total slack summed across the mode's execution window at
    its earliest feasible start (lower = tighter fit = better). Standard
    context-aware mode score; no specific paper citation."""
    scores = []
    for mode in activity.modes:
        t = _find_earliest_feasible_start(
            mode.duration, mode.renewable_demands,
            project.renewable_capacities, resource_profile, earliest_possible
        )
        total_slack = 0
        for dt in range(mode.duration) if mode.duration > 0 else [0]:
            for r in range(len(project.renewable_capacities)):
                used = resource_profile[r][t + dt] if (t + dt) < len(resource_profile[r]) else 0
                available = project.renewable_capacities[r] - used
                total_slack += available - mode.renewable_demands[r]
        scores.append(total_slack)
    return scores


def _index_scores(activity: Activity, **_) -> list:
    """Score = mode index (lower index preferred)."""
    return list(range(len(activity.modes)))


# ---------------------------------------------------------------------------
# Base rule registry
# ---------------------------------------------------------------------------

_CONTEXT_FREE_BASE = {"shortest_duration", "longest_duration", "min_resource"}
_CONTEXT_AWARE_BASE = {"earliest_start", "earliest_finish", "resource_fitting"}

_BASE_MODE_RULES = {
    "shortest_duration": _shortest_duration_scores,
    "longest_duration": _longest_duration_scores,
    "min_resource": _min_resource_scores,
    "LTRU": _ltru_scores,
    "earliest_start": _earliest_start_scores,
    "earliest_finish": _earliest_finish_scores,
    "resource_fitting": _resource_fitting_scores,
    "INDEX": _index_scores,
}


# ---------------------------------------------------------------------------
# Composite rule builder
# ---------------------------------------------------------------------------

def make_composite_mode_rule(primary_name: str, tiebreak_name: str):
    """Build a mode selection function that uses primary_name as the main rule
    and tiebreak_name to break ties. Selects the mode with the lowest
    (primary, tiebreak) tuple."""
    primary_fn = _BASE_MODE_RULES[primary_name]
    tiebreak_fn = _BASE_MODE_RULES[tiebreak_name]

    def composite(activity: Activity, **kw) -> int:
        kw["activity"] = activity
        scores = list(zip(primary_fn(**kw), tiebreak_fn(**kw)))
        return min(range(len(scores)), key=lambda m: scores[m])

    return composite


# ---------------------------------------------------------------------------
# Build the full registry
# ---------------------------------------------------------------------------

_PRIMARY_NAMES = ["shortest_duration", "longest_duration", "min_resource",
                  "earliest_start", "earliest_finish", "resource_fitting"]
_TIEBREAK_NAMES = ["shortest_duration", "longest_duration", "min_resource",
                   "earliest_start", "earliest_finish", "resource_fitting"]

CONTEXT_FREE_RULES = {}
CONTEXT_AWARE_RULES = {}

for p in _PRIMARY_NAMES:
    for t in _TIEBREAK_NAMES:
        if p == t:
            continue
        name = f"{p}/{t}"
        rule_fn = make_composite_mode_rule(p, t)

        # Context-aware if either primary or tie-breaker is context-aware
        is_ca = p in _CONTEXT_AWARE_BASE or t in _CONTEXT_AWARE_BASE
        if is_ca:
            CONTEXT_AWARE_RULES[name] = rule_fn
        else:
            CONTEXT_FREE_RULES[name] = rule_fn

# Random mode placeholder — the actual function is created per-combo by
# make_random_mode() with a local, seeded RNG.  This placeholder is kept in
# the registry so that iteration over MODE_RULES still includes "random".
def _random_mode_placeholder(activity: Activity, **_) -> int:
    """Placeholder — should not be called directly; use get_mode_fn()."""
    raise RuntimeError(
        "random_mode called without seeding; use get_mode_fn() to obtain "
        "a properly seeded random mode function"
    )

# LTRU as a standalone context-free rule (ties broken by mode index, consistent
# with how B&K 2007 present it — no explicit tie-breaker paired with LTRU).
CONTEXT_FREE_RULES["LTRU"] = make_composite_mode_rule("LTRU", "INDEX")

for _name in ("random", "random2", "random3", "random4"):
    CONTEXT_FREE_RULES[_name] = _random_mode_placeholder

MODE_RULES = {**CONTEXT_FREE_RULES, **CONTEXT_AWARE_RULES}


# ---------------------------------------------------------------------------
# Seeded random mode factory
# ---------------------------------------------------------------------------

def combo_seed(project_seed: int, sgs_name: str, pr_name: str, mr_name: str) -> int:
    """Deterministic seed from instance data and heuristic combination name.

    Uses SHA-256 rather than hash() because hash() on strings is salted by
    PYTHONHASHSEED and varies across processes.
    """
    key = f"{project_seed}:{sgs_name}:{pr_name}:{mr_name}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def make_random_mode(seed: int):
    """Create a random mode selector with a local, deterministically seeded RNG."""
    rng = random.Random(seed)

    def _random_mode(activity: Activity, **_) -> int:
        return rng.randint(0, len(activity.modes) - 1)

    return _random_mode


def get_mode_fn(mr_name: str, project, sgs_name: str, pr_name: str):
    """Get a mode selection function for the given combo.

    For stochastic rules (currently only 'random'), returns a closure with a
    local RNG seeded from both the instance data and the full heuristic
    combination name.  Deterministic rules are returned as-is.
    """
    if mr_name.startswith("random"):
        seed = combo_seed(project.seed(), sgs_name, pr_name, mr_name)
        return make_random_mode(seed)
    return MODE_RULES[mr_name]
