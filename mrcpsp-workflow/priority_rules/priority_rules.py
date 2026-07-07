from __future__ import annotations

from .heuristics import *
"""Priority rules for activity ordering in schedule generation schemes.

Priority rules surveyed in:
  - Hartmann & Kolisch (2000), Kolisch & Hartmann (2006)
  - Lova, Tormos & Barber (2006) for MRCPSP specifically
  - Melchiors, Kolisch & Kanet (2024)

Each base rule returns a list of numeric values (lower = higher priority).
Composite rules pair a primary rule with a tie-breaker, returning tuples.

All priority rules have been written in a static form, to limit complexity.

References:
-  A. Lova, P. Tormos, and F. Barber, ‘Multi-mode resource constrained project scheduling:
   Scheduling schemes, priority rules and mode selection rules’, Inteligencia Artificial. 
   Revista Iberoamericana de Inteligencia Artificial, vol. 10, no. 30, pp. 69–86, 2006.
   
-  E. W. Davis and J. H. Patterson, ‘A comparison of heuristic and optimum solutions in
   resource-constrainedproject scheduling’, Management science, vol. 21, 
   no. 8, pp. 944–955, 1975.
   
-  R. Kolisch, ‘Serial and parallel resource-constrained project scheduling methods 
   revisited: Theory and computation’, European Journal of Operational Research, 
   vol. 90, no. 2, pp. 320–333, 1996.
   
-  G. Ulusoy and L. Özdamar, ‘Heuristic Performance and Network/Resource Characteristics in
   Resource-Constrained Project Scheduling’, The Journal of the Operational Research Society,
   vol. 40, no. 12, pp. 1145–1152, 1989.
"""

import random
from mrcpsp import Project

# ---------------------------------------------------------------------------
# Base priority rules — each returns list[numeric], lower = higher priority
# ---------------------------------------------------------------------------

def _lft_values(project: Project, mode_assignments: list[int]) -> list:
    return LFT.prioritise(project, mode_assignments)


def _lst_values(project: Project, mode_assignments: list[int]) -> list:
    return LST.prioritise(project, mode_assignments)


def _lstlft_values(project: Project, mode_assignments: list[int]) -> list:
    return LSTLFT.prioritise(project, mode_assignments)


def _rwk_values(project: Project, mode_assignments: list[int]) -> list:
    return RWK.prioritise(project, mode_assignments)


def _mslk_values(project: Project, mode_assignments: list[int]) -> list:
    return MSLK.prioritise(project, mode_assignments)


def _mts_values(project: Project, mode_assignments: list[int]) -> list:
    return MTS.prioritise(project, mode_assignments)


def _grpw_values(project: Project, mode_assignments: list[int]) -> list:
    return GRPW.prioritise(project, mode_assignments)


def _wrup_values(project: Project, mode_assignments: list[int]) -> list:
    return WRUP.prioritise(project, mode_assignments)


def _spt_values(project: Project, mode_assignments: list[int]) -> list:
    return SPT.prioritise(project, mode_assignments)


def _mis_values(project: Project, mode_assignments: list[int]) -> list:
    return NIS.prioritise(project, mode_assignments)


def _grd_values(project: Project, mode_assignments: list[int]) -> list:
    return GRD.prioritise(project, mode_assignments)


def _index_values(project: Project, mode_assignments: list[int]) -> list:
    return AN.prioritise(project, mode_assignments)


_BASE_RULES = {
    "LFT": _lft_values,
    "LST": _lst_values,
    "LSTLFT": _lstlft_values,
    "MSLK": _mslk_values,
    "MTS": _mts_values,
    "GRPW": _grpw_values,
    "WRUP": _wrup_values,
    "RWK": _rwk_values,
    "SPT": _spt_values,
    "MIS": _mis_values,
    "GRD": _grd_values,
    "INDEX": _index_values,
}

# ---------------------------------------------------------------------------
# Composite rule builder
# ---------------------------------------------------------------------------

def make_composite_rule(primary_name: str, tiebreak_name: str):
    """Build a priority function that uses primary_name as the main rule
    and tiebreak_name to break ties. Returns tuples for lexicographic sort."""
    primary_fn = _BASE_RULES[primary_name]
    tiebreak_fn = _BASE_RULES[tiebreak_name]

    def composite(project: Project, mode_assignments: list[int] = None, **kw) -> list:
        kw["project"] = project
        if mode_assignments is not None:
            kw["mode_assignments"] = mode_assignments
        return list(zip(primary_fn(**kw), tiebreak_fn(**kw)))

    return composite


# ---------------------------------------------------------------------------
# Build the full registry
# ---------------------------------------------------------------------------

_PRIMARY_NAMES = ["LFT", "LST", "LSTLFT", "MSLK", "MTS", "GRPW", "WRUP", "RWK",
                  "SPT", "MIS", "GRD"]
_TIEBREAK_NAMES = ["LFT", "LST", "LSTLFT", "MSLK", "MTS", "GRPW", "WRUP", "RWK",
                   "SPT", "MIS", "GRD"]

PRIORITY_RULES = {}
for p in _PRIMARY_NAMES:
    for t in _TIEBREAK_NAMES:
        if p == t:
            continue
        PRIORITY_RULES[f"{p}/{t}"] = make_composite_rule(p, t)


# ---------------------------------------------------------------------------
# Random priority rule (placeholder + seeded factory)
# ---------------------------------------------------------------------------

def _random_priority_placeholder(project: Project, **_) -> list:
    """Placeholder — should not be called directly; use get_priority_fn()."""
    raise RuntimeError(
        "random priority called without seeding; use get_priority_fn() to "
        "obtain a properly seeded random priority function"
    )


PRIORITY_RULES["random"] = _random_priority_placeholder


def make_random_priority(seed: int):
    """Create a random priority function with a local, deterministically seeded RNG.

    Returns one random float per activity on each call; lower = higher priority.
    """
    rng = random.Random(seed)

    def _random_priority(project: Project, **_) -> list:
        return [rng.random() for _ in range(project.num_activities)]

    return _random_priority


def get_priority_fn(pr_name: str, project, sgs_name: str, mr_name: str):
    """Get a priority function for the given combo.

    For stochastic rules (currently only 'random'), returns a closure with a
    local RNG seeded from both the instance data and the full heuristic
    combination name. Deterministic rules are returned as-is.
    """
    if pr_name.startswith("random"):
        from mode_rules import combo_seed
        seed = combo_seed(project.seed(), sgs_name, pr_name, mr_name)
        return make_random_priority(seed)
    return PRIORITY_RULES[pr_name]
