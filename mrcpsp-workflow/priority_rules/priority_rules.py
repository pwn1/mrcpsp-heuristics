from __future__ import annotations

from abc import ABC, abstractmethod

from priority_rules.critical_path import CriticalPathMethodCalculator

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

def _rwk_values(project: Project, mode_assignments) -> list:
    return RWK.prioritise(project, mode_assignments)


def _mslk_values(project: Project, mode_assignments: list[int]) -> list:
    """Minimum Slack: slack = LST - EST = (LFT - duration) - EST.

    Lower slack = more critical = higher priority.
    Hartmann & Kolisch (2000), Kolisch & Hartmann (2006).
    """
    return CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments).slack


def _mts_values(project: Project, **_) -> list:
    """Most Total Successors: count of transitive successors. Higher =
    higher priority (negated). Kolisch 1996 EJOR Table 1, attributed
    there to Alvarez-Valdes & Tamarit (1989)."""
    all_succs = PriorityRule._compute_successors_recursive(project)
    return [-len(s) for s in all_succs]


def _grpw_values(project: Project, mode_assignments: list[int]) -> list:
    """Greatest Rank Positional Weight: d_j + sum of immediate successor
    durations. Kolisch 1996 EJOR Table 1 (attributed there to
    Alvarez-Valdes & Tamarit 1989); also Lova, Tormos & Barber (2006)
    Table 1. Higher = higher priority (negated for lower-is-better
    convention). Lova computes this with shortest-mode durations; we
    use current-mode."""
    durations = [
        project.activities[i].modes[mode_assignments[i]].duration
        for i in range(project.num_activities)
    ]
    return [
        -(durations[i] + sum(durations[s] for s in project.activities[i].successors))
        for i in range(project.num_activities)
    ]


def _wrup_values(project: Project, mode_assignments: list[int]) -> list:
    """Weighted Resource Utilisation Ratio and Precedence.

    Ulusoy & Özdamar (1989), as tabulated by Kolisch (1996) EJOR Table 1:
      WRUP(j) = 0.7 * |S_j| + 0.3 * sum_{r in renewable} k_jr / K_r
    where S_j is the set of immediate successors and the resource sum is over
    renewable resources only. Higher WRUP = higher priority (negated here for
    the lower-is-better convention).
    """
    n = project.num_activities
    caps = project.renewable_capacities
    result = []
    for i in range(n):
        mode = project.activities[i].modes[mode_assignments[i]]
        num_succ = len(project.activities[i].successors)
        demand_ratio = sum(
            d / c for d, c in zip(mode.renewable_demands, caps) if c > 0
        )
        result.append(-(0.7 * num_succ + 0.3 * demand_ratio))
    return result


def _spt_values(project: Project, mode_assignments: list[int]) -> list:
    """Shortest Processing Time: activity duration. Lower = higher
    priority. Lova, Tormos & Barber (2006) Table 1 (computed there with
    shortest-mode durations; we use current-mode)."""
    return [
        project.activities[i].modes[mode_assignments[i]].duration
        for i in range(project.num_activities)
    ]


def _mis_values(project: Project, **_) -> list:
    """Most Immediate Successors: count of direct successors. Higher =
    higher priority (negated). Called NIS in Lova, Tormos & Barber (2006)
    Table 1."""
    return [-len(project.activities[i].successors)
            for i in range(project.num_activities)]


def _grd_values(project: Project, mode_assignments: list[int]) -> list:
    """Greatest Resource Demand: d_j * sum of renewable demands.
    Lova, Tormos & Barber (2006) Table 1. Lower = higher priority (negated)."""
    return [
        -(project.activities[i].modes[mode_assignments[i]].duration *
          sum(project.activities[i].modes[mode_assignments[i]].renewable_demands))
        for i in range(project.num_activities)
    ]


def _index_values(project: Project, **_) -> list:
    return list(range(project.num_activities))


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


class PriorityRule(ABC):
    """
    We use the convention that lower values, mean an item is higher priority.

    This means, for some priority rules which are traditionally "higher is better"
    we have to take their negation.
    """
    @staticmethod
    @abstractmethod
    def prioritise(project:Project, mode_assignments: list[int]) -> list[int]:
        pass

    @staticmethod
    def _compute_successors_recursive(project: Project) -> list[set[int]]:
        """Compute the transitive closure of successors for each activity."""
        all_successors: list[set[int] | None] = [None] * project.num_activities

        def _get(act_id: int) -> set[int]:
            cached = all_successors[act_id]
            if cached is not None:
                return cached

            result = set()
            for s in project.activities[act_id].successors:
                result.add(s)
                result |= _get(s)

            all_successors[act_id] = result
            return result

        for a in range(project.num_activities):
            _get(a)

        # By now every slot has been filled in, so this is safe.
        return [s if s is not None else set() for s in all_successors]


class LFT(PriorityRule):
    """ Lastest Finish Time: Calculated by completing a CPM (critical
    path method) backwards pass.

    Described in Davis and Patterson (1975) as being calculated by
    "usual critical path methods", it was found by Lova, Tormos & Barber
    (2006) to be in the top 3 best performing heuristics out of 14
    heuristics on both serial and parallel schedule generation schemes. """

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments).latest_finish_time

class LST(PriorityRule):
    """
    Described in Davis and Patterson (1975) as being calculated by critical path
    methods, it was found by Lova, Tormos & Barber (2006) to be in the top 3
    best performing heuristics out of 14 heuristics on both serial and parallel
    schedule generation schemes.

    It is of course related to LFT as LST_j = LFT_j - d_j (where d_j is the
    duration of task j).

    Note: As Davis and Patterson (1975) prove, this is equivalent to the MINSLACK
    (minimum slack) priority rule when using parallel schedule generation.
    """
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments).latest_start_time

class LSTLFT(PriorityRule):
    """ Lova, Tormos & Barber (2006) define LSTLFT_j = LFT_j + LST_j
    (the sum of latest start time and latest finish time). They found
    it to have the best performance out of 14 heuristics with both serial and
    parallel schedule generation schemes."""
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        cpm_schedule = CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments)
        return [
            cpm_schedule.latest_start_time[i] + cpm_schedule.latest_finish_time[i]
            for i in range(len(cpm_schedule.latest_start_time))
        ]

class RWK(PriorityRule):
    """ Lova, Tormos & Barber (2006) define Maximum Remaining Work (RWK)
    as "the sum of the [...] duration of the activity and all
    its [transitive] successors".

    In Lova, Tormos & Barber (2006), they compute this heuristic using
    the shortest duration mode for each activity. We however, use the
    current-mode durations. For context-aware assignment, this means the first
    pass will be the equivalent (as shortest modes are assumed). However, for
    the second pass/context-unaware assignment it allows us to be more
    precise with our calculations. This same approach is used in the GRPW
    heuristic.

    In this heuristic, a higher value corresponds to higher priority,
    therefore we have to negate the values to fit the lower-is-better
    convention."""
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        all_successors = PriorityRule._compute_successors_recursive(project)

        durations = project.durations_given_modes(mode_assignments)

        return [
            -(durations[i] + sum(durations[s] for s in all_successors[i]))
            for i in range(project.num_activities)
        ]

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
