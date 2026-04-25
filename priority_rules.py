from __future__ import annotations
"""Priority rules for activity ordering in schedule generation schemes.

Priority rules surveyed in:
  - Hartmann & Kolisch (2000), Kolisch & Hartmann (2006)
  - Lova, Tormos & Barber (2006) for MRCPSP specifically
  - Melchiors, Kolisch & Kanet (2024)

Each base rule returns a list of numeric values (lower = higher priority).
Composite rules pair a primary rule with a tie-breaker, returning tuples.
"""

import random
from mrcpsp import Project


def _compute_successors_recursive(project: Project) -> list[set[int]]:
    """Compute the transitive closure of successors for each activity."""
    all_succs = [None] * project.num_activities

    def _get(act_id: int) -> set[int]:
        if all_succs[act_id] is not None:
            return all_succs[act_id]
        result = set()
        for s in project.activities[act_id].successors:
            result.add(s)
            result |= _get(s)
        all_succs[act_id] = result
        return result

    for a in range(project.num_activities):
        _get(a)
    return all_succs


def _topological_order(project: Project) -> list[int]:
    """Compute a topological ordering of activities using Kahn's algorithm."""
    n = project.num_activities
    in_degree = [0] * n
    for act in project.activities:
        for s in act.successors:
            in_degree[s] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for s in project.activities[node].successors:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    return order


# ---------------------------------------------------------------------------
# Base priority rules — each returns list[numeric], lower = higher priority
# ---------------------------------------------------------------------------

def _cpm_passes(project: Project, mode_assignments: list[int]):
    """Compute CPM forward pass (EST) and backward pass (LFT).

    Returns (est, lft, durations, topo) where:
      est[i] = earliest start time of activity i
      lft[i] = latest finish time of activity i
    """
    n = project.num_activities
    durations = [
        project.activities[i].modes[mode_assignments[i]].duration
        for i in range(n)
    ]
    est = [0] * n
    topo = _topological_order(project)
    for act_id in topo:
        for s in project.activities[act_id].successors:
            est[s] = max(est[s], est[act_id] + durations[act_id])
    makespan_ub = max(est[i] + durations[i] for i in range(n))
    lft = [makespan_ub] * n
    for act_id in reversed(topo):
        for s in project.activities[act_id].successors:
            lft[act_id] = min(lft[act_id], lft[s] - durations[s])
    return est, lft, durations, topo


def _lft_values(project: Project, mode_assignments: list[int]) -> list:
    """Latest Finish Time: LFT_j from the CPM backward pass. Lower =
    higher priority. Classical priority rule; tabulated in Kolisch 1996
    EJOR Table 1 (attributed there to Davis & Patterson 1975) and in
    Lova, Tormos & Barber (2006) Table 1, where it is a top-4 rule for
    MRCPSP under both serial and parallel SGS."""
    _, lft, _, _ = _cpm_passes(project, mode_assignments)
    return lft


def _lst_values(project: Project, mode_assignments: list[int]) -> list:
    """Latest Start Time: LST_j = LFT_j - d_j. Lower = higher priority.
    Lova, Tormos & Barber (2006) list this among the top-4 priority rules
    for MRCPSP with the serial SGS, alongside LSTLFT, LFT, and RWK."""
    _, lft, durations, _ = _cpm_passes(project, mode_assignments)
    return [lft[i] - durations[i] for i in range(project.num_activities)]


def _lstlft_values(project: Project, mode_assignments: list[int]) -> list:
    """Combined Latest Start and Finish Time: LST_j + LFT_j = 2*LFT_j - d_j.
    Lower = higher priority. Lova, Tormos & Barber (2006) report this as the
    best single priority rule for MRCPSP with the serial SGS."""
    _, lft, durations, _ = _cpm_passes(project, mode_assignments)
    return [2 * lft[i] - durations[i] for i in range(project.num_activities)]


def _rwk_values(project: Project, **_) -> list:
    """Remaining Work (Lova et al. 2006): own shortest-mode duration plus
    shortest-mode durations of all transitive successors. Higher = higher
    priority (negated for lower-is-better convention). Distinct from GRPW,
    which uses current-mode durations rather than shortest-mode."""
    all_succs = _compute_successors_recursive(project)
    min_dur = [min(m.duration for m in a.modes) for a in project.activities]
    return [
        -(min_dur[i] + sum(min_dur[s] for s in all_succs[i]))
        for i in range(project.num_activities)
    ]


def _mslk_values(project: Project, mode_assignments: list[int]) -> list:
    """Minimum Slack: slack = LST - EST = (LFT - duration) - EST.

    Lower slack = more critical = higher priority.
    Hartmann & Kolisch (2000), Kolisch & Hartmann (2006).
    """
    est, lft, durations, _ = _cpm_passes(project, mode_assignments)
    n = project.num_activities
    return [lft[i] - durations[i] - est[i] for i in range(n)]


def _mts_values(project: Project, **_) -> list:
    """Most Total Successors: count of transitive successors. Higher =
    higher priority (negated). Kolisch 1996 EJOR Table 1, attributed
    there to Alvarez-Valdes & Tamarit (1989)."""
    all_succs = _compute_successors_recursive(project)
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
