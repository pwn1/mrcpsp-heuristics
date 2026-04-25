from __future__ import annotations
"""Schedule Generation Schemes for multi-mode RCPSP.

References:
  - Kolisch (1996): Serial and parallel SGS (foundational, not on DBLP)
  - Sprecher & Drexl (1998): Multi-mode sequencing algorithm
  - Hartmann & Kolisch (2000), Kolisch & Hartmann (2006): experimental evaluation
  - Lova, Tormos & Barber (2006): SGS + priority rules + mode selection for MRCPSP
"""

from typing import Callable
from mrcpsp import Project, Schedule


def _make_resource_profile(num_resources: int, horizon: int) -> list[list[int]]:
    return [[0] * horizon for _ in range(num_resources)]


def _update_resource_profile(profile, start: int, duration: int, demands):
    for r, d in enumerate(demands):
        row = profile[r]
        for t in range(start, start + duration):
            row[t] += d


def _compute_horizon(project: Project) -> int:
    return sum(max(m.duration for m in a.modes) for a in project.activities) + 1


def _fits_renewable(profile, caps, t: int, duration: int, demands) -> bool:
    if duration == 0:
        return True
    for dt in range(duration):
        for r, d in enumerate(demands):
            if profile[r][t + dt] + d > caps[r]:
                return False
    return True


def _find_earliest_feasible_start(duration, demands, capacities, profile, earliest):
    """Earliest start >= earliest where renewable resource constraints are met.
    Scans backward over the duration window on conflict, so we can skip past
    the latest conflicting slot rather than advancing by 1."""
    if duration == 0:
        return earliest
    t = earliest
    while True:
        skip = 0
        for dt in range(duration - 1, -1, -1):
            col = t + dt
            for r, d in enumerate(demands):
                if profile[r][col] + d > capacities[r]:
                    skip = dt + 1
                    break
            if skip:
                break
        if not skip:
            return t
        t += skip


def _check_nonrenewable_feasibility(project, mode_assignments) -> bool:
    for nr in range(project.num_nonrenewable):
        total = sum(
            project.activities[i].modes[mode_assignments[i]].nonrenewable_demands[nr]
            for i in range(project.num_activities)
        )
        if total > project.nonrenewable_capacities[nr]:
            return False
    return True


def _repair_nonrenewable(project: Project, mode_assignments: list[int]) -> bool:
    """Greedily repair mode assignments to satisfy non-renewable constraints.

    At each step, picks the mode switch that gives the best total NR reduction,
    with duration increase as tie-breaker. Returns True if feasible.
    """
    n = project.num_activities
    nnr = project.num_nonrenewable
    caps = project.nonrenewable_capacities

    totals = [
        sum(project.activities[i].modes[mode_assignments[i]].nonrenewable_demands[nr]
            for i in range(n))
        for nr in range(nnr)
    ]

    for _ in range(n * 10):
        excesses = [max(0, totals[nr] - caps[nr]) for nr in range(nnr)]
        if sum(excesses) == 0:
            return True

        best = (0, float("inf"), -1, -1)  # (reduction, dur_cost, act, mode)
        for i in range(n):
            act = project.activities[i]
            if len(act.modes) <= 1:
                continue
            cur_mode = mode_assignments[i]
            cur_dur = act.modes[cur_mode].duration
            cur_nr = act.modes[cur_mode].nonrenewable_demands

            for m, new_mode in enumerate(act.modes):
                if m == cur_mode:
                    continue
                new_nr = new_mode.nonrenewable_demands
                reduction = 0
                for nr in range(nnr):
                    diff = cur_nr[nr] - new_nr[nr]
                    new_ex = max(0, totals[nr] - diff - caps[nr])
                    reduction += excesses[nr] - new_ex
                dur_cost = new_mode.duration - cur_dur
                if reduction > best[0] or (reduction == best[0] and reduction > 0
                                           and dur_cost < best[1]):
                    best = (reduction, dur_cost, i, m)

        if best[2] == -1 or best[0] <= 0:
            return False
        i, m = best[2], best[3]
        old_nr = project.activities[i].modes[mode_assignments[i]].nonrenewable_demands
        new_nr = project.activities[i].modes[m].nonrenewable_demands
        for nr in range(nnr):
            totals[nr] += new_nr[nr] - old_nr[nr]
        mode_assignments[i] = m

    return _check_nonrenewable_feasibility(project, mode_assignments)


# ---------------------------------------------------------------------------
# Core scheduling loops
# ---------------------------------------------------------------------------

def _serial_schedule(project: Project, priorities, mode_assignments,
                     mode_fn: Callable = None) -> Schedule:
    """Serial SGS core. If mode_fn is given, it is called to pick each
    activity's mode at scheduling time (context-aware first pass); otherwise
    mode_assignments is used as-is (fixed-mode pass).
    """
    n = project.num_activities
    profile = _make_resource_profile(project.num_renewable, _compute_horizon(project))
    start_times = [0] * n
    finish_times = [0] * n
    preds = project.predecessors
    succs = [a.successors for a in project.activities]
    remaining = [len(p) for p in preds]
    ready = [j for j in range(n) if remaining[j] == 0]

    for _ in range(n):
        act_id = min(ready, key=lambda j: (priorities[j], j))
        ready.remove(act_id)

        ep = 0
        for p in preds[act_id]:
            if finish_times[p] > ep:
                ep = finish_times[p]
        if mode_fn is not None:
            mode_assignments[act_id] = mode_fn(
                activity=project.activities[act_id], project=project,
                resource_profile=profile, earliest_possible=ep,
            )
        mode = project.activities[act_id].modes[mode_assignments[act_id]]
        st = _find_earliest_feasible_start(
            mode.duration, mode.renewable_demands,
            project.renewable_capacities, profile, ep,
        )
        start_times[act_id] = st
        finish_times[act_id] = st + mode.duration
        _update_resource_profile(profile, st, mode.duration, mode.renewable_demands)

        for s in succs[act_id]:
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(s)

    return Schedule(mode_assignments=list(mode_assignments), start_times=start_times)


def _parallel_schedule(project: Project, priorities, mode_assignments,
                       mode_fn: Callable = None) -> Schedule | None:
    """Parallel SGS core. See _serial_schedule for mode_fn semantics."""
    n = project.num_activities
    horizon = _compute_horizon(project)
    profile = _make_resource_profile(project.num_renewable, horizon)
    start_times = [0] * n
    finish_times = [0] * n
    preds = project.predecessors
    succs = [a.successors for a in project.activities]
    remaining = [len(p) for p in preds]
    # An activity is eligible at time t iff all preds finished by t. We track
    # `pending`: activities with remaining=0 but possibly not yet reached by t.
    pending = [j for j in range(n) if remaining[j] == 0]
    num_scheduled = 0
    caps = project.renewable_capacities

    t = 0
    while num_scheduled < n:
        eligible = [j for j in pending
                    if all(finish_times[p] <= t for p in preds[j])]
        eligible.sort(key=lambda j: (priorities[j], j))

        scheduled_any = False
        for act_id in eligible:
            if mode_fn is not None:
                mode_assignments[act_id] = mode_fn(
                    activity=project.activities[act_id], project=project,
                    resource_profile=profile, earliest_possible=t,
                )
            mode = project.activities[act_id].modes[mode_assignments[act_id]]
            if not _fits_renewable(profile, caps, t, mode.duration, mode.renewable_demands):
                continue

            start_times[act_id] = t
            finish_times[act_id] = t + mode.duration
            _update_resource_profile(profile, t, mode.duration, mode.renewable_demands)
            pending.remove(act_id)
            for s in succs[act_id]:
                remaining[s] -= 1
                if remaining[s] == 0:
                    pending.append(s)
            num_scheduled += 1
            scheduled_any = True

        if not scheduled_any:
            t += 1
            if t >= horizon:
                return None

    return Schedule(mode_assignments=list(mode_assignments), start_times=start_times)


# ---------------------------------------------------------------------------
# Public SGS entry points
# ---------------------------------------------------------------------------

def _run_sgs(project: Project, priority_fn, mode_fn,
             mode_is_context_aware: bool, core) -> Schedule | None:
    """Shared driver: context-free assigns modes up front; context-aware picks
    modes during a first pass; both then NR-repair and re-schedule."""
    n = project.num_activities
    mode_assignments = [0] * n

    if mode_is_context_aware:
        proxy_modes = [min(range(len(a.modes)), key=lambda m: a.modes[m].duration)
                       for a in project.activities]
        priorities = priority_fn(project=project, mode_assignments=proxy_modes)
        core(project, priorities, mode_assignments, mode_fn=mode_fn)
    else:
        for i in range(n):
            mode_assignments[i] = mode_fn(activity=project.activities[i])

    if not _repair_nonrenewable(project, mode_assignments):
        return None
    priorities = priority_fn(project=project, mode_assignments=mode_assignments)
    return core(project, priorities, mode_assignments)


def serial_sgs(project, priority_fn, mode_fn, mode_is_context_aware=False):
    """Serial Schedule Generation Scheme. Schedules activities one at a time in
    priority order at their earliest feasible start. For context-aware mode
    rules, a two-pass approach is used (first pass selects modes, NR repair,
    second pass re-schedules)."""
    return _run_sgs(project, priority_fn, mode_fn, mode_is_context_aware,
                    _serial_schedule)


def parallel_sgs(project, priority_fn, mode_fn, mode_is_context_aware=False):
    """Parallel Schedule Generation Scheme. Advances time step by step,
    scheduling all eligible activities at each decision point. Two-pass for
    context-aware mode rules (same as serial_sgs)."""
    return _run_sgs(project, priority_fn, mode_fn, mode_is_context_aware,
                    _parallel_schedule)


SGS_SCHEMES = {
    "serial": serial_sgs,
    "parallel": parallel_sgs,
}
