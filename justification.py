from __future__ import annotations
"""Iterated double justification, adapted from Valls et al. (2005) to MRCPSP.

Reference: Valls, Ballestin & Quintanilla (2005), "Justification and RCPSP:
a technique that pays", EJOR 165(2):375-386.

Pass orderings follow the operational specification on page 383: right-justify
processes activities in decreasing order of finish time and pushes each to its
latest feasible start; left-justify processes in increasing order of start time
and pushes each to its earliest feasible start.

Three departures from the paper:
- The paper's DJ is one right pass followed by one left pass (page 378). This
  implementation iterates R+L until no start time changes (up to max_iter),
  which is at least as strong but not what the paper calls DJ.
- The dummy source/sink are not skipped (the paper formally excludes them on
  page 377). With duration 0 they carry no resource and are pinned by lb/ub,
  so behaviorally inert.
- Mode assignments are held fixed throughout. The paper is single-mode RCPSP;
  freezing modes is a standard MRCPSP adaptation that preserves NR feasibility.
"""

from mrcpsp import Project, Schedule


def _build_profile(project: Project, start_times: list[int],
                   modes: list[int], horizon: int) -> list[list[int]]:
    profile = [[0] * horizon for _ in range(project.num_renewable)]
    for i in range(project.num_activities):
        mode = project.activities[i].modes[modes[i]]
        for t in range(start_times[i], start_times[i] + mode.duration):
            for r, d in enumerate(mode.renewable_demands):
                profile[r][t] += d
    return profile


def _apply(profile, start, duration, demands, sign):
    for r, d in enumerate(demands):
        for t in range(start, start + duration):
            profile[r][t] += sign * d


def _fits(profile, caps, start, duration, demands) -> bool:
    for t in range(start, start + duration):
        for r, d in enumerate(demands):
            if profile[r][t] + d > caps[r]:
                return False
    return True


def _latest_feasible_start(profile, caps, duration, demands,
                           lb: int, ub: int) -> int:
    """Largest t in [lb, ub] where activity (duration, demands) fits. ub is
    inclusive. Caller guarantees at least one feasible t exists (lb always is)."""
    if duration == 0:
        return ub
    for t in range(ub, lb - 1, -1):
        if _fits(profile, caps, t, duration, demands):
            return t
    return lb


def _earliest_feasible_start(profile, caps, duration, demands, lb: int) -> int:
    if duration == 0:
        return lb
    t = lb
    while True:
        if _fits(profile, caps, t, duration, demands):
            return t
        t += 1


def justify(project: Project, schedule: Schedule,
            max_iter: int = 10) -> Schedule:
    """Iterative right/left justification. Returns a new Schedule whose
    makespan is <= schedule.compute_makespan(project)."""
    n = project.num_activities
    modes = list(schedule.mode_assignments)
    start_times = list(schedule.start_times)
    durations = [project.activities[i].modes[modes[i]].duration for i in range(n)]
    demands = [project.activities[i].modes[modes[i]].renewable_demands
               for i in range(n)]
    caps = project.renewable_capacities
    succs = [project.activities[i].successors for i in range(n)]
    preds = project.predecessors

    for _ in range(max_iter):
        changed = False

        # ---- Right-justify: latest possible, processing by latest finish first
        T = max(start_times[i] + durations[i] for i in range(n))
        horizon = T + 1
        profile = _build_profile(project, start_times, modes, horizon)
        order = sorted(range(n), key=lambda i: -(start_times[i] + durations[i]))
        for j in order:
            _apply(profile, start_times[j], durations[j], demands[j], -1)
            if succs[j]:
                ub = min(start_times[s] for s in succs[j]) - durations[j]
            else:
                ub = T - durations[j]
            lb = start_times[j]
            if ub < lb:
                ub = lb
            new_start = _latest_feasible_start(
                profile, caps, durations[j], demands[j], lb, ub
            )
            if new_start != start_times[j]:
                changed = True
            start_times[j] = new_start
            _apply(profile, start_times[j], durations[j], demands[j], +1)

        # ---- Left-justify: earliest possible, processing by earliest start first
        T = max(start_times[i] + durations[i] for i in range(n))
        horizon = T + 1
        profile = _build_profile(project, start_times, modes, horizon)
        order = sorted(range(n), key=lambda i: start_times[i])
        for j in order:
            _apply(profile, start_times[j], durations[j], demands[j], -1)
            if preds[j]:
                lb = max(start_times[p] + durations[p] for p in preds[j])
            else:
                lb = 0
            new_start = _earliest_feasible_start(
                profile, caps, durations[j], demands[j], lb
            )
            if new_start != start_times[j]:
                changed = True
            start_times[j] = new_start
            _apply(profile, start_times[j], durations[j], demands[j], +1)

        if not changed:
            break

    return Schedule(mode_assignments=modes, start_times=start_times)
