from __future__ import annotations
"""Iterated first-improvement local search post-processing for MRCPSP.

Seeded with the best schedule from the constructive heuristic sweep, the
search explores three neighborhoods within a time budget:

- Mode change: replace one activity's mode with another (skip if the result
  violates non-renewable budgets).
- Activity-list swap: exchange two activities in the priority order.
- Activity-list shift: remove an activity from the priority order and reinsert
  it at a different position.

Each neighbor is evaluated by re-running the serial SGS with the modified
(priority list, modes) and applying double justification. First improvement is
accepted; on plateau the best-so-far is perturbed (random swaps + mode flips)
and search continues. Best-so-far is always preserved.

Mode-change neighbors are scanned in critical-path-first order: shortening an
activity that has slack cannot reduce the makespan, so we try zero-slack
activities first.

References:
- Hartmann (2001), 'Project scheduling with multiple modes: a genetic
  algorithm', Annals of OR 102, 111-135.
- Kolisch & Hartmann (2006), 'Experimental investigation of heuristics for
  RCPSP', EJOR 174, 23-37.
- Boctor (1996), 'Resource-constrained project scheduling by simulated
  annealing', Int. J. Production Research 34, 2335-2351 — critical-path
  targeting for mode change.
"""

import random
import time

from mrcpsp import Project, Schedule
from sgs import _serial_schedule, _check_nonrenewable_feasibility
from justification import justify
from time_window_pruning import _cpm

PERTURB_SWAPS = 3
PERTURB_MODE_FLIPS = 3


def _priority_list_from_schedule(schedule: Schedule, n: int) -> list[int]:
    return sorted(range(n), key=lambda i: (schedule.start_times[i], i))


def _list_to_priorities(priority_list: list[int]) -> list[int]:
    n = len(priority_list)
    priorities = [0] * n
    for pos, act_id in enumerate(priority_list):
        priorities[act_id] = pos
    return priorities


def _evaluate(project: Project, priority_list: list[int],
              modes: list[int]) -> Schedule | None:
    if not _check_nonrenewable_feasibility(project, modes):
        return None
    priorities = _list_to_priorities(priority_list)
    sched = _serial_schedule(project, priorities, list(modes))
    return justify(project, sched)


def _critical_activities(project: Project, schedule: Schedule) -> list[int]:
    n = project.num_activities
    durs = [project.activities[i].modes[schedule.mode_assignments[i]].duration
            for i in range(n)]
    horizon = schedule.compute_makespan(project)
    es, lf = _cpm(project, durs, horizon)
    return [j for j in range(n) if lf[j] - es[j] == durs[j]]


def _try_mode_changes(project, priority_list, modes, best_ms, rng,
                      activity_order, deadline):
    for j in activity_order:
        if deadline is not None and time.monotonic() >= deadline:
            return None
        cur_m = modes[j]
        n_modes = len(project.activities[j].modes)
        m_order = list(range(n_modes))
        rng.shuffle(m_order)
        for m in m_order:
            if m == cur_m:
                continue
            new_modes = list(modes)
            new_modes[j] = m
            sched = _evaluate(project, priority_list, new_modes)
            if sched is None:
                continue
            ms = sched.compute_makespan(project)
            if ms < best_ms:
                return new_modes, sched, ms
    return None


def _try_swaps(project, priority_list, modes, best_ms, rng, deadline):
    n = len(priority_list)
    # Skip adjacent swaps: equivalent to a shift between the same two positions,
    # which the shift neighborhood already covers.
    pairs = [(a, b) for a in range(n) for b in range(a + 2, n)]
    rng.shuffle(pairs)
    for a, b in pairs:
        if deadline is not None and time.monotonic() >= deadline:
            return None
        new_list = list(priority_list)
        new_list[a], new_list[b] = new_list[b], new_list[a]
        sched = _evaluate(project, new_list, modes)
        if sched is None:
            continue
        ms = sched.compute_makespan(project)
        if ms < best_ms:
            return new_list, sched, ms
    return None


def _try_shifts(project, priority_list, modes, best_ms, rng, deadline):
    n = len(priority_list)
    moves = [(src, dst) for src in range(n) for dst in range(n) if src != dst]
    rng.shuffle(moves)
    for src, dst in moves:
        if deadline is not None and time.monotonic() >= deadline:
            return None
        new_list = list(priority_list)
        act = new_list.pop(src)
        new_list.insert(dst, act)
        sched = _evaluate(project, new_list, modes)
        if sched is None:
            continue
        ms = sched.compute_makespan(project)
        if ms < best_ms:
            return new_list, sched, ms
    return None


def _perturb(priority_list, modes, project, rng,
             n_swaps=PERTURB_SWAPS, n_flips=PERTURB_MODE_FLIPS):
    new_list = list(priority_list)
    new_modes = list(modes)
    n = len(new_list)
    for _ in range(n_swaps):
        a = rng.randrange(n)
        b = rng.randrange(n)
        new_list[a], new_list[b] = new_list[b], new_list[a]
    for _ in range(n_flips):
        j = rng.randrange(n)
        n_modes = len(project.activities[j].modes)
        if n_modes > 1:
            new_modes[j] = rng.randrange(n_modes)
    return new_list, new_modes


def local_search(project: Project, seed_schedule: Schedule, *,
                 time_budget: float | None = None,
                 iterations: int | None = None,
                 seed: int = 42) -> tuple[Schedule, dict]:
    """Run iterated first-improvement local search from seed_schedule.

    Specify exactly one of time_budget (seconds, real-time) or iterations
    (outer-loop rounds; deterministic given seed). One outer-loop round
    consists of a full neighborhood scan (mode-change, swap, shift) followed
    by either acceptance of the first improving move or a perturbation.
    Returns (best_schedule, stats).
    """
    if (time_budget is None) == (iterations is None):
        raise ValueError(
            "specify exactly one of time_budget or iterations")
    rng = random.Random(seed)
    n = project.num_activities

    best_modes = list(seed_schedule.mode_assignments)
    best_list = _priority_list_from_schedule(seed_schedule, n)
    best_schedule = seed_schedule
    seed_ms = best_ms = seed_schedule.compute_makespan(project)

    cur_modes = list(best_modes)
    cur_list = list(best_list)
    cur_schedule = best_schedule
    cur_ms = best_ms

    t0 = time.monotonic()
    deadline = t0 + time_budget if time_budget is not None else None
    moves_accepted = perturbations = rounds = 0

    def stop():
        if iterations is not None:
            return rounds >= iterations
        return time.monotonic() >= deadline

    while not stop():
        critical = _critical_activities(project, cur_schedule)
        crit_set = set(critical)
        non_crit = [j for j in range(n) if j not in crit_set]
        rng.shuffle(critical)
        rng.shuffle(non_crit)
        act_order = critical + non_crit

        improved = False
        for try_fn in (_try_mode_changes, _try_swaps, _try_shifts):
            if deadline is not None and time.monotonic() >= deadline:
                break
            if try_fn is _try_mode_changes:
                result = try_fn(project, cur_list, cur_modes, cur_ms, rng,
                                act_order, deadline)
            else:
                result = try_fn(project, cur_list, cur_modes, cur_ms, rng,
                                deadline)
            if result is not None:
                if try_fn is _try_mode_changes:
                    cur_modes, cur_schedule, cur_ms = result
                else:
                    cur_list, cur_schedule, cur_ms = result
                moves_accepted += 1
                if cur_ms < best_ms:
                    best_modes = list(cur_modes)
                    best_list = list(cur_list)
                    best_schedule = cur_schedule
                    best_ms = cur_ms
                improved = True
                break

        if not improved:
            cur_list, cur_modes = _perturb(best_list, best_modes, project, rng)
            sched = _evaluate(project, cur_list, cur_modes)
            if sched is None:
                cur_list = list(best_list)
                cur_modes = list(best_modes)
                cur_schedule = best_schedule
                cur_ms = best_ms
            else:
                cur_schedule = sched
                cur_ms = sched.compute_makespan(project)
            perturbations += 1

        rounds += 1

    elapsed = time.monotonic() - t0
    stats = {
        'seed_ms': seed_ms,
        'final_ms': best_ms,
        'improvement': seed_ms - best_ms,
        'time_used': elapsed,
        'rounds': rounds,
        'moves_accepted': moves_accepted,
        'perturbations': perturbations,
    }
    return best_schedule, stats
