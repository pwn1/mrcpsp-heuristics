#!/usr/bin/env python3
"""Standalone solution checker for multi-mode RCPSP.

This file is intentionally self-contained with ZERO imports from other
project modules. It has its own instance parser and data structures so
that it can be verified independently of the heuristic implementation.

Usage:
    python3 check_solution.py <instance.mm> <solution.csv>

Solution file format (CSV, one header line):
    activity,mode,start
    0,0,0
    1,2,3
    ...

Activities are 0-based. Modes are 0-based.
"""

import csv
import re
import sys


# ---------------------------------------------------------------------------
# Self-contained data structures (no imports from mrcpsp.py)
# ---------------------------------------------------------------------------

class _Mode:
    __slots__ = ("duration", "renewable", "nonrenewable")
    def __init__(self, duration, renewable, nonrenewable):
        self.duration = duration
        self.renewable = list(renewable)
        self.nonrenewable = list(nonrenewable)


class _Activity:
    __slots__ = ("id", "modes", "successors")
    def __init__(self, id, modes, successors):
        self.id = id
        self.modes = list(modes)
        self.successors = list(successors)


class _Instance:
    __slots__ = ("n", "num_r", "num_nr", "r_cap", "nr_cap", "activities")
    def __init__(self, n, num_r, num_nr, r_cap, nr_cap, activities):
        self.n = n
        self.num_r = num_r
        self.num_nr = num_nr
        self.r_cap = list(r_cap)
        self.nr_cap = list(nr_cap)
        self.activities = list(activities)


# ---------------------------------------------------------------------------
# Self-contained instance parser (no imports from parser.py)
# ---------------------------------------------------------------------------

def _parse_instance(filepath):
    """Parse a PSPLIB multi-mode .mm file. Returns an _Instance."""
    with open(filepath) as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0

    def _skip_to(*markers):
        nonlocal i
        while i < len(lines):
            for m in markers:
                if m in lines[i]:
                    return
            i += 1
        raise ValueError(f"Marker(s) {markers!r} not found in {filepath}")

    # Number of activities (including supersource/sink)
    _skip_to("jobs", "Jobs")
    n = int(re.search(r":\s*(\d+)", lines[i]).group(1))

    # Resource counts
    _skip_to("- renewable")
    num_r = int(re.search(r":\s*(\d+)", lines[i]).group(1))
    i += 1
    _skip_to("- nonrenewable")
    num_nr = int(re.search(r":\s*(\d+)", lines[i]).group(1))

    # Precedence relations
    _skip_to("PRECEDENCE RELATIONS")
    i += 1
    while i < len(lines) and (lines[i].startswith("jobnr") or lines[i].startswith("-")):
        i += 1

    activities = [None] * n
    for _ in range(n):
        while i < len(lines) and not lines[i].strip():
            i += 1
        parts = lines[i].split()
        job = int(parts[0]) - 1
        num_succs = int(parts[2])
        succs = [int(s) - 1 for s in parts[3:3 + num_succs]]
        activities[job] = _Activity(id=job, modes=[], successors=succs)
        i += 1

    # Requests / durations
    _skip_to("REQUESTS/DURATIONS")
    i += 1
    while i < len(lines) and (
        lines[i].strip().startswith("jobnr") or
        lines[i].strip().startswith("-") or
        not lines[i].strip()
    ):
        i += 1

    cur_job = None
    while i < len(lines) and not lines[i].strip().startswith("*"):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if lines[i][0] not in (" ", "\t"):
            cur_job = int(parts[0]) - 1
            dur = int(parts[2])
            demands = [int(x) for x in parts[3:]]
        else:
            dur = int(parts[1])
            demands = [int(x) for x in parts[2:]]
        activities[cur_job].modes.append(
            _Mode(dur, demands[:num_r], demands[num_r:num_r + num_nr])
        )
        i += 1

    # Resource capacities
    _skip_to("RESOURCE AVAILABILITIES", "RESOURCEAVAILABILITIES")
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and re.match(r"\s*[RN]", lines[i]):
        i += 1
    caps = [int(x) for x in lines[i].split()]

    return _Instance(n, num_r, num_nr, caps[:num_r], caps[num_r:num_r + num_nr], activities)


# ---------------------------------------------------------------------------
# Self-contained solution parser
# ---------------------------------------------------------------------------

def _parse_solution(filepath):
    """Parse a solution CSV. Returns (mode_assignments, start_times) as lists."""
    modes = {}
    starts = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            act = int(row["activity"])
            modes[act] = int(row["mode"])
            starts[act] = int(row["start"])
    n = max(modes.keys()) + 1
    return [modes[i] for i in range(n)], [starts[i] for i in range(n)]


# ---------------------------------------------------------------------------
# Solution checker
# ---------------------------------------------------------------------------

def check_solution(inst, mode_assignments, start_times):
    """Check all MRCPSP constraints. Returns list of error strings (empty = valid)."""
    errors = []
    n = inst.n

    # --- Check dimensions ---
    if len(mode_assignments) != n:
        errors.append(f"Expected {n} mode assignments, got {len(mode_assignments)}")
        return errors
    if len(start_times) != n:
        errors.append(f"Expected {n} start times, got {len(start_times)}")
        return errors

    # --- Check mode indices are valid ---
    for i in range(n):
        if mode_assignments[i] < 0 or mode_assignments[i] >= len(inst.activities[i].modes):
            errors.append(
                f"Activity {i}: mode index {mode_assignments[i]} invalid "
                f"(has {len(inst.activities[i].modes)} modes)"
            )
    if errors:
        return errors

    # --- Check start times are non-negative ---
    for i in range(n):
        if start_times[i] < 0:
            errors.append(f"Activity {i}: negative start time {start_times[i]}")

    # --- Check precedence constraints ---
    for act in inst.activities:
        mode = act.modes[mode_assignments[act.id]]
        finish = start_times[act.id] + mode.duration
        for s in act.successors:
            if finish > start_times[s]:
                errors.append(
                    f"Precedence: activity {act.id} finishes at t={finish} "
                    f"but successor {s} starts at t={start_times[s]}"
                )

    # --- Check renewable resource constraints (per time slot) ---
    makespan = max(
        start_times[i] + inst.activities[i].modes[mode_assignments[i]].duration
        for i in range(n)
    )
    for t in range(makespan):
        usage = [0] * inst.num_r
        for i in range(n):
            mode = inst.activities[i].modes[mode_assignments[i]]
            if start_times[i] <= t < start_times[i] + mode.duration:
                for r in range(inst.num_r):
                    usage[r] += mode.renewable[r]
        for r in range(inst.num_r):
            if usage[r] > inst.r_cap[r]:
                errors.append(
                    f"Renewable resource R{r+1} at t={t}: "
                    f"usage {usage[r]} > capacity {inst.r_cap[r]}"
                )

    # --- Check non-renewable resource constraints (whole schedule) ---
    for nr in range(inst.num_nr):
        total = sum(
            inst.activities[i].modes[mode_assignments[i]].nonrenewable[nr]
            for i in range(n)
        )
        if total > inst.nr_cap[nr]:
            errors.append(
                f"Non-renewable resource N{nr+1}: "
                f"total usage {total} > capacity {inst.nr_cap[nr]}"
            )

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 check_solution.py <instance.mm> <solution.csv>")
        print()
        print("Solution CSV format:")
        print("  activity,mode,start")
        print("  0,0,0")
        print("  1,2,5")
        print("  ...")
        sys.exit(1)

    instance_path = sys.argv[1]
    solution_path = sys.argv[2]

    inst = _parse_instance(instance_path)
    modes, starts = _parse_solution(solution_path)

    errors = check_solution(inst, modes, starts)

    # Compute makespan
    makespan = max(
        starts[i] + inst.activities[i].modes[modes[i]].duration
        for i in range(inst.n)
    )

    if errors:
        print(f"INVALID (makespan={makespan}, {len(errors)} violations)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"VALID (makespan={makespan})")
        sys.exit(0)


if __name__ == "__main__":
    main()
