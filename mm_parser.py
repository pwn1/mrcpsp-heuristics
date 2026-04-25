"""Parser for PSPLIB multi-mode RCPSP instance files."""

import re
from mrcpsp import Activity, Mode, Project


def parse_psplib(filepath: str) -> Project:
    """Parse a PSPLIB multi-mode instance file (.mm extension).

    Handles both the MMLIB format (Van Peteghem & Vanhoucke) and the
    classic PSPLIB format (Kolisch & Sprecher).
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines = [line.rstrip("\n") for line in lines]
    i = 0

    def skip_to(*markers: str) -> int:
        nonlocal i
        while i < len(lines):
            for m in markers:
                if m in lines[i]:
                    return i
            i += 1
        raise ValueError(f"Could not find any of {markers} in {filepath}")

    # Parse num_activities
    skip_to("jobs", "Jobs")
    num_activities = int(re.search(r":\s*(\d+)", lines[i]).group(1))

    # Parse resource counts
    skip_to("- renewable")
    num_renewable = int(re.search(r":\s*(\d+)", lines[i]).group(1))
    i += 1
    skip_to("- nonrenewable")
    num_nonrenewable = int(re.search(r":\s*(\d+)", lines[i]).group(1))

    # Parse precedence relations
    skip_to("PRECEDENCE RELATIONS")
    i += 1  # skip header
    # Skip dashed lines
    while i < len(lines) and (lines[i].startswith("jobnr") or lines[i].startswith("-")):
        i += 1

    activities = [None] * num_activities
    for _ in range(num_activities):
        while i < len(lines) and not lines[i].strip():
            i += 1
        parts = lines[i].split()
        job_id = int(parts[0]) - 1  # 0-based
        # parts[1] = num_modes, parts[2] = num_successors
        num_succs = int(parts[2])
        successors = [int(s) - 1 for s in parts[3:3 + num_succs]]
        activities[job_id] = Activity(id=job_id, modes=[], successors=successors)
        i += 1

    # Parse requests/durations
    skip_to("REQUESTS/DURATIONS")
    i += 1
    # Skip header lines (jobnr, dashes, etc.)
    while i < len(lines) and (
        lines[i].strip().startswith("jobnr") or
        lines[i].strip().startswith("-") or
        not lines[i].strip()
    ):
        i += 1

    current_job = None
    while i < len(lines) and not lines[i].strip().startswith("*"):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()

        # Determine if this line starts a new job or continues the current one
        # New job lines have the job number as the first field
        # Continuation lines for additional modes may omit the job number
        # or the line may start with whitespace
        if lines[i][0] not in (' ', '\t'):
            # New job
            current_job = int(parts[0]) - 1
            mode_nr = int(parts[1])
            duration = int(parts[2])
            demands = [int(x) for x in parts[3:]]
        else:
            # Continuation of current job (additional mode)
            mode_nr = int(parts[0])
            duration = int(parts[1])
            demands = [int(x) for x in parts[2:]]

        renewable_demands = demands[:num_renewable]
        nonrenewable_demands = demands[num_renewable:num_renewable + num_nonrenewable]
        activities[current_job].modes.append(
            Mode(duration=duration,
                 renewable_demands=renewable_demands,
                 nonrenewable_demands=nonrenewable_demands)
        )
        i += 1

    # Parse resource availabilities
    skip_to("RESOURCE AVAILABILITIES", "RESOURCEAVAILABILITIES")
    i += 1
    # Skip header line(s)
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and re.match(r'\s*[RN]', lines[i]):
        i += 1  # skip "R 1  R 2  N 1  N 2" header
    caps = [int(x) for x in lines[i].split()]
    renewable_capacities = caps[:num_renewable]
    nonrenewable_capacities = caps[num_renewable:num_renewable + num_nonrenewable]

    return Project(
        num_activities=num_activities,
        num_renewable=num_renewable,
        num_nonrenewable=num_nonrenewable,
        renewable_capacities=renewable_capacities,
        nonrenewable_capacities=nonrenewable_capacities,
        activities=activities,
    )
