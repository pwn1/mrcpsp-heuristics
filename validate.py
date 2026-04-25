from __future__ import annotations
"""Schedule validation for MRCPSP."""

from mrcpsp import Project, Schedule


def validate_schedule(project: Project, schedule: Schedule) -> list[str]:
    """Validate a schedule against all MRCPSP constraints.

    Returns a list of violation descriptions (empty = valid).
    """
    errors = []
    n = project.num_activities

    # Check mode assignments are valid indices
    for i in range(n):
        if schedule.mode_assignments[i] < 0 or schedule.mode_assignments[i] >= len(project.activities[i].modes):
            errors.append(f"Activity {i}: invalid mode {schedule.mode_assignments[i]}")

    if errors:
        return errors

    # Check precedence constraints
    for act in project.activities:
        for s in act.successors:
            mode = act.modes[schedule.mode_assignments[act.id]]
            finish = schedule.start_times[act.id] + mode.duration
            if finish > schedule.start_times[s]:
                errors.append(
                    f"Precedence violation: activity {act.id} finishes at {finish} "
                    f"but successor {s} starts at {schedule.start_times[s]}"
                )

    # Check renewable resource constraints
    makespan = schedule.compute_makespan(project)
    for t in range(makespan):
        usage = [0] * project.num_renewable
        for i in range(n):
            mode = project.activities[i].modes[schedule.mode_assignments[i]]
            st = schedule.start_times[i]
            if st <= t < st + mode.duration:
                for r in range(project.num_renewable):
                    usage[r] += mode.renewable_demands[r]
        for r in range(project.num_renewable):
            if usage[r] > project.renewable_capacities[r]:
                errors.append(
                    f"Renewable resource {r} exceeded at time {t}: "
                    f"usage {usage[r]} > capacity {project.renewable_capacities[r]}"
                )

    # Check non-renewable resource constraints
    for nr in range(project.num_nonrenewable):
        total = sum(
            project.activities[i].modes[schedule.mode_assignments[i]].nonrenewable_demands[nr]
            for i in range(n)
        )
        if total > project.nonrenewable_capacities[nr]:
            errors.append(
                f"Non-renewable resource {nr} exceeded: "
                f"total {total} > capacity {project.nonrenewable_capacities[nr]}"
            )

    return errors
