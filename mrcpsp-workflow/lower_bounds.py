"""Sound lower bounds on the MRCPSP makespan.

Two cheap, classical bounds combined:

  LB0 (critical path): longest source-to-sink path in the precedence DAG
       using each activity's shortest-mode duration. Sound because every
       feasible schedule respects precedence and uses durations >= shortest.

  LB1 (resource workload, Brucker & Knust 2003): for each renewable r,
       ceil(sum_j min_m (d_{j,m} * k_{j,r,m}) / R_r). Sound because every
       feasible schedule consumes at least the minimum-energy mode per
       (activity, resource), and total renewable-time available within
       [0, T] is T * R_r.

Combined LB = max(LB0, max_r LB1_r). Non-renewables give no makespan
bound (they're total-budget constraints, not per-time-unit).
"""

from mrcpsp import Project
from time_window_pruning import _cpm


def critical_path_lb(project: Project) -> int:
    shortest = [min(m.duration for m in act.modes) for act in project.activities]
    es, _ = _cpm(project, shortest, horizon=0)
    sinks = [v for v in range(project.num_activities)
             if not project.activities[v].successors]
    return max(es[v] + shortest[v] for v in sinks)


def resource_workload_lb(project: Project) -> int:
    if project.num_renewable == 0:
        return 0
    best = 0
    for r in range(project.num_renewable):
        cap = project.renewable_capacities[r]
        if cap <= 0:
            assert all(m.renewable_demands[r] == 0
                       for act in project.activities for m in act.modes), \
                f"resource {r} has zero capacity but nonzero demand"
            continue
        work = sum(min(m.duration * m.renewable_demands[r] for m in act.modes)
                   for act in project.activities)
        bound = (work + cap - 1) // cap
        if bound > best:
            best = bound
    return best


def compute_lower_bound(project: Project) -> int:
    return max(critical_path_lb(project), resource_workload_lb(project))
