"""Time-window mode pruning conditional on a known upper bound.

For each activity j, compute ES(j) and LF(j) by CPM using the shortest mode
duration of every activity, with horizon = UB. Both bounds are sound:
any feasible schedule with makespan <= UB has actual ES(j) >= ES_short(j)
and actual LF(j) <= LF_short(j). A mode m of j is therefore infeasible if
its duration exceeds the slack window LF_short(j) - ES_short(j).
"""

from mrcpsp import Project


def _topo_order(project: Project) -> list[int]:
    n = project.num_activities
    indeg = [0] * n
    for act in project.activities:
        for s in act.successors:
            indeg[s] += 1
    queue = [j for j in range(n) if indeg[j] == 0]
    order = []
    while queue:
        j = queue.pop()
        order.append(j)
        for s in project.activities[j].successors:
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)
    return order


def _cpm(project: Project, durations: list[int], horizon: int):
    """Forward + backward CPM with given per-activity durations and horizon.
    Returns (ES, LF) lists indexed by activity."""
    n = project.num_activities
    order = _topo_order(project)

    es = [0] * n
    for j in order:
        for s in project.activities[j].successors:
            cand = es[j] + durations[j]
            if cand > es[s]:
                es[s] = cand

    lf = [horizon] * n
    for j in reversed(order):
        for s in project.activities[j].successors:
            cand = lf[s] - durations[s]
            if cand < lf[j]:
                lf[j] = cand
    return es, lf


def top_k_longest_paths(project: Project, durations: list[int], k: int):
    """Return up to k longest source-to-sink paths in the precedence DAG,
    where path length is the sum of durations along the path. Each result
    entry is (length, [activity_ids in execution order]). Sorted by length
    descending. Activity ids are 0-based."""
    n = project.num_activities
    preds: list[list[int]] = [[] for _ in range(n)]
    for j, act in enumerate(project.activities):
        for s in act.successors:
            preds[s].append(j)

    paths: list[list[tuple[int, list[int]]]] = [[] for _ in range(n)]
    for v in _topo_order(project):
        if not preds[v]:
            paths[v] = [(durations[v], [v])]
        else:
            cands: list[tuple[int, list[int]]] = []
            for p in preds[v]:
                for length, path in paths[p]:
                    cands.append((length + durations[v], path + [v]))
            cands.sort(key=lambda x: -x[0])
            paths[v] = cands[:k]

    sinks = [v for v in range(n) if not project.activities[v].successors]
    merged = [pl for v in sinks for pl in paths[v]]
    merged.sort(key=lambda x: -x[0])
    return merged[:k]


def time_window_prunable(project: Project, ub: int) -> list[list[int]]:
    """For each activity, return mode indices whose duration exceeds the
    CPM slack window LF(j) - ES(j) computed with shortest-mode durations
    and horizon = ub."""
    shortest = [min(m.duration for m in act.modes) for act in project.activities]
    es, lf = _cpm(project, shortest, ub)
    out: list[list[int]] = []
    for j, act in enumerate(project.activities):
        window = lf[j] - es[j]
        out.append([mi for mi, m in enumerate(act.modes) if m.duration > window])
    return out
