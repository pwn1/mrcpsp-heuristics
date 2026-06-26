def find_earliest_feasible_start(duration, demands, capacities, profile, earliest):
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