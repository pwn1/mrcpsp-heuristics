from abc import ABC

from mrcpsp import Project


class AbstractScheduler(ABC):
    def _make_resource_profile(self,num_resources: int, horizon: int) -> list[list[int]]:
        return [[0] * horizon for _ in range(num_resources)]

    def _update_resource_profile(self, profile, start: int, duration: int, demands):
        for r, d in enumerate(demands):
            row = profile[r]
            for t in range(start, start + duration):
                row[t] += d

    def _compute_horizon(self, project: Project) -> int:
        return sum(max(m.duration for m in a.modes) for a in project.activities) + 1

    def _fits_renewable(self, profile, caps, t: int, duration: int, demands) -> bool:
        if duration == 0:
            return True
        for dt in range(duration):
            for r, d in enumerate(demands):
                if profile[r][t + dt] + d > caps[r]:
                    return False
        return True

    def _find_earliest_feasible_start(self, duration, demands, capacities, profile, earliest):
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
