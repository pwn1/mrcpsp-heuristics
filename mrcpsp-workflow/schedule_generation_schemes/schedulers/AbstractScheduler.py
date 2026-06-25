from abc import ABC

from mrcpsp import Project


class AbstractScheduler(ABC):
    @staticmethod
    def _make_resource_profile(num_resources: int, horizon: int) -> list[list[int]]:
        return [[0] * horizon for _ in range(num_resources)]

    @staticmethod
    def _update_resource_profile(profile, start: int, duration: int, demands):
        for r, d in enumerate(demands):
            row = profile[r]
            for t in range(start, start + duration):
                row[t] += d

    @staticmethod
    def _compute_horizon(project: Project) -> int:
        return sum(max(m.duration for m in a.modes) for a in project.activities) + 1

    @staticmethod
    def _fits_renewable(profile, caps, t: int, duration: int, demands) -> bool:
        if duration == 0:
            return True
        for dt in range(duration):
            for r, d in enumerate(demands):
                if profile[r][t + dt] + d > caps[r]:
                    return False
        return True