from abc import ABC, abstractmethod
from typing import Callable

from mrcpsp import Project, Schedule


class Scheduler(ABC):
    @staticmethod
    def _make_resource_profile(num_resources: int, horizon: int) -> list[list[int]]:
        return [[0] * horizon for _ in range(num_resources)]

    @staticmethod
    def _update_resource_profile(
            profile: list[list[int]],
            start: int,
            duration: int,
            demands: list
    ):
        for r, d in enumerate(demands):
            row = profile[r]
            for t in range(start, start + duration):
                row[t] += d

    @staticmethod
    def _compute_horizon(
            project: Project
    ) -> int:
        return sum(max(m.duration for m in a.modes) for a in project.activities) + 1

    @staticmethod
    def _fits_renewable(
            profile: list[list[int]],
            caps: list[int],
            t: int,
            duration: int,
            demands: list[int]
    ) -> bool:
        if duration == 0:
            return True
        for dt in range(duration):
            for r, d in enumerate(demands):
                if profile[r][t + dt] + d > caps[r]:
                    return False
        return True

    @abstractmethod
    def context_aware_pass(
            self,
            project: Project,
            priorities: list[tuple[int]],
            mode_fn: Callable
    ) -> Schedule:
        ...

    @abstractmethod
    def fixed_mode_pass(
            self,
            project: Project,
            priorities: list[tuple[int]],
            mode_assignments: list[int]
    ) -> Schedule:
        ...
