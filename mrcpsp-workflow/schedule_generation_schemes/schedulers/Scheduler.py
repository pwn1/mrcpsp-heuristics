from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from mrcpsp import Project, Schedule

@dataclass
class SchedulerState:
    mode_assignments : list[int]
    horizon : int
    profile : list[list[int]]
    start_times : list[int]
    finish_times : list[int]
    predecessor_list : list[list[int]]
    remaining : list[int]
    ready : list[int]

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

    def _set_scheduler_state(
            self,
            project: Project,
            input_mode_assignments: list[int],
    ):
        mode_assignments = input_mode_assignments.copy()
        horizon = self._compute_horizon(project)
        profile = self._make_resource_profile(project.num_renewable, horizon)
        start_times = [0] * project.num_activities
        finish_times = [0] * project.num_activities
        predecessor_list = project.predecessors
        remaining = [len(p) for p in predecessor_list]
        ready = [j for j in range(project.num_activities) if remaining[j] == 0]

        return SchedulerState(
            mode_assignments=mode_assignments,
            horizon=horizon,
            profile=profile,
            start_times=start_times,
            finish_times=finish_times,
            predecessor_list=predecessor_list,
            remaining=remaining,
            ready=ready,
        )

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
