from typing import Protocol, Callable

from mrcpsp import Project
from priority_rules import PriorityHeuristic
from schedule_generation_schemes.schedulers import Scheduler


class InitialModeAssigner(Protocol):
    @staticmethod
    def assign_modes(
            project: Project,
            priority_heuristic: PriorityHeuristic,
            mode_fn: Callable,
            core_scheduler: Scheduler
    ) -> list[int]: ...


class ContextAwareModeAssigner:
    @staticmethod
    def assign_modes(
            project: Project,
            priority_heuristic: PriorityHeuristic,
            mode_fn: Callable,
            core_scheduler: Scheduler
    ) -> list[int]:
        proxy_modes = [min(range(len(a.modes)), key=lambda m: a.modes[m].duration)
                       for a in project.activities]
        priorities = priority_heuristic.prioritise(project=project, mode_assignments=proxy_modes)
        schedule = core_scheduler.context_aware_pass(project, priorities, mode_fn=mode_fn)
        return schedule.mode_assignments


class ContextUnAwareModeAssigner:
    @staticmethod
    def assign_modes(
            project: Project,
            priority_heuristic: PriorityHeuristic,
            mode_fn: Callable,
            core_scheduler: Scheduler
    ) -> list[int]:
        return [
            mode_fn(activity=project.activities[i]) for i in range(project.num_activities)
        ]
