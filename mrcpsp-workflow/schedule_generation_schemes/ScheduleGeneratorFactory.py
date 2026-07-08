from typing import Callable

from priority_rules import PriorityHeuristic
from schedule_generation_schemes.InitialModeAssigner import ContextAwareModeAssigner, \
    ContextUnAwareModeAssigner
from schedule_generation_schemes.ScheduleGenerator import ScheduleGenerator
from schedule_generation_schemes.schedulers import ParallelScheduler, SerialScheduler


class ScheduleGeneratorFactory:
    @staticmethod
    def create(
            parallel: bool,
            priority_heuristic: PriorityHeuristic,
            mode_fn: Callable,
            mode_is_context_aware: bool
    ) -> ScheduleGenerator:
        core_scheduler = ParallelScheduler() if parallel else SerialScheduler()

        initial_mode_assigner = ContextAwareModeAssigner() if mode_is_context_aware else ContextUnAwareModeAssigner()

        return ScheduleGenerator(core_scheduler, priority_heuristic, mode_fn, initial_mode_assigner)
