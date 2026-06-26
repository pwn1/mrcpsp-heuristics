from typing import Callable

from schedule_generation_schemes.InitialModeAssigner import ContextAwareModeAssigner, \
    ContextUnAwareModeAssigner
from schedule_generation_schemes.ScheduleGenerator import ScheduleGenerator
from schedule_generation_schemes.schedulers import ParallelScheduler, SerialScheduler


class ScheduleGeneratorFactory:
    @staticmethod
    def create(
            parallel: bool,
            priority_fn: Callable,
            mode_fn: Callable,
            mode_is_context_aware: bool
    ) -> ScheduleGenerator:
        core_scheduler = ParallelScheduler() if parallel else SerialScheduler()

        initial_mode_assigner = ContextAwareModeAssigner() if mode_is_context_aware else ContextUnAwareModeAssigner()

        return ScheduleGenerator(core_scheduler, priority_fn, mode_fn, initial_mode_assigner)
