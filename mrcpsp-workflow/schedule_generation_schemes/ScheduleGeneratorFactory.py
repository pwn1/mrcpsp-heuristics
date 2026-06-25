from schedule_generation_schemes.InitialModeAssigner import ContextAwareModeAssigner, \
    ContextUnAwareModeAssigner
from schedule_generation_schemes.ScheduleGenerator import ScheduleGenerator


class ScheduleGeneratorFactory:
    @staticmethod
    def create(core, priority_fn, mode_fn, mode_is_context_aware):
        initial_mode_assigner = ContextAwareModeAssigner() if mode_is_context_aware else ContextUnAwareModeAssigner()
        return ScheduleGenerator(core, priority_fn, mode_fn, initial_mode_assigner)
