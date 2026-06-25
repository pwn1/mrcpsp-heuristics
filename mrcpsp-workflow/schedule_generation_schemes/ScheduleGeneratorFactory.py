from schedule_generation_schemes.ScheduleGenerator import ScheduleGenerator


class ScheduleGeneratorFactory:
    @staticmethod
    def create(core, priority_fn, mode_fn, mode_is_context_aware):
        schedule_generator = ScheduleGenerator(core,priority_fn,mode_fn, mode_is_context_aware)
        return schedule_generator