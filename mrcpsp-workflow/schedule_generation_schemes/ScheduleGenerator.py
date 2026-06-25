from typing import Callable

from mrcpsp import Project, Schedule
from schedule_generation_schemes.InitialModeAssigner import InitialModeAssigner
from schedule_generation_schemes.NonRenewableRepair import NonRenewableRepair
from schedule_generation_schemes.schedulers import Scheduler


class ScheduleGenerator:
    def __init__(
            self,
            core_scheduler: Scheduler,
            priority_fn: Callable,
            mode_fn: Callable,
            initial_mode_assigner: InitialModeAssigner
    ):
        self.core_scheduler = core_scheduler
        self.priority_fn = priority_fn
        self.mode_fn = mode_fn
        self.initial_mode_assigner = initial_mode_assigner

    def run(self, project: Project) -> Schedule | None:
        mode_assignments = (
            self
            .initial_mode_assigner
            .assign_modes(project, self.priority_fn, self.mode_fn, self.core_scheduler)
        )

        repaired_mode_assignments = NonRenewableRepair().repair_nonrenewable(project, mode_assignments)

        if repaired_mode_assignments is None: return None

        mode_assignments = repaired_mode_assignments

        priorities = self.priority_fn(project=project, mode_assignments=mode_assignments)
        return self.core_scheduler.fixed_mode_pass(project, priorities, mode_assignments)
