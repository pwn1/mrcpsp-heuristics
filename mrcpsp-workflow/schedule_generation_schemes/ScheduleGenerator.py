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
        """
        Returns schedule for the project.

        This is inspired from the general algorithm described by A. Lova, P. Tormos,
        and F. Barber. However, they describe a single pass approach, where a context
        aware mode rule (EFFT) is used to assign the modes with a single S-SGS/P-SGS
        pass.

        However, unlike A. Lova, P. Tormos, and F. Barber, we consider non-renewable
        resources as well. This means that we take a two pass approach. This first pass
        for context aware mode heuristics is the same as described. However, once
        this has been completed we take these modes and use a local search algorithm
        to try and repair any non-renewable resource violations. Once we have done this,
        we do another pass of the S-SGS/P-SGS, but set the modes to those found by this
        initial process. (Note: For context unaware mode heuristics, only a single pass
        is completed, as we set the modes statically at the beginning.)
        """

        mode_assignments = (
            self
            .initial_mode_assigner
            .assign_modes(project, self.priority_fn, self.mode_fn, self.core_scheduler)
        )

        mode_assignments = NonRenewableRepair().repair_nonrenewable(project, mode_assignments)

        # If the repair fails, we must exit early
        if mode_assignments is None: return None

        priorities = self.priority_fn(project=project, mode_assignments=mode_assignments)
        return self.core_scheduler.fixed_mode_pass(project, priorities, mode_assignments)
