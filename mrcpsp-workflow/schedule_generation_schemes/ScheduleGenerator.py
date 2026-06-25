from dataclasses import dataclass

from mrcpsp import Project, Schedule
from schedule_generation_schemes.InitialModeAssigner import InitialModeAssigner
from schedule_generation_schemes.NonRenewableRepair import NonRenewableRepair


@dataclass
class ModeAssignmentScore:
    nonrenewable_score: int
    duration_score: int

    def __lt__(self, other: "ModeAssignmentScore") -> bool:
        return (self.nonrenewable_score, self.duration_score) < (other.nonrenewable_score, other.duration_score)

class ScheduleGenerator:
    def __init__(self, core, priority_fn, mode_fn, initial_mode_assigner : InitialModeAssigner):
        self.core = core
        self.priority_fn = priority_fn
        self.mode_fn = mode_fn
        self.initial_mode_assigner = initial_mode_assigner

    def run(self, project: Project) -> Schedule | None:
        mode_assignments = (
            self
            .initial_mode_assigner
            .assign_modes(project, self.priority_fn, self.mode_fn, self.core)
        )

        repaired_mode_assignments = NonRenewableRepair().repair_nonrenewable(project, mode_assignments)

        if repaired_mode_assignments is None: return None

        mode_assignments = repaired_mode_assignments

        priorities = self.priority_fn(project=project, mode_assignments=mode_assignments)
        return self.core.fixed_mode_pass(project, priorities, mode_assignments)
