from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class GRPW(PriorityHeuristic):
    """Similar to RWK, greatest rank positional weight (GRPW) is defined
    as being the sum of duration of an activity and all its immediate
    successors by Lova, Tormos & Barber (2006). Similar to RWK, we
    calculate it using the actual mode assignments, and not the
    minimum mode assignments.

    Similar to RWK, we have to negate the result to fit the lower-is-better
    convention.
    """
    @staticmethod
    def get_name() -> str:
        return "GRPW"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        durations = project.durations_given_modes(mode_assignments)
        return [
            -(durations[i] + sum(durations[s] for s in project.activities[i].successors))
            for i in range(project.num_activities)
        ]