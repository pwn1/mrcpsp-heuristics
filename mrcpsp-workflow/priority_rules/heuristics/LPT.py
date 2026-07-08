from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class LPT(PriorityHeuristic):
    """Longest Processing Time (SPT) is defined in Lova, Tormos & Barber (2006)
    in Table 1. Lova, Tormos & Barber (2006) take the shortest mode duration,
    whereas in this implementation we use the duration of the provided mode
    assignment.

    Because preference is given to the longest process, we have to negate the results.
    """
    @staticmethod
    def get_name() -> str:
        return "LPT"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return [
            -duration for duration in project.durations_given_modes(mode_assignments)
        ]