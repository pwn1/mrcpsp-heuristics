from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class SPT(PriorityHeuristic):
    """Shortest Processing Time (SPT) is defined in Lova, Tormos & Barber (2006)
    in Table 1. Lova, Tormos & Barber (2006) take the shortest mode duration,
    whereas in this implementation we use the duration of the provided mode
    assignment."""
    @staticmethod
    def get_name() -> str:
        return "SPT"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return project.durations_given_modes(mode_assignments)