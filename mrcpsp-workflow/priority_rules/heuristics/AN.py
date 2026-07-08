from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class AN(PriorityHeuristic):
    """ Uses the activity number as the priority of the task, defined in Lova,
    Tormos & Barber (2006) Table 1.
    """
    @staticmethod
    def get_name() -> str:
        return "AN"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return list(range(project.num_activities))
