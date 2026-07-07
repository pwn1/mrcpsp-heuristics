from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class NIS(PriorityHeuristic):
    """ Defined as number of immediate successors by Lova, Tormos & Barber (2006)
    Table 1.
    We negate the value to fit with lower=higher priority convention.
    """
    @staticmethod
    def get_name() -> str:
        return "MIS"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return [
            -len(project.activities[i].successors)
            for i in range(project.num_activities)
        ]
