from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class GRD(PriorityHeuristic):
    """ Greatest resource demand (GRD) defined in Lova, Tormos & Barber (2006)
    Table 1. Calculated as the product of the activity duration and sum of
    renewable resource requirements.
    """
    @staticmethod
    def get_name() -> str:
        return "GRD"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        priorities = []
        for activity_index in range (project.num_activities):
            mode = project.activities[activity_index].modes[mode_assignments[activity_index]]
            priority = -(mode.duration*sum(mode.renewable_demands))
            priorities.append(priority)
        return priorities