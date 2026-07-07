from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class WRUP(PriorityHeuristic):
    """Weighted Resource Utilisation Ratio and Precedence (WRUP) is tabulated in
    Kolisch (1996) Table 1, and presented in Ulusoy & Özdamar (1989).

    It is calculated by taking the weighted sum of the number of immediate successor
    activities, and the resource demand ratio.

    Similarly to previous heuristics, we negate the value to fit with lower=higher
    priority convention.
    """

    PRECEDENCE_WEIGHT = 0.7
    RESOURCE_UTILIZATION_WEIGHT = 1 - PRECEDENCE_WEIGHT
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        priorities = []
        for activity_index in range (project.num_activities):
            mode = project.activities[activity_index].modes[mode_assignments[activity_index]]
            num_immediate_successors = len(project.activities[activity_index].successors)
            resource_demand_ratio = sum(
                demand/total
                for demand, total in zip(mode.renewable_demands, project.renewable_capacities)
                if total>0
            )
            priority_value = (
                    WRUP.PRECEDENCE_WEIGHT * num_immediate_successors +
                    WRUP.RESOURCE_UTILIZATION_WEIGHT * resource_demand_ratio
            )
            priorities.append(-priority_value)
        return priorities