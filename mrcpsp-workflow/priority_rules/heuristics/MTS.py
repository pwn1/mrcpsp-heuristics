from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class MTS(PriorityHeuristic):
    """ Kolisch (1996) defines most total successors (MTS) as the total
    number of transitive successors an activity has. This definition is
    found in Table 1, with attribution to Alvarez-Valdes and Tamarit (1989).

    More total successors is defined as better, however we negate this to fit
    with the lower=higher priority convention.
    """
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        all_successors = PriorityHeuristic._compute_successors_recursive(project)
        return [-len(s) for s in all_successors]