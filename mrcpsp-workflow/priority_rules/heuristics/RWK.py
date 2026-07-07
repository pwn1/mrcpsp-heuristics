from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic

class RWK(PriorityHeuristic):
    """ Lova, Tormos & Barber (2006) define Maximum Remaining Work (RWK)
    as "the sum of the [...] duration of the activity and all
    its [transitive] successors".

    In Lova, Tormos & Barber (2006), they compute this heuristic using
    the shortest duration mode for each activity. We however, use the
    current-mode durations. For context-aware assignment, this means the first
    pass will be the equivalent (as shortest modes are assumed). However, for
    the second pass/context-unaware assignment it allows us to be more
    precise with our calculations. This same approach is used in the GRPW
    heuristic.

    In this heuristic, a higher value corresponds to higher priority,
    therefore we have to negate the values to fit the lower-is-better
    convention."""
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        all_successors = PriorityHeuristic._compute_successors_recursive(project)

        durations = project.durations_given_modes(mode_assignments)

        return [
            -(durations[i] + sum(durations[s] for s in all_successors[i]))
            for i in range(project.num_activities)
        ]
