from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic
from priority_rules.critical_path import CriticalPathMethodCalculator


class FREE(PriorityHeuristic):
    """ Minimum free slack (FREE) is defined by Lova, Tormos & Barber (2006),
    and they found it to place in the top 5 (out of 14) heuristics for both
    serial and parallel schedule generation schemes.
    """
    @staticmethod
    def get_name() -> str:
        return "FREE"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        cpm = CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments)

        priorities = []
        for activity_index in range(project.num_activities):
            successor_indexes = project.activities[activity_index].successors
            earliest_successor_est = min(
                [cpm.earliest_start_time[i]
                for i in successor_indexes]
                # We implement this dummy value for the sink task
                or [cpm.earliest_finish_time[activity_index]]
            )
            priorities.append(earliest_successor_est-cpm.earliest_finish_time[activity_index])

        return priorities