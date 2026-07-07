from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic
from priority_rules.critical_path import CriticalPathMethodCalculator

class LSTLFT(PriorityHeuristic):
    """ Lova, Tormos & Barber (2006) define LSTLFT_j = LFT_j + LST_j
    (the sum of latest start time and latest finish time). They found
    it to have the best performance out of 14 heuristics with both serial and
    parallel schedule generation schemes."""
    @staticmethod
    def get_name() -> str:
        return "LSTLFT"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        cpm_schedule = CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments)
        return [
            cpm_schedule.latest_start_time[i] + cpm_schedule.latest_finish_time[i]
            for i in range(len(cpm_schedule.latest_start_time))
        ]
