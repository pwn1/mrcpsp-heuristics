from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic
from priority_rules.critical_path import CriticalPathMethodCalculator


class MSLK(PriorityHeuristic):
    """ Davis & Patterson (1975) define minimum job slack (MSLK) as "the difference
    between the critical path analysis-determined Late Start Time (LST) and Early Start
    Time (EST)." It is one of the 14 heuristics Lova, Tormos & Barber (2006) consider,
    and produces the 6th best result on average out of 14 heuristics for both serial and
    parallel schedule generation schemes.
    """
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return (
            CriticalPathMethodCalculator
                .get_cpm_schedule(project, mode_assignments)
                .slack
        )