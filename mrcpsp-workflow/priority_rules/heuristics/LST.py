from mrcpsp import Project
from priority_rules.priority_heuristic_abc import PriorityHeuristic
from priority_rules.critical_path import CriticalPathMethodCalculator


class LST(PriorityHeuristic):
    """
    Described in Davis and Patterson (1975) as being calculated by critical path
    methods, it was found by Lova, Tormos & Barber (2006) to be in the top 3
    best performing heuristics out of 14 heuristics on both serial and parallel
    schedule generation schemes.

    It is of course related to LFT as LST_j = LFT_j - d_j (where d_j is the
    duration of task j).

    Note: As Davis and Patterson (1975) prove, this is equivalent to the MINSLACK
    (minimum slack) priority rule when using parallel schedule generation.
    """
    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments).latest_start_time