from mrcpsp import Project

from priority_rules.priority_heuristic_abc import PriorityHeuristic
from priority_rules.critical_path import CriticalPathMethodCalculator

class LFT(PriorityHeuristic):
    """ Lastest Finish Time: Calculated by completing a CPM (critical
    path method) backwards pass.

    Described in Davis and Patterson (1975) as being calculated by
    "usual critical path methods", it was found by Lova, Tormos & Barber
    (2006) to be in the top 3 best performing heuristics out of 14
    heuristics on both serial and parallel schedule generation schemes. """
    @staticmethod
    def get_name() -> str:
        return "LFT"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments).latest_finish_time