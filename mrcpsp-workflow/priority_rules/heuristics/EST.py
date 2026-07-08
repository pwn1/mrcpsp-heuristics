from mrcpsp import Project

from priority_rules.priority_heuristic_abc import PriorityHeuristic
from priority_rules.critical_path import CriticalPathMethodCalculator

class EST(PriorityHeuristic):
    """ Earliest Start Time: Calculated by completing a CPM (critical
    path method) analysis. Tabulated by Lova, Tormos & Barber
    (2006) as one of the heuristics considered."""
    @staticmethod
    def get_name() -> str:
        return "EST"

    @staticmethod
    def prioritise(project: Project, mode_assignments: list[int]) -> list[int]:
        return CriticalPathMethodCalculator.get_cpm_schedule(project, mode_assignments).earliest_start_time