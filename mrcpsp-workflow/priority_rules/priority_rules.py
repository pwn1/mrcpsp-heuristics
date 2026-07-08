from __future__ import annotations

from .heuristics import *
from .priority_heuristic_abc import PriorityHeuristic

"""Priority rules for activity ordering in schedule generation schemes.

Priority rules surveyed in:
  - Hartmann & Kolisch (2000), Kolisch & Hartmann (2006)
  - Lova, Tormos & Barber (2006) for MRCPSP specifically
  - Melchiors, Kolisch & Kanet (2024)

Each base rule returns a list of numeric values (lower = higher priority).
Composite rules pair a primary rule with a tie-breaker, returning tuples.

All priority rules have been written in a static form, to limit complexity.

References:
-  A. Lova, P. Tormos, and F. Barber, ‘Multi-mode resource constrained project scheduling:
   Scheduling schemes, priority rules and mode selection rules’, Inteligencia Artificial. 
   Revista Iberoamericana de Inteligencia Artificial, vol. 10, no. 30, pp. 69–86, 2006.
   
-  E. W. Davis and J. H. Patterson, ‘A comparison of heuristic and optimum solutions in
   resource-constrainedproject scheduling’, Management science, vol. 21, 
   no. 8, pp. 944–955, 1975.
   
-  R. Kolisch, ‘Serial and parallel resource-constrained project scheduling methods 
   revisited: Theory and computation’, European Journal of Operational Research, 
   vol. 90, no. 2, pp. 320–333, 1996.
   
-  G. Ulusoy and L. Özdamar, ‘Heuristic Performance and Network/Resource Characteristics in
   Resource-Constrained Project Scheduling’, The Journal of the Operational Research Society,
   vol. 40, no. 12, pp. 1145–1152, 1989.
"""

from mrcpsp import Project

# ---------------------------------------------------------------------------
# Composite rule builder
# ---------------------------------------------------------------------------

class CompositeRule(PriorityHeuristic):
    def __init__(
            self,
            primary_heuristic:PriorityHeuristic,
            tiebreak_heuristic:PriorityHeuristic
    ):
        self._primary_heuristic = primary_heuristic
        self._tiebreak_heuristic = tiebreak_heuristic

    def get_name(self) -> str:
        return f"{self._primary_heuristic.get_name()}/{self._tiebreak_heuristic.get_name()}"

    def prioritise(self, project: Project, mode_assignments: list[int]) -> list[int]:
        return list(zip(
            self._primary_heuristic.prioritise(project, mode_assignments),
            self._tiebreak_heuristic.prioritise(project, mode_assignments)
        ))

# ---------------------------------------------------------------------------
# Build the full registry
# ---------------------------------------------------------------------------

HEURISTIC_LIST = [AN, EFT, EST, FREE, GRD, GRPW, LFT, LPT, LST, LSTLFT, MSLK, MTS, NIS, RWK, SPT, WRUP]

PRIORITY_RULES = {}
for p in HEURISTIC_LIST:
    for t in HEURISTIC_LIST:
        if p==t:
            continue
        PRIORITY_RULES[f'{p.get_name()}/{t.get_name()}'] = CompositeRule(p,t)