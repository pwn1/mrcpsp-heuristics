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

import random
from mrcpsp import Project

# ---------------------------------------------------------------------------
# Composite rule builder
# ---------------------------------------------------------------------------

class CompositeRule:
    def __init__(
            self,
            primary_heuristic:PriorityHeuristic,
            tiebreak_heuristic:PriorityHeuristic
    ):
        self._primary_heuristic = primary_heuristic
        self._tiebreak_heuristic = tiebreak_heuristic

    def return_composite_func(self):
        def composite(project: Project, mode_assignments: list[int]) -> list:
            return list(zip(
                self._primary_heuristic.prioritise(project,mode_assignments),
                self._tiebreak_heuristic.prioritise(project,mode_assignments)
            ))
        return composite


# ---------------------------------------------------------------------------
# Build the full registry
# ---------------------------------------------------------------------------

HEURISTIC_LIST = [AN, GRD, GRPW, LFT, LST, LSTLFT, MSLK, MTS, NIS, RWK, SPT, WRUP]

# ---------------------------------------------------------------------------
# Random priority rule (placeholder + seeded factory)
# ---------------------------------------------------------------------------
PRIORITY_RULES = {}
for p in HEURISTIC_LIST:
    for t in HEURISTIC_LIST:
        if p==t:
            continue
        PRIORITY_RULES[f'{p.get_name()}/{t.get_name()}'] = CompositeRule(p,t).return_composite_func()

def make_random_priority(seed: int):
    """Create a random priority function with a local, deterministically seeded RNG.

    Returns one random float per activity on each call; lower = higher priority.
    """
    rng = random.Random(seed)

    def _random_priority(project: Project, **_) -> list:
        return [rng.random() for _ in range(project.num_activities)]

    return _random_priority

def get_priority_fn(pr_name: str, project, sgs_name: str, mr_name: str):
    """Get a priority function for the given combo.

    For stochastic rules (currently only 'random'), returns a closure with a
    local RNG seeded from both the instance data and the full heuristic
    combination name. Deterministic rules are returned as-is.
    """
    if pr_name.startswith("random"):
        from mode_rules import combo_seed
        seed = combo_seed(project.seed(), sgs_name, pr_name, mr_name)
        return make_random_priority(seed)
    return PRIORITY_RULES[pr_name]
