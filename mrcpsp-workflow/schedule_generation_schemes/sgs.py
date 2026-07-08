from __future__ import annotations

from typing import Callable

from mrcpsp import Project
from priority_rules import PriorityHeuristic
from schedule_generation_schemes.ScheduleGeneratorFactory import ScheduleGeneratorFactory

"""Schedule Generation Schemes for multi-mode RCPSP.

References:
  - R. Kolisch, ‘Serial and parallel resource-constrained project 
  scheduling methods revisited: Theory and computation’, European 
  Journal of Operational Research, vol. 90, no. 2, pp. 320–333, 1996.
  
  Details pseudocode for parallel and serial scheme generation for RCPSP 
  (which have been adapted for MRCPSP).
  
  - A. Lova, P. Tormos, and F. Barber, ‘Multi-mode resource constrained project scheduling:
   Scheduling schemes, priority rules and mode selection rules’, Inteligencia Artificial. 
   Revista Iberoamericana de Inteligencia Artificial, vol. 10, no. 30, pp. 69–86, 2006.
   
   Details serial and parallel scheme generation for MRCPSP, and integration
   of these schemes with priority and mode selection rules.
"""


# ---------------------------------------------------------------------------
# Public SGS entry points
# ---------------------------------------------------------------------------

def serial_sgs(
        project: Project,
        priority_heuristic: PriorityHeuristic,
        mode_fn: Callable,
        mode_is_context_aware: bool = False
):
    """Serial Schedule Generation Scheme. Schedules activities one at a time in
    priority order at their earliest feasible start. For context-aware mode
    rules, a two-pass approach is used (first pass selects modes, NR repair,
    second pass re-schedules)."""
    schedule_generator = ScheduleGeneratorFactory.create(
        parallel=False,
        priority_heuristic=priority_heuristic,
        mode_fn=mode_fn,
        mode_is_context_aware=mode_is_context_aware
    )
    return schedule_generator.run(project)


def parallel_sgs(
        project: Project,
        priority_heuristic: PriorityHeuristic,
        mode_fn: Callable,
        mode_is_context_aware: bool = False
):
    """Parallel Schedule Generation Scheme. Advances time step by step,
    scheduling all eligible activities at each decision point. Two-pass for
    context-aware mode rules (same as serial_sgs)."""
    schedule_generator = ScheduleGeneratorFactory.create(
        parallel=True,
        priority_heuristic=priority_heuristic,
        mode_fn=mode_fn,
        mode_is_context_aware=mode_is_context_aware
    )
    return schedule_generator.run(project)


SGS_SCHEMES = {
    "serial": serial_sgs,
    "parallel": parallel_sgs,
}
