from __future__ import annotations

from typing import Callable

from mrcpsp import Project
from schedule_generation_schemes.ScheduleGeneratorFactory import ScheduleGeneratorFactory
from schedule_generation_schemes.schedulers import SerialScheduler, ParallelScheduler

"""Schedule Generation Schemes for multi-mode RCPSP.

References:
  - Kolisch (1996): Serial and parallel SGS (foundational, not on DBLP)
  - Sprecher & Drexl (1998): Multi-mode sequencing algorithm
  - Hartmann & Kolisch (2000), Kolisch & Hartmann (2006): experimental evaluation
  - Lova, Tormos & Barber (2006): SGS + priority rules + mode selection for MRCPSP
"""

# ---------------------------------------------------------------------------
# Public SGS entry points
# ---------------------------------------------------------------------------

def serial_sgs(
        project:Project,
        priority_fn:Callable,
        mode_fn:Callable,
        mode_is_context_aware:bool=False
):
    """Serial Schedule Generation Scheme. Schedules activities one at a time in
    priority order at their earliest feasible start. For context-aware mode
    rules, a two-pass approach is used (first pass selects modes, NR repair,
    second pass re-schedules)."""
    schedule_generator = ScheduleGeneratorFactory.create(
        core=SerialScheduler(),
        priority_fn=priority_fn,
        mode_fn=mode_fn,
        mode_is_context_aware=mode_is_context_aware
    )
    return schedule_generator.run(project)


def parallel_sgs(
        project:Project,
        priority_fn:Callable,
        mode_fn:Callable,
        mode_is_context_aware:bool=False
):
    """Parallel Schedule Generation Scheme. Advances time step by step,
    scheduling all eligible activities at each decision point. Two-pass for
    context-aware mode rules (same as serial_sgs)."""
    schedule_generator = ScheduleGeneratorFactory.create(
        core=ParallelScheduler(),
        priority_fn=priority_fn,
        mode_fn=mode_fn,
        mode_is_context_aware=mode_is_context_aware
    )
    return schedule_generator.run(project)



SGS_SCHEMES = {
    "serial": serial_sgs,
    "parallel": parallel_sgs,
}
