from __future__ import annotations
"""Data structures for the Multi-Mode Resource-Constrained Project Scheduling Problem."""

from dataclasses import dataclass


@dataclass
class Mode:
    duration: int
    renewable_demands: list[int]    # demand on each renewable resource
    nonrenewable_demands: list[int] # demand on each non-renewable resource


@dataclass
class Activity:
    id: int
    modes: list[Mode]
    successors: list[int]


@dataclass
class Project:
    num_activities: int  # including source and sink
    num_renewable: int
    num_nonrenewable: int
    renewable_capacities: list[int]
    nonrenewable_capacities: list[int]
    activities: list[Activity]  # indexed by activity id (0-based)

    @property
    def predecessors(self) -> list[list[int]]:
        """Compute predecessor lists from successor lists."""
        preds = [[] for _ in range(self.num_activities)]
        for act in self.activities:
            for succ in act.successors:
                preds[succ].append(act.id)
        return preds

    def seed(self) -> int:
        """Deterministic integer seed derived from the instance data
        (durations, renewable demands, non-renewable demands for every mode of
        every activity). Used to seed the global RNG before each heuristic run
        so stochastic rules are reproducible per instance.
        """
        parts = []
        for act in self.activities:
            for m in act.modes:
                parts.append(m.duration)
                parts.extend(m.renewable_demands)
                parts.extend(m.nonrenewable_demands)
        # hash() on a tuple of ints is stable across Python invocations
        # (unlike str/bytes hashing, which is salted by PYTHONHASHSEED).
        return hash(tuple(parts)) & 0xFFFFFFFF


@dataclass
class Schedule:
    """A complete schedule: mode assignment and start time for each activity."""
    mode_assignments: list[int]   # mode index for each activity
    start_times: list[int]        # start time for each activity

    def compute_makespan(self, project: Project) -> int:
        """Compute makespan as the maximum finish time across all activities."""
        max_finish = 0
        for act in project.activities:
            mode = act.modes[self.mode_assignments[act.id]]
            finish = self.start_times[act.id] + mode.duration
            if finish > max_finish:
                max_finish = finish
        return max_finish
