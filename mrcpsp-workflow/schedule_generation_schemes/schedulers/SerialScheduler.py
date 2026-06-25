from typing import Callable

from mrcpsp import Project, Schedule
from schedule_generation_schemes.helpers import find_earliest_feasible_start
from schedule_generation_schemes.schedulers.Scheduler import Scheduler


class SerialScheduler(Scheduler):

    def context_aware_pass(
            self,
            project: Project,
            priorities: list[tuple[int]],
            mode_fn: Callable
    ) -> Schedule:
        mode_assignments = [0] * project.num_activities
        return self._run(project, priorities, mode_assignments, mode_fn=mode_fn)

    def fixed_mode_pass(
            self,
            project: Project,
            priorities: list[tuple[int]],
            mode_assignments: list[int]
    ) -> Schedule:
        return self._run(project, priorities, list(mode_assignments), mode_fn=None)

    def _run(
            self,
            project: Project,
            priorities: list[tuple[int]],
            input_mode_assignments: list[int],
            mode_fn: Callable | None
    ) -> Schedule:
        mode_assignments = input_mode_assignments.copy()

        project = project
        n = project.num_activities
        profile = self._make_resource_profile(project.num_renewable, self._compute_horizon(project))
        start_times = [0] * n
        finish_times = [0] * n
        preds = project.predecessors
        succs = [a.successors for a in project.activities]
        remaining = [len(p) for p in preds]
        ready = [j for j in range(n) if remaining[j] == 0]

        for _ in range(n):
            act_id = min(ready, key=lambda j: (priorities[j], j))
            ready.remove(act_id)

            ep = max((finish_times[p] for p in preds[act_id]), default=0)

            if mode_fn is not None:
                mode_assignments[act_id] = mode_fn(
                    activity=project.activities[act_id], project=project,
                    resource_profile=profile, earliest_possible=ep,
                )
            mode = project.activities[act_id].modes[mode_assignments[act_id]]
            st = find_earliest_feasible_start(
                mode.duration, mode.renewable_demands,
                project.renewable_capacities, profile, ep,
            )
            start_times[act_id] = st
            finish_times[act_id] = st + mode.duration
            self._update_resource_profile(profile, st, mode.duration, mode.renewable_demands)

            for s in succs[act_id]:
                remaining[s] -= 1
                if remaining[s] == 0:
                    ready.append(s)

        return Schedule(mode_assignments=mode_assignments, start_times=start_times, project=project)
