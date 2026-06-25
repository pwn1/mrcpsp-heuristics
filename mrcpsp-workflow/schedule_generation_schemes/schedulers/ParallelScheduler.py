from typing import Callable

from mrcpsp import Project, Schedule
from schedule_generation_schemes.schedulers.Scheduler import Scheduler


class ParallelScheduler(Scheduler):

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
        n = project.num_activities
        horizon = self._compute_horizon(project)
        profile = self._make_resource_profile(project.num_renewable, horizon)
        start_times = [0] * n
        finish_times = [0] * n
        preds = project.predecessors
        succs = [a.successors for a in project.activities]
        remaining = [len(p) for p in preds]
        # An activity is eligible at time t iff all preds finished by t. We track
        # `pending`: activities with remaining=0 but possibly not yet reached by t.
        pending = [j for j in range(n) if remaining[j] == 0]
        num_scheduled = 0
        caps = project.renewable_capacities

        t = 0
        while num_scheduled < n:
            eligible = [j for j in pending
                        if all(finish_times[p] <= t for p in preds[j])]
            eligible.sort(key=lambda j: (priorities[j], j))

            scheduled_any = False
            for act_id in eligible:
                if mode_fn is not None:
                    mode_assignments[act_id] = mode_fn(
                        activity=project.activities[act_id], project=project,
                        resource_profile=profile, earliest_possible=t,
                    )
                mode = project.activities[act_id].modes[mode_assignments[act_id]]
                if not self._fits_renewable(profile, caps, t, mode.duration, mode.renewable_demands):
                    continue

                start_times[act_id] = t
                finish_times[act_id] = t + mode.duration
                self._update_resource_profile(profile, t, mode.duration, mode.renewable_demands)
                pending.remove(act_id)
                for s in succs[act_id]:
                    remaining[s] -= 1
                    if remaining[s] == 0:
                        pending.append(s)
                num_scheduled += 1
                scheduled_any = True

            if not scheduled_any:
                t += 1
                if t >= horizon:
                    return None

        return Schedule(mode_assignments=list(mode_assignments), start_times=start_times, project=project)
