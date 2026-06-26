from typing import Callable

from mrcpsp import Project, Schedule, Activity
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
        state = self._set_scheduler_state(project, input_mode_assignments)

        num_scheduled = 0

        t = 0
        while num_scheduled < project.num_activities:
            eligible = [j for j in state.ready
                        if all(state.finish_times[p] <= t for p in state.predecessor_list[j])]
            eligible.sort(key=lambda j: (priorities[j], j))

            scheduled_any = False
            for act_id in eligible:
                if mode_fn is not None:
                    state.mode_assignments[act_id] = mode_fn(
                        activity=project.activities[act_id], project=project,
                        resource_profile=state.profile, earliest_possible=t,
                    )
                mode = project.activities[act_id].modes[state.mode_assignments[act_id]]
                if not self._fits_renewable(
                        state.profile,
                        project.renewable_capacities,
                        t,
                        mode.duration,
                        mode.renewable_demands
                ):
                    continue

                state.start_times[act_id] = t
                state.finish_times[act_id] = t + mode.duration
                self._update_resource_profile(state.profile, t, mode.duration, mode.renewable_demands)
                state.ready.remove(act_id)

                for s in project.activities[act_id].successors:
                    state.remaining[s] -= 1
                    if state.remaining[s] == 0:
                        state.ready.append(s)
                num_scheduled += 1
                scheduled_any = True

            if not scheduled_any:
                t += 1
                if t >= state.horizon:
                    raise AssertionError(
                        f"Project {project} failed to schedule in ParallelScheduler, "
                        f"by producing a schedule with length longer than the horizon."
                    )

        return Schedule(
            mode_assignments=list(state.mode_assignments),
            start_times=state.start_times,
            project=project
        )
