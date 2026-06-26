from typing import Callable

from mrcpsp import Project, Schedule
from schedule_generation_schemes.helpers import find_earliest_feasible_start
from schedule_generation_schemes.schedulers.Scheduler import Scheduler


class SerialScheduler(Scheduler):

    def context_aware_pass(
            self,
            project: Project,
            priorities: list[int|tuple[int]],
            mode_fn: Callable
    ) -> Schedule:
        mode_assignments = [0] * project.num_activities
        return self._run(project, priorities, mode_assignments, mode_fn=mode_fn)

    def fixed_mode_pass(
            self,
            project: Project,
            priorities: list[int|tuple[int]],
            mode_assignments: list[int]
    ) -> Schedule:
        return self._run(project, priorities, list(mode_assignments), mode_fn=None)

    def _run(
            self,
            project: Project,
            priorities: list[int|tuple[int]],
            input_mode_assignments: list[int],
            mode_fn: Callable | None
    ) -> Schedule:
        state = self._set_scheduler_state(project, input_mode_assignments)

        for _ in range(project.num_activities):
            act_id = min(state.ready, key=lambda j: (priorities[j], j))
            state.ready.remove(act_id)

            ep = max((state.finish_times[p] for p in state.predecessor_list[act_id]), default=0)

            if mode_fn is not None:
                state.mode_assignments[act_id] = mode_fn(
                    activity=project.activities[act_id], project=project,
                    resource_profile=state.profile, earliest_possible=ep,
                )
            mode = project.activities[act_id].modes[state.mode_assignments[act_id]]
            st = find_earliest_feasible_start(
                mode.duration, mode.renewable_demands,
                project.renewable_capacities, state.profile, ep,
            )
            state.start_times[act_id] = st
            state.finish_times[act_id] = st + mode.duration
            self._update_resource_profile(state.profile, st, mode.duration, mode.renewable_demands)

            for s in project.activities[act_id].successors:
                state.remaining[s] -= 1
                if state.remaining[s] == 0:
                    state.ready.append(s)

        return Schedule(mode_assignments=state.mode_assignments, start_times=state.start_times, project=project)
