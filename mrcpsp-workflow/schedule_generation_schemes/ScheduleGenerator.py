from mrcpsp import Project, Schedule
from schedule_generation_schemes.InitialModeAssigner import InitialModeAssigner


class ScheduleGenerator:
    def __init__(self, core, priority_fn, mode_fn, initial_mode_assigner : InitialModeAssigner):
        self.core = core
        self.priority_fn = priority_fn
        self.mode_fn = mode_fn
        self.initial_mode_assigner = initial_mode_assigner

    def run(self, project: Project) -> Schedule | None:
        mode_assignments = (self.initial_mode_assigner
                            .assign_modes(project, self.priority_fn, self.mode_fn, self.core))

        if not self._repair_nonrenewable(project, mode_assignments):
            return None
        priorities = self.priority_fn(project=project, mode_assignments=mode_assignments)
        return self.core(project, priorities, mode_assignments)

    def _repair_nonrenewable(self, project: Project, mode_assignments: list[int]) -> bool:
        """Greedily repair mode assignments to satisfy non-renewable constraints.

        At each step, picks the mode switch that gives the best total NR reduction,
        with duration increase as tie-breaker. Returns True if feasible.
        """
        n = project.num_activities
        nnr = project.num_nonrenewable
        caps = project.nonrenewable_capacities

        totals = [
            sum(project.activities[i].modes[mode_assignments[i]].nonrenewable_demands[nr]
                for i in range(n))
            for nr in range(nnr)
        ]

        for _ in range(n * 10):
            excesses = [max(0, totals[nr] - caps[nr]) for nr in range(nnr)]
            if sum(excesses) == 0:
                return True

            best = (0, float("inf"), -1, -1)  # (reduction, dur_cost, act, mode)
            for i in range(n):
                act = project.activities[i]
                if len(act.modes) <= 1:
                    continue
                cur_mode = mode_assignments[i]
                cur_dur = act.modes[cur_mode].duration
                cur_nr = act.modes[cur_mode].nonrenewable_demands

                for m, new_mode in enumerate(act.modes):
                    if m == cur_mode:
                        continue
                    new_nr = new_mode.nonrenewable_demands
                    reduction = 0
                    for nr in range(nnr):
                        diff = cur_nr[nr] - new_nr[nr]
                        new_ex = max(0, totals[nr] - diff - caps[nr])
                        reduction += excesses[nr] - new_ex
                    dur_cost = new_mode.duration - cur_dur
                    if reduction > best[0] or (reduction == best[0] and reduction > 0
                                               and dur_cost < best[1]):
                        best = (reduction, dur_cost, i, m)

            if best[2] == -1 or best[0] <= 0:
                return False
            i, m = best[2], best[3]
            old_nr = project.activities[i].modes[mode_assignments[i]].nonrenewable_demands
            new_nr = project.activities[i].modes[m].nonrenewable_demands
            for nr in range(nnr):
                totals[nr] += new_nr[nr] - old_nr[nr]
            mode_assignments[i] = m

        return self._check_nonrenewable_feasibility(project, mode_assignments)

    def _check_nonrenewable_feasibility(self, project, mode_assignments) -> bool:
        for nr in range(project.num_nonrenewable):
            total = sum(
                project.activities[i].modes[mode_assignments[i]].nonrenewable_demands[nr]
                for i in range(project.num_activities)
            )
            if total > project.nonrenewable_capacities[nr]:
                return False
        return True