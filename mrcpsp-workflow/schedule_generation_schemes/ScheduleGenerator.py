from dataclasses import dataclass

from mrcpsp import Project, Schedule
from schedule_generation_schemes.InitialModeAssigner import InitialModeAssigner


@dataclass
class ModeAssignmentScore:
    nonrenewable_score: int
    duration_score: int

    def __lt__(self, other: "ModeAssignmentScore") -> bool:
        return (self.nonrenewable_score, self.duration_score) < (other.nonrenewable_score, other.duration_score)

class ScheduleGenerator:
    def __init__(self, core, priority_fn, mode_fn, initial_mode_assigner : InitialModeAssigner):
        self.core = core
        self.priority_fn = priority_fn
        self.mode_fn = mode_fn
        self.initial_mode_assigner = initial_mode_assigner

    def run(self, project: Project) -> Schedule | None:
        mode_assignments = (self.initial_mode_assigner
                            .assign_modes(project, self.priority_fn, self.mode_fn, self.core))

        repaired_mode_assignments = self._repair_nonrenewable(project, mode_assignments)
        if repaired_mode_assignments=="no solution found":
            return None
        mode_assignments = repaired_mode_assignments

        priorities = self.priority_fn(project=project, mode_assignments=mode_assignments)
        return self.core(project, priorities, mode_assignments)




    def _repair_nonrenewable(self, project: Project, mode_assignments: list[int]) -> list[int]|str:
        """ Uses a greedy local hill climbing search to try and statisfy non-renewable
        constraints.

        At each step, it picks a mode switch which give the best total non-renewable reduction,
        with duration increase as tie-breaker.

        Returns repaired mode-assignments is feasible solution is found, and the string
        'infeasible' if no solution is found (either because we got stuck in a local optima,
        or hit the iteration cap).
        """

        iteration_limit = project.num_activities * 10

        current_mode_assignment = mode_assignments.copy()

        for _ in range(iteration_limit):
            current_score = self._get_mode_assignment_score(project, current_mode_assignment)

            if current_score[0] == 0: return current_mode_assignment

            new_best_mode_assignment = current_mode_assignment
            best_mode_assignment_score = current_score

            neighbourhood = self._generate_neighbourhood(project, current_mode_assignment)
            for neighbour in neighbourhood:
                neighbour_score = self._get_mode_assignment_score(project, neighbour)

                if neighbour_score < best_mode_assignment_score:
                    new_best_mode_assignment = neighbour
                    best_mode_assignment_score = neighbour_score

            if new_best_mode_assignment == current_mode_assignment:
                return "no solution found"
            current_mode_assignment = new_best_mode_assignment

        return "no solution found" # Iteration cap has been hit

    @staticmethod
    def _generate_neighbourhood(project:Project, current_mode_assignment: list[int]) -> list[list[int]]:
        neighbourhood = []
        for activity_index in range(project.num_activities):
            activity = project.activities[activity_index]
            for mode_index, mode in enumerate(activity.modes):
                new_assignment = current_mode_assignment.copy()
                new_assignment[activity_index] = mode_index
                neighbourhood.append(new_assignment)

        return neighbourhood

    @staticmethod
    def _get_mode_assignment_score(project: Project, mode_assignments: list[int])-> tuple[int, int]:
        total_non_renewable_use = [
            sum(project.activities[i].modes[mode_assignments[i]].nonrenewable_demands[nr]
                for i in range(project.num_activities))
            for nr in range(project.num_nonrenewable)
        ]

        non_renewable_scores = [
            max(0, total_non_renewable_use[nr] - project.nonrenewable_capacities[nr])
            for nr in range(project.num_nonrenewable)
        ]

        duration_score = [
            project.activities[i].modes[mode_assignments[i]].duration
            for i in range(project.num_activities)
        ]

        return sum(non_renewable_scores),sum(duration_score)