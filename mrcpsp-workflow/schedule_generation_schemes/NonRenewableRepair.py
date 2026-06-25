from dataclasses import dataclass

from mrcpsp import Project

@dataclass
class ModeAssignmentScore:
    nonrenewable_score: int
    duration_score: int

    def __lt__(self, other: "ModeAssignmentScore") -> bool:
        return (self.nonrenewable_score, self.duration_score) < (other.nonrenewable_score, other.duration_score)

class NonRenewableRepair:

    def repair_nonrenewable(self, project: Project, mode_assignments: list[int]) -> list[int]|None:
        """ Uses a greedy local hill climbing search to try and statisfy non-renewable
        constraints.

        At each step, it picks a mode switch which give the best total non-renewable reduction,
        with duration increase as tie-breaker.

        Returns repaired mode-assignments is feasible solution is found, and None
        if no solution is found (either because we got stuck in a local optima,
        or hit the iteration cap).
        """

        iteration_limit = project.num_activities * 10

        current_mode_assignment = mode_assignments.copy()

        for _ in range(iteration_limit):
            current_score = self._get_mode_assignment_score(project, current_mode_assignment)

            if current_score.nonrenewable_score == 0: return current_mode_assignment

            neighbourhood = self._generate_neighbourhood(project, current_mode_assignment)

            best_neighbour = self._get_best_neighbour(current_score, neighbourhood,project)

            if best_neighbour is None: return None
            current_mode_assignment = best_neighbour

        return None # Iteration cap has been hit

    def _get_best_neighbour(self, current_score, neighbourhood,project):
        """
        Return best neighbour, or None if no neighbours beat the current score.
        """
        best_neighbour = None
        best_neighbour_score = current_score
        for neighbour in neighbourhood:
            neighbour_score = self._get_mode_assignment_score(project, neighbour)
            if neighbour_score < best_neighbour_score:
                best_neighbour = neighbour
                best_neighbour_score = neighbour_score
        return best_neighbour

    @staticmethod
    def _generate_neighbourhood(project:Project, current_mode_assignment: list[int]) -> list[list[int]]:
        neighbourhood = []
        for activity_index in range(project.num_activities):
            activity = project.activities[activity_index]
            for mode_index in range(len(activity.modes)):
                if mode_index != current_mode_assignment[activity_index]:
                    new_assignment = current_mode_assignment.copy()
                    new_assignment[activity_index] = mode_index
                    neighbourhood.append(new_assignment)

        return neighbourhood

    @staticmethod
    def _get_mode_assignment_score(project: Project, mode_assignments: list[int])-> ModeAssignmentScore:
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

        return ModeAssignmentScore(
            nonrenewable_score=sum(non_renewable_scores),
            duration_score=sum(duration_score)
        )