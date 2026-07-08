from dataclasses import dataclass

from mrcpsp import Project


@dataclass(frozen=True)
class CpmAssumptions:
    project:Project
    mode_assignments: list[int]


@dataclass(frozen=True)
class CpmSchedule:
    assumptions: CpmAssumptions
    earliest_start_time: list[int]
    earliest_finish_time: list[int]
    latest_start_time: list[int]
    latest_finish_time: list[int]

    @property
    def slack(self) -> list[int]:
        return [
            self.latest_start_time[i]-self.earliest_start_time[i]
            for i in range(self.assumptions.project.num_activities)
        ]

class CriticalPathMethodCalculator:
    @staticmethod
    def get_cpm_schedule(project: Project, mode_assignments: list[int]):
        durations = project.durations_given_modes(mode_assignments)

        topological_order = CriticalPathMethodCalculator._get_topological_order(project)

        earliest_start_time = CriticalPathMethodCalculator._calculate_earliest_start_time(
            durations,
            project,
            topological_order
        )
        earliest_finish_time = [
            earliest_start_time[i] + durations[i]
            for i in range(project.num_activities)
        ]

        makespan = max(earliest_start_time[i] + durations[i] for i in range(project.num_activities))

        latest_finish_time = CriticalPathMethodCalculator._find_latest_finish_time(
            durations,
            project,
            topological_order,
            makespan,
        )
        latest_start_time = [
            latest_finish_time[i] - durations[i]
            for i in range(project.num_activities)
        ]

        return CpmSchedule(
            assumptions=CpmAssumptions(
                project=project,
                mode_assignments=mode_assignments,
            ),
            earliest_start_time=earliest_start_time,
            earliest_finish_time=earliest_finish_time,
            latest_start_time=latest_start_time,
            latest_finish_time=latest_finish_time,
        )


    @staticmethod
    def _get_topological_order(project: Project) -> list[int]:
        """Compute a topological ordering of activities using Kahn's algorithm."""
        n = project.num_activities
        in_degree = [0] * n
        for act in project.activities:
            for s in act.successors:
                in_degree[s] += 1

        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for s in project.activities[node].successors:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
        return order

    @staticmethod
    def _calculate_earliest_start_time(
            durations: list[int],
            project: Project,
            topological_order: list[int]
    ) -> list[int]:
        earliest_start_time = [0] * project.num_activities

        for activity_id in topological_order:
            activity = project.activities[activity_id]
            activity_end_time = earliest_start_time[activity_id] + durations[activity_id]

            for successor_id in activity.successors:
                earliest_start_time[successor_id] = max(
                    earliest_start_time[successor_id],
                    activity_end_time
                )
        return earliest_start_time

    @staticmethod
    def _find_latest_finish_time(
            durations: list[int],
            project: Project,
            topological_order: list[int],
            makespan: int,
    ) -> list[int]:
        latest_finish_time = [makespan] * project.num_activities

        for activity_id in reversed(topological_order):
            activity = project.activities[activity_id]

            for successor_id in activity.successors:
                latest_finish_time[activity_id] = min(
                    latest_finish_time[activity_id],
                    latest_finish_time[successor_id] - durations[successor_id]
                )
        return latest_finish_time
