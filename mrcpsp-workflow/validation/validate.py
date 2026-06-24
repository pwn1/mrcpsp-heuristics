from __future__ import annotations

"""Schedule validations for MRCPSP."""

from mrcpsp import *

class ScheduleValidator:
    def validate(self, schedule: Schedule) -> list[str]:
        """Validate a schedule against all MRCPSP constraints.

        Returns a list of violation descriptions (empty = valid).
        """
        invalid_mode_errors = self._check_modes_are_valid(schedule)

        # If modes are invalid, this will cause out of bounds errors for subsequent checks,
        # so we early return
        if invalid_mode_errors: return invalid_mode_errors

        return [
            *self._check_precedence_constraints(schedule),
            *self._check_renewable_resource_constraints(schedule),
            *self._check_nonrenewable_resource_constraints(schedule)
        ]

    @staticmethod
    def _check_modes_are_valid(schedule: Schedule) -> list[str]:
        # Check mode assignments are valid indices
        errors = []
        project = schedule.project
        for activity_index in range(project.num_activities):
            activity = project.activities[activity_index]
            mode_assignment = schedule.mode_assignments[activity_index]
            valid_assignment = 0<=mode_assignment < len(activity.modes)

            if not valid_assignment:
                errors.append(f"Activity {activity.id}: invalid mode {mode_assignment}")

        return errors

    @staticmethod
    def _check_precedence_constraints(schedule: Schedule) -> list[str]:
        errors = []
        scheduled_activities = schedule.scheduled_activities

        for activity in scheduled_activities:
            successors = [scheduled_activities[index] for index in activity.successors]
            for successor in successors:
                if activity.end_time > successor.start_time:
                    errors.append(
                        f"Precedence violation: activity {activity.id} finishes at {activity.end_time} "
                        f"but successor {successor.id} starts at {successor.start_time}"
                    )

        return errors

    def _check_renewable_resource_constraints(self, schedule: Schedule) -> list[str]:
        errors = []
        project = schedule.project

        # Iterate over each timeslot, and calculate resource usage at each point
        for timeslot in range(schedule.compute_makespan()):

            active_activities_renewable_demands = [
                activity.selected_mode.renewable_demands
                for activity in schedule.scheduled_activities if
                activity.start_time <= timeslot < activity.end_time
            ]

            usage = self._sum_quotas_helper(active_activities_renewable_demands, project.num_renewable)

            timeslot_renewable_errors = [
                    f"Renewable resource {r} exceeded at time {timeslot}: "
                    f"usage {usage[r]} > capacity {project.renewable_capacities[r]}"
                    for r in range(project.num_renewable) if usage[r] > project.renewable_capacities[r]
                ]

            errors.extend(timeslot_renewable_errors)

        return errors

    def _check_nonrenewable_resource_constraints(self, schedule: Schedule) -> list[str]:
        project = schedule.project

        activity_nonrenewable_demands = [
            activity.selected_mode.nonrenewable_demands
            for activity in schedule.scheduled_activities
        ]

        usage = self._sum_quotas_helper(activity_nonrenewable_demands, project.num_nonrenewable)

        return [
            f"Non-renewable resource {nr} exceeded: "
            f"total {usage[nr]} > capacity {project.nonrenewable_capacities[nr]}"
            for nr in range(project.num_nonrenewable) if usage[nr] > project.nonrenewable_capacities[nr]
        ]

    @staticmethod
    def _sum_quotas_helper(quota_list: list[list[int]], default_size:int) -> list[int]:
        return [
            int(sum(quotas)) for quotas in zip(
                *quota_list, [0]*default_size
            )
        ]