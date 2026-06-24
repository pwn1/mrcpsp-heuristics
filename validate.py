from __future__ import annotations

"""Schedule validation for MRCPSP."""

from mrcpsp import Project, Schedule


class ScheduleValidator:
    def validate(self, schedule: Schedule) -> list[str]:
        """Validate a MRCPSP schedule."""
        errors = self.check_modes_are_valid(schedule)

        # If modes are invalid, this will cause out of bounds errors for subsequent checks,
        # so we early return
        if errors: return errors

        errors.extend(self.check_precedence_constraints(schedule))

        errors.extend(self._check_renewable_resource_constraints(schedule))

        # Check non-renewable resource constraints
        project = schedule.project
        for nr in range(project.num_nonrenewable):
            total = sum(
                project.activities[i].modes[schedule.mode_assignments[i]].nonrenewable_demands[nr]
                for i in range(project.num_activities)
            )
            if total > project.nonrenewable_capacities[nr]:
                errors.append(
                    f"Non-renewable resource {nr} exceeded: "
                    f"total {total} > capacity {project.nonrenewable_capacities[nr]}"
                )

        return errors

    @staticmethod
    def _check_renewable_resource_constraints(schedule: Schedule):
        errors = []

        project = schedule.project

        # Iterate over each timeslot, and calculate resource usage at each point
        for timeslot in range(schedule.compute_makespan()):

            active_activities_renewable_demands = [
                activity.selected_mode.renewable_demands
                for activity in schedule.scheduled_activities if
                activity.start_time <= timeslot < activity.end_time
            ]

            usage = [
                int(sum(renewable_quotas)) for renewable_quotas in zip(
                    *active_activities_renewable_demands, [0] * project.num_renewable
                )
            ]

            timeslot_renewable_errors = [
                    f"Renewable resource {r} exceeded at time {timeslot}: "
                    f"usage {usage[r]} > capacity {project.renewable_capacities[r]}"
                    for r in range(project.num_renewable) if usage[r] > project.renewable_capacities[r]
                ]

            errors.extend(timeslot_renewable_errors)

        return errors

    @staticmethod
    def check_precedence_constraints(schedule: Schedule):
        # Check precedence constraints
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

    @staticmethod
    def check_modes_are_valid(schedule: Schedule):
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


def validate_schedule(project: Project, schedule: Schedule) -> list[str]:
    """Validate a schedule against all MRCPSP constraints.

    Returns a list of violation descriptions (empty = valid).
    """
    return ScheduleValidator().validate(schedule)
