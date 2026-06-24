# from mrcpsp import Project, Mode, Activity, Schedule
# from validate import ScheduleValidator
from mrcpsp import *
from validation import ScheduleValidator

SOURCE_SINK_MODE = [Mode(duration=0, renewable_demands=[0, 0, 0], nonrenewable_demands=[0, 0])]
ACTIVITY_MODES = [
    Mode(duration=1, renewable_demands=[1, 1, 0], nonrenewable_demands=[0, 0]),
    Mode(duration=2, renewable_demands=[0, 1, 0], nonrenewable_demands=[0, 1]),
    Mode(duration=2, renewable_demands=[1, 0, 0], nonrenewable_demands=[1, 0]),
]
ACTIVITIES = [
    Activity(id=0, modes=SOURCE_SINK_MODE, successors=[1]),
    Activity(id=1, modes=ACTIVITY_MODES, successors=[2,3]),
    Activity(id=2, modes=ACTIVITY_MODES, successors=[4]),
    Activity(id=3, modes=ACTIVITY_MODES, successors=[4]),
    Activity(id=4, modes=SOURCE_SINK_MODE, successors=[]),
]
PROJECT = Project(
            num_activities=5,
            num_renewable=3,
            num_nonrenewable=2,
            renewable_capacities=[1,1,0],
            nonrenewable_capacities=[1,1],
            activities=ACTIVITIES,
        )

SCHEDULE = Schedule(
    mode_assignments=[0, 0, 1, 2, 0],
    start_times=[0, 0, 1, 1, 3],
    project=PROJECT,
)
class TestValidate:
    def test_valid_schedule(self):
        expected_output = []
        assert ScheduleValidator().validate(SCHEDULE) == expected_output

    def test_invalid_mode_indices(self):
        invalid_mode_indices_schedule = Schedule(
            # The initial source task only has mode 0, so assigning mode 1 is invalid
            mode_assignments=[1,0,0,0,0],
            start_times= SCHEDULE.start_times,
            project=PROJECT,
        )
        expected_output = ["Activity 0: invalid mode 1"]
        assert ScheduleValidator().validate(invalid_mode_indices_schedule) == expected_output

    def test_invalid_precedence(self):
        invalid_precedence_schedule = Schedule(
            mode_assignments=[0, 0, 1, 2, 0],
            start_times=[4, 0, 1, 1, 3],
            project=PROJECT,
        )
        expected_output = ["Precedence violation: activity 0 finishes at 4 but successor 1 starts at 0"]
        assert ScheduleValidator().validate(invalid_precedence_schedule) == expected_output

    def test_invalid_renewable_demands(self):
        invalid_renewable_demands_schedule = Schedule(
            mode_assignments=[0, 0, 0, 0, 0],
            start_times=[0, 0, 1, 1, 2],
            project=PROJECT,
        )
        expected_output = ["Renewable resource 0 exceeded at time 1: usage 2 > capacity 1",
                           "Renewable resource 1 exceeded at time 1: usage 2 > capacity 1"]
        assert ScheduleValidator().validate(invalid_renewable_demands_schedule) == expected_output

    def test_invalid_nonrenewable_demands(self):
        invalid_nonrenewable_demands_schedule = Schedule(
            mode_assignments=[0, 1, 1, 2, 0],
            start_times=[0, 0, 2, 2, 4],
            project=PROJECT,
        )
        expected_output = [ "Non-renewable resource 1 exceeded: total 2 > capacity 1"]
        assert ScheduleValidator().validate(invalid_nonrenewable_demands_schedule) == expected_output