from mrcpsp import Project, Mode, Activity, Schedule
from validate import validate_schedule

SOURCE_SINK_MODE = [Mode(duration=0, renewable_demands=[0, 0], nonrenewable_demands=[0, 0])]
ACTIVITY_MODES = [
    Mode(duration=1, renewable_demands=[1, 1], nonrenewable_demands=[1, 1]),
    Mode(duration=2, renewable_demands=[0, 1], nonrenewable_demands=[0, 1]),
    Mode(duration=2, renewable_demands=[1, 0], nonrenewable_demands=[1, 0]),
]
ACTIVITIES = [
    Activity(id=0, modes=SOURCE_SINK_MODE, successors=[1]),
    Activity(id=1, modes=ACTIVITY_MODES, successors=[2]),
    Activity(id=2, modes=SOURCE_SINK_MODE, successors=[]),
]
PROJECT = Project(
            num_activities=3,
            num_renewable=2,
            num_nonrenewable=2,
            renewable_capacities=[1,1],
            nonrenewable_capacities=[1,1],
            activities=ACTIVITIES,
        )

SCHEDULE = Schedule(
    mode_assignments=[0, 2, 0],
    start_times=[0, 0, 2]
)
class TestValidate:
    def test_valid_schedule(self):
        assert validate_schedule(PROJECT, SCHEDULE) == []
