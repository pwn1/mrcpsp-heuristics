from mrcpsp import *

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