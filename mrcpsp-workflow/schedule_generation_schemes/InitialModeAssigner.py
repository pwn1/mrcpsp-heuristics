from typing import Protocol

class InitialModeAssigner(Protocol):
    @staticmethod
    def assign_modes(project, priority_fn, mode_fn, core): ...

class ContextAwareModeAssigner:
    @staticmethod
    def assign_modes(project, priority_fn, mode_fn, core):
        proxy_modes = [min(range(len(a.modes)), key=lambda m: a.modes[m].duration)
                       for a in project.activities]
        priorities = priority_fn(project=project, mode_assignments=proxy_modes)
        mode_assignments = [0] * project.num_activities
        core(project, priorities, mode_assignments, mode_fn=mode_fn)
        return mode_assignments

class ContextUnAwareModeAssigner:
    @staticmethod
    def assign_modes(project, priority_fn, mode_fn, core):
        return [
            mode_fn(activity=project.activities[i]) for i in range(project.num_activities)
        ]