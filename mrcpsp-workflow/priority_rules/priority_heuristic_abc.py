from abc import ABC, abstractmethod

from mrcpsp import Project


class PriorityHeuristic(ABC):
    """
    We use the convention that lower values, mean an item is higher priority.

    This means, for some priority rules which are traditionally "higher is better"
    we have to take their negation.
    """
    @staticmethod
    @abstractmethod
    def prioritise(project:Project, mode_assignments: list[int]) -> list[int]:
        pass

    @staticmethod
    def _compute_successors_recursive(project: Project) -> list[set[int]]:
        """Compute the transitive closure of successors for each activity."""
        all_successors: list[set[int] | None] = [None] * project.num_activities

        def _get(act_id: int) -> set[int]:
            cached = all_successors[act_id]
            if cached is not None:
                return cached

            result = set()
            for s in project.activities[act_id].successors:
                result.add(s)
                result |= _get(s)

            all_successors[act_id] = result
            return result

        for a in range(project.num_activities):
            _get(a)

        # By now every slot has been filled in, so this is safe.
        return [s if s is not None else set() for s in all_successors]