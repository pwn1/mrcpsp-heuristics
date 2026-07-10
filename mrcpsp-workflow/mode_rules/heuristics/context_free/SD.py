from mrcpsp import Activity


class SD:
    """ Shortest Duration (SD) always chooses the mode which takes the smallest
    amount of time to execute.

    This is a static (context-free) re-interpretation of the SFM (shortest
    Feasible Mode), presented in Boctor (1993)."""
    @staticmethod
    def get_name():
        return "SD"

    @staticmethod
    def get_mode_scores(activity: Activity):
        return [mode.duration for mode in activity.modes]