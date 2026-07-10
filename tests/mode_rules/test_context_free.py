from default_mrcpsp_resources import ACTIVITIES
from mode_rules import SD


class TestContextFreeModeRule:
    def test_sd(self):
        # SD (Shortest Duration)
        expected_scores = [1,2,2]
        actual_scores = SD.get_mode_scores(ACTIVITIES[1])
        assert actual_scores == expected_scores