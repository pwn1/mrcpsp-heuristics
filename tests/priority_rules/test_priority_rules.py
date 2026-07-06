from default_mrcpsp_resources import PROJECT
from priority_rules import LFT


class TestPriorityRules:
    def test_lft(self):
        # Latest Finish Time
        expected_priorities = [0,1,2,2,2]
        actual_priorities = LFT.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities