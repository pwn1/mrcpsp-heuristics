from default_mrcpsp_resources import PROJECT
from priority_rules import LFT, LST, LSTLFT
from priority_rules.priority_rules import RWK


class TestPriorityRules:
    def test_lft(self):
        # Latest Finish Time
        expected_priorities = [0,1,2,2,2]
        actual_priorities = LFT.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_lst(self):
        expected_priorities = [0,0,1,1,2]
        actual_priorities = LST.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_lstlft(self):
        expected_priorities = [0,1,3,3,4]
        actual_priorities = LSTLFT.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_rwk(self):
        expected_priorities = [-3,-3,-1,-1,0]
        actual_priorities = RWK.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities