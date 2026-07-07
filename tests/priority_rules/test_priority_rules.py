from default_mrcpsp_resources import PROJECT
from priority_rules.heuristics import AN, GRD, GRPW, LFT, LST, LSTLFT, MSLK, MTS, NIS, RWK, SPT, WRUP

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

    def test_grpw(self):
        expected_priorities = [-1, -3, -1, -1, 0]
        actual_priorities = GRPW.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_mslk_with_all_items_on_critical_path(self):
        # Expected priorities are all 0, because all tasks are on the critical path with no
        # slack
        expected_priorities = [0,0,0,0,0]
        actual_priorities = MSLK.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_mslk_with_slack(self):
        expected_priorities = [0,0,0,1,0]
        actual_priorities = MSLK.prioritise(PROJECT,[0,0,1,0,0])
        assert actual_priorities == expected_priorities

    def test_mts(self):
        expected_priorities = [-4,-3,-1,-1,-0]
        actual_priorities = MTS.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_wrup(self):
        expected_priorities = [-0.7,-2,-1.3,-1.3,0]
        actual_priorities = WRUP.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_spt(self):
        expected_priorities = [0,1,2,1,0]
        actual_priorities = SPT.prioritise(PROJECT,[0,0,2,0,0])
        assert actual_priorities == expected_priorities

    def test_nis(self):
        expected_priorities = [-1,-2,-1,-1,0]
        actual_priorities = NIS.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities

    def test_grd(self):
        expected_priorities = [0,-2,-2,-2,0]
        actual_priorities = GRD.prioritise(PROJECT,[0,0,1,2,0])
        assert actual_priorities == expected_priorities

    def test_an(self):
        expected_priorities = [0,1,2,3,4]
        actual_priorities = AN.prioritise(PROJECT,[0,0,0,0,0])
        assert actual_priorities == expected_priorities