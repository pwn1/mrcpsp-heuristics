from mode_rules import get_mode_fn
from priority_rules import get_priority_fn, PRIORITY_RULES
from schedule_generation_schemes import *

from default_mrcpsp_resources import *

PRIORITY_RULE = get_priority_fn("LFT/LST",PROJECT,"","")
NON_CONTEXT_AWARE_MODE_RULE = get_mode_fn("shortest_duration/longest_duration",PROJECT,"","")
CONTEXT_AWARE_MODE_RULE = get_mode_fn("earliest_start/earliest_finish",PROJECT,"","")

# TODO: Create synthetic test priority and mode rules, to better test the underlying algorithms
#  and remove dependence on these rules from the tests

class TestSerialSGS:
    def test_serial_context_unaware(self):
        result = serial_sgs(PROJECT, PRIORITY_RULE, NON_CONTEXT_AWARE_MODE_RULE, mode_is_context_aware=False)
        assert result.start_times == [0,0,1,2,3]
        assert result.mode_assignments == [0,0,0,0,0]

    def test_parallel_context_unaware(self):
        result = parallel_sgs(PROJECT, PRIORITY_RULE, NON_CONTEXT_AWARE_MODE_RULE, mode_is_context_aware=False)
        assert result.start_times == [0,0,1,2,3]
        assert result.mode_assignments == [0,0,0,0,0]

    def test_serial_context_aware(self):
        result = serial_sgs(PROJECT, PRIORITY_RULE, CONTEXT_AWARE_MODE_RULE, mode_is_context_aware=True)
        assert result.start_times == [0, 0, 1, 2, 3]
        assert result.mode_assignments == [0, 0, 0, 0, 0]

    def test_parallel_context_aware(self):
        result = parallel_sgs(PROJECT, PRIORITY_RULE, CONTEXT_AWARE_MODE_RULE, mode_is_context_aware=True)
        assert result.start_times == [0, 0, 1, 2, 3]
        assert result.mode_assignments == [0, 0, 0, 0, 0]