import pytest

from herd_optimizer.schemas import CandidateDecision, HerdCaseConfig
from herd_optimizer.workflow import _apply_decision_to_case


def test_apply_decision_to_case_scales_feed_and_slweight():
    case = HerdCaseConfig(feed1=10.0, feed2=4.0, slaughter_weight=100.0)
    decision = CandidateDecision(feed1_scale=1.2, feed2_scale=0.5, slweight_scale=1.1)
    scaled = _apply_decision_to_case(case, decision)
    assert scaled.feed1 == 12.0
    assert scaled.feed2 == 2.0
    assert scaled.slaughter_weight == pytest.approx(110.0)
