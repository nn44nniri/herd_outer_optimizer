from pathlib import Path

from herd_optimizer.ligaps_sync import LiGAPSSynchronizer
from herd_optimizer.schemas import HerdCaseConfig


def test_sync_patcher_injects_json_marker():
    root = Path(__file__).resolve().parents[1]
    sync = LiGAPSSynchronizer(
        simulator_path=root / "assets" / "LiGAPSBeef20180301_herd_worked.py",
        climate_history_path=root / "assets" / "FRACHA19982012.csv",
    )
    src = sync._build_patched_source(HerdCaseConfig(), "FRACHA19982012.csv")
    assert "__LIGAPS_JSON__" in src
    assert "case_ids = [1]" in src
