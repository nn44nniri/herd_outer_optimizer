from pathlib import Path

from herd_optimizer.climate import aggregate_regime, read_climate_history, slice_season, split_past_future
from herd_optimizer.schemas import SeasonState


def test_climate_schema_and_split():
    path = Path(__file__).resolve().parents[1] / "assets" / "FRACHA19982012.csv"
    df = read_climate_history(path)
    state = SeasonState(current_day=10, season_end_day=30)
    season = slice_season(df, state)
    past, future = split_past_future(df, state)
    assert len(season) == 30
    assert len(past) == 10
    assert len(future) == 20
    agg = aggregate_regime(past)
    assert agg.n_days == 10
