from pathlib import Path

from herd_optimizer.climate import generate_random_future_climate, read_climate_history, split_past_future
from herd_optimizer.schemas import SeasonState


def test_random_future_preserves_past_and_length():
    path = Path(__file__).resolve().parents[1] / "assets" / "FRACHA19982012.csv"
    df = read_climate_history(path)
    state = SeasonState(current_day=15, season_end_day=40)
    randomized = generate_random_future_climate(df, state, seed=7)
    past_a, future_a = split_past_future(df, state)
    past_b, future_b = split_past_future(randomized, state)
    assert len(past_a) == len(past_b)
    assert len(future_a) == len(future_b)
    assert past_a.equals(past_b)
    assert not future_a.equals(future_b)
