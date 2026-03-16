from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import CandidateDecision, ClimateAggregation, SeasonState

CLIMATE_COLUMNS = ["RAD", "MINT", "MAXT", "VPR", "WIND", "RAIN", "OKTA"]


def read_climate_history(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in CLIMATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required climate columns: {missing}")
    df = df.copy()
    for col in CLIMATE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in ("YR", "DOY"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def slice_season(df: pd.DataFrame, state: SeasonState) -> pd.DataFrame:
    start = max(0, state.season_start_day - 1)
    end = min(len(df), state.season_end_day)
    season = df.iloc[start:end].copy().reset_index(drop=True)
    if season.empty:
        raise ValueError("Selected season slice is empty")
    return season


def aggregate_regime(df: pd.DataFrame) -> ClimateAggregation:
    return ClimateAggregation(
        rad=float(df["RAD"].sum()) if "RAD" in df else 0.0,
        mint=float(df["MINT"].sum()) if "MINT" in df else 0.0,
        maxt=float(df["MAXT"].sum()) if "MAXT" in df else 0.0,
        vpr=float(df["VPR"].sum()) if "VPR" in df else 0.0,
        wind=float(df["WIND"].sum()) if "WIND" in df else 0.0,
        rain=float(df["RAIN"].sum()) if "RAIN" in df else 0.0,
        okta=float(df["OKTA"].sum()) if "OKTA" in df else 0.0,
        n_days=int(len(df)),
    )


def split_past_future(df: pd.DataFrame, state: SeasonState) -> tuple[pd.DataFrame, pd.DataFrame]:
    season = slice_season(df, state)
    cut = min(len(season), max(0, state.current_day - state.season_start_day + 1))
    past = season.iloc[:cut].copy().reset_index(drop=True)
    future = season.iloc[cut:].copy().reset_index(drop=True)
    return past, future


def _clip_okta(values: pd.Series) -> pd.Series:
    return values.clip(lower=0.0, upper=8.0)


def apply_candidate_to_future(
    full_history: pd.DataFrame,
    state: SeasonState,
    decision: CandidateDecision,
) -> pd.DataFrame:
    modified = full_history.copy()
    start = max(0, state.season_start_day - 1)
    end = min(len(modified), state.season_end_day)
    cut = min(end, max(start, state.current_day))
    future = modified.iloc[cut:end].copy()
    if not future.empty:
        for col in CLIMATE_COLUMNS:
            if col in future.columns:
                future.loc[:, col] = pd.to_numeric(future[col], errors="coerce").astype(float)
        future.loc[:, "RAD"] = future["RAD"] * decision.rad_scale
        future.loc[:, "MINT"] = future["MINT"] + decision.mint_shift
        future.loc[:, "MAXT"] = future["MAXT"] + decision.maxt_shift
        future.loc[:, "VPR"] = future["VPR"] * decision.vpr_scale
        future.loc[:, "WIND"] = np.clip(future["WIND"] * decision.wind_scale, 0.01, None)
        future.loc[:, "RAIN"] = np.clip(future["RAIN"] * decision.rain_scale, 0.0, None)
        future.loc[:, "OKTA"] = _clip_okta(future["OKTA"] + decision.okta_shift)
        modified.iloc[cut:end] = future
    return modified


def generate_random_future_climate(
    full_history: pd.DataFrame,
    state: SeasonState,
    seed: int | None = None,
) -> pd.DataFrame:
    """Bootstrap-generate a random future climate regime for exploitation mode."""
    randomized = full_history.copy()
    start = max(0, state.season_start_day - 1)
    end = min(len(randomized), state.season_end_day)
    cut = min(end, max(start, state.current_day))
    future = randomized.iloc[cut:end].copy().reset_index(drop=True)
    if future.empty:
        return randomized

    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, len(future), size=len(future))
    sampled = future.iloc[sampled_idx].reset_index(drop=True)

    if "DOY" in sampled.columns:
        start_doy = int(randomized.iloc[cut - 1]["DOY"]) + 1 if cut > 0 else state.season_start_day
        sampled.loc[:, "DOY"] = np.arange(start_doy, start_doy + len(sampled))
    if "YR" in sampled.columns and cut > 0:
        sampled.loc[:, "YR"] = randomized.iloc[cut - 1]["YR"]

    randomized.iloc[cut:end] = sampled.values
    return randomized


def compute_target_triplet(
    full_history: pd.DataFrame,
    state: SeasonState,
    decision: CandidateDecision,
) -> tuple[ClimateAggregation, ClimateAggregation, ClimateAggregation]:
    adjusted_full = apply_candidate_to_future(full_history, state, decision)
    full_target = aggregate_regime(slice_season(adjusted_full, state))
    realized_past, _ = split_past_future(adjusted_full, state)
    _, required_future = split_past_future(adjusted_full, state)
    return full_target, aggregate_regime(realized_past), aggregate_regime(required_future)


def save_adjusted_climate_csv(
    full_history: pd.DataFrame,
    state: SeasonState,
    decision: CandidateDecision,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    adjusted = apply_candidate_to_future(full_history, state, decision)
    adjusted.to_csv(output_path, index=False)
    return output_path


def aggregation_dict(agg: ClimateAggregation) -> dict[str, float | int]:
    return agg.to_dict()
