from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SeasonState:
    current_day: int
    season_start_day: int = 1
    season_end_day: int = 365

    @property
    def remaining_days(self) -> int:
        return max(0, self.season_end_day - self.current_day)

    @property
    def elapsed_days(self) -> int:
        return max(0, self.current_day - self.season_start_day)

    @property
    def elapsed_fraction(self) -> float:
        total = max(1, self.season_end_day - self.season_start_day + 1)
        return min(1.0, max(0.0, self.elapsed_days / total))


@dataclass(slots=True)
class HerdCaseConfig:
    genotype: int = 1
    location: str = "FRANCE1"
    housing1: int = 0
    housing2: int = 1
    housing3: int = 0
    feed1: float = 20.0
    feed2: float = 0.0
    feednr: int = 1
    slaughter_weight: float = 936.1493
    debug: bool = False


@dataclass(slots=True)
class CandidateDecision:
    rad_scale: float = 1.0
    mint_shift: float = 0.0
    maxt_shift: float = 0.0
    vpr_scale: float = 1.0
    wind_scale: float = 1.0
    rain_scale: float = 1.0
    okta_shift: float = 0.0
    feed1_scale: float = 1.0
    feed2_scale: float = 1.0
    slweight_scale: float = 1.0

    def as_vector(self) -> list[float]:
        return [
            self.rad_scale,
            self.mint_shift,
            self.maxt_shift,
            self.vpr_scale,
            self.wind_scale,
            self.rain_scale,
            self.okta_shift,
            self.feed1_scale,
            self.feed2_scale,
            self.slweight_scale,
        ]

    @classmethod
    def from_vector(cls, values: list[float]) -> "CandidateDecision":
        keys = [f.name for f in fields(cls)]
        if len(values) != len(keys):
            raise ValueError(f"Expected {len(keys)} values, got {len(values)}")
        return cls(**dict(zip(keys, values)))


@dataclass(slots=True)
class ClimateAggregation:
    rad: float
    mint: float
    maxt: float
    vpr: float
    wind: float
    rain: float
    okta: float
    n_days: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class SimulationResult:
    success: bool
    case_number: int
    beef_production_herd_kg: float
    beef_production_repr_cow_kg: float
    beef_production_bull_calf_kg: float
    feed_efficiency_herd_g_per_kg_dm: float
    feed_efficiency_repr_g_per_kg_dm: float
    feed_efficiency_bull_g_per_kg_dm: float
    slaughter_weight_bull_calf_kg: float
    feed_fraction_repr_cow: float
    runtime_seconds: float
    stdout: str = ""
    stderr: str = ""
    plot_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def objective_vector(self) -> list[float]:
        # Both objectives are maximized.
        return [
            float(self.beef_production_herd_kg),
            float(self.feed_efficiency_herd_g_per_kg_dm),
        ]

    def constraint_vector(self) -> list[float]:
        # qNEHVI constraint convention: <= 0 is feasible.
        simulation_failure = 0.0 if self.success else 1.0
        negative_weights = -min(self.slaughter_weight_bull_calf_kg, 0.0)
        return [simulation_failure, negative_weights]


@dataclass(slots=True)
class OptimizationRecord:
    x: list[float]
    objectives: list[float]
    constraints: list[float]
    decision: dict[str, Any]
    result: dict[str, Any]
    climate_target_full: dict[str, Any]
    climate_realized_past: dict[str, Any]
    climate_required_future: dict[str, Any]


@dataclass(slots=True)
class OptimizationConfig:
    simulator_path: Path
    climate_history_path: Path
    output_dir: Path
    season_state: SeasonState = field(default_factory=SeasonState)
    herd_case: HerdCaseConfig = field(default_factory=HerdCaseConfig)
    n_initial: int = 6
    n_iterations: int = 12
    batch_size: int = 1
    random_seed: int = 1234
    use_qnehvi: bool = True
    use_hvkg_placeholder: bool = False
    ref_point: tuple[float, float] = (0.0, 0.0)
    lower_bounds: tuple[float, ...] = (0.70, -5.0, -5.0, 0.70, 0.50, 0.50, -2.0, 0.50, 0.50, 0.70)
    upper_bounds: tuple[float, ...] = (1.30, 5.0, 5.0, 1.30, 1.50, 1.50, 2.0, 1.50, 1.50, 1.30)

    def validate(self) -> None:
        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Search-space bounds must have the same length")
        if self.n_initial < 2:
            raise ValueError("n_initial must be at least 2")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(slots=True)
class ExploitationReport:
    season_state: dict[str, Any]
    current_target_id: int
    objective_estimate: dict[str, float]
    optimal_full_cumulative_regime: dict[str, Any]
    realized_past_regime: dict[str, Any]
    required_future_regime: dict[str, Any]
    delta_to_optimal_full_regime: dict[str, Any]
    recommended_decision: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
