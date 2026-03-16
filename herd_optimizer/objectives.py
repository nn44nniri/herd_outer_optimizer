from __future__ import annotations

from typing import Any

from .schemas import SimulationResult


def summarize_result(result: SimulationResult) -> dict[str, Any]:
    return {
        "beef_production_herd_kg": result.beef_production_herd_kg,
        "feed_efficiency_herd_g_per_kg_dm": result.feed_efficiency_herd_g_per_kg_dm,
        "slaughter_weight_bull_calf_kg": result.slaughter_weight_bull_calf_kg,
        "feed_fraction_repr_cow": result.feed_fraction_repr_cow,
        "runtime_seconds": result.runtime_seconds,
        "success": result.success,
    }


def result_to_training_row(result: SimulationResult) -> tuple[list[float], list[float]]:
    return result.objective_vector(), result.constraint_vector()
