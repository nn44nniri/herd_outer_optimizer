"""Outer-loop livestock optimizer around LiGAPS-Beef."""

from .schemas import (
    CandidateDecision,
    ClimateAggregation,
    ExploitationReport,
    HerdCaseConfig,
    OptimizationConfig,
    OptimizationRecord,
    SeasonState,
    SimulationResult,
)
from .workflow import train_optimizer, generate_operation_report

__all__ = [
    "CandidateDecision",
    "ClimateAggregation",
    "ExploitationReport",
    "HerdCaseConfig",
    "OptimizationConfig",
    "OptimizationRecord",
    "SeasonState",
    "SimulationResult",
    "train_optimizer",
    "generate_operation_report",
]
