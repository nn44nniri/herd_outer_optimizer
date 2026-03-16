from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from .botorch_loop import BotorchMOBO, build_record, sobol_initial_design
from .climate import aggregation_dict, compute_target_triplet, generate_random_future_climate, read_climate_history
from .ligaps_sync import LiGAPSSynchronizer
from .objectives import result_to_training_row, summarize_result
from .schemas import CandidateDecision, ExploitationReport, HerdCaseConfig, OptimizationConfig, OptimizationRecord, SeasonState


ProgressCallback = Callable[[str], None]


def _default_progress(message: str) -> None:
    print(message, flush=True)


def _format_progress(prefix: str, index: int, total: int, elapsed_s: float) -> str:
    pct = 100.0 if total <= 0 else (100.0 * index / total)
    return f"[{prefix}] {index}/{total} ({pct:5.1f}%) elapsed={elapsed_s:0.1f}s"


def _decision_from_vector(x: np.ndarray) -> CandidateDecision:
    names = [f.name for f in fields(CandidateDecision)]
    return CandidateDecision(**{name: float(val) for name, val in zip(names, x.tolist())})


def _apply_decision_to_case(base_case: HerdCaseConfig, decision: CandidateDecision) -> HerdCaseConfig:
    return HerdCaseConfig(
        genotype=base_case.genotype,
        location=base_case.location,
        housing1=base_case.housing1,
        housing2=base_case.housing2,
        housing3=base_case.housing3,
        feed1=base_case.feed1 * decision.feed1_scale,
        feed2=base_case.feed2 * decision.feed2_scale,
        feednr=base_case.feednr,
        slaughter_weight=base_case.slaughter_weight * decision.slweight_scale,
        debug=base_case.debug,
    )


def _records_to_frame(records: list[OptimizationRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        row: dict[str, Any] = {"trial": idx}
        row.update({f"x_{i}": v for i, v in enumerate(rec.x)})
        row.update({"objective_meat": rec.objectives[0], "objective_feed_efficiency": rec.objectives[1]})
        row.update({"constraint_failure": rec.constraints[0], "constraint_negative_weight": rec.constraints[1]})
        for key, value in rec.decision.items():
            row[f"decision_{key}"] = value
        for key, value in rec.result.items():
            row[f"result_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def train_optimizer(
    config: OptimizationConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config.validate()
    progress = progress_callback or _default_progress
    config.output_dir.mkdir(parents=True, exist_ok=True)

    synchronizer = LiGAPSSynchronizer(config.simulator_path, config.climate_history_path)
    history = read_climate_history(config.climate_history_path)
    records: list[OptimizationRecord] = []

    initial_X = sobol_initial_design(config)
    total_evals = len(initial_X) + (config.n_iterations * config.batch_size if config.use_qnehvi else 0)
    completed = 0
    start_time = perf_counter()

    progress("[train] starting outer-loop optimization")
    progress(f"[train] output_dir={config.output_dir}")
    progress(f"[train] planned_evaluations={total_evals}")

    def _evaluate_point(x: np.ndarray) -> OptimizationRecord:
        decision = _decision_from_vector(x)
        effective_case = _apply_decision_to_case(config.herd_case, decision)
        result = synchronizer.evaluate(decision=decision, state=config.season_state, case=effective_case)
        objectives, constraints = result_to_training_row(result)
        full_target, realized_past, required_future = compute_target_triplet(history, config.season_state, decision)
        return build_record(
            x=x,
            objectives=objectives,
            constraints=constraints,
            decision=decision,
            result=summarize_result(result),
            full_target=aggregation_dict(full_target),
            realized_past=aggregation_dict(realized_past),
            required_future=aggregation_dict(required_future),
        )

    for x in initial_X:
        progress(f"[train] evaluating initial design point {completed + 1}/{total_evals}")
        records.append(_evaluate_point(np.asarray(x, dtype=float)))
        completed += 1
        progress(_format_progress("train", completed, total_evals, perf_counter() - start_time))

    if config.use_qnehvi:
        optimizer = BotorchMOBO(config)
        progress("[train] fitting surrogate-guided optimizer")
        for iteration in range(config.n_iterations):
            progress(f"[train] acquisition iteration {iteration + 1}/{config.n_iterations}: fitting and proposing candidates")
            train_X = np.asarray([rec.x for rec in records], dtype=float)
            train_Y = np.asarray([rec.objectives for rec in records], dtype=float)
            train_C = np.asarray([rec.constraints for rec in records], dtype=float)
            batch = optimizer.propose(train_X, train_Y, train_C)
            if getattr(optimizer, "last_strategy", "") == "fallback_random":
                progress("[train] surrogate variation is too small for stable MOBO updates; using a bounded random fallback batch")
            else:
                progress("[train] surrogate proposal strategy=qLogNEHVI")
            for batch_idx, x in enumerate(batch, start=1):
                progress(f"[train] evaluating acquisition candidate {batch_idx}/{len(batch)} for iteration {iteration + 1}")
                records.append(_evaluate_point(np.asarray(x, dtype=float)))
                completed += 1
                progress(_format_progress("train", completed, total_evals, perf_counter() - start_time))

    progress("[train] writing optimization archive")
    frame = _records_to_frame(records)
    frame.to_csv(config.output_dir / "optimization_archive.csv", index=False)

    archive = {
        "config": {
            "season_state": asdict(config.season_state),
            "herd_case": asdict(config.herd_case),
            "ref_point": list(config.ref_point),
            "lower_bounds": list(config.lower_bounds),
            "upper_bounds": list(config.upper_bounds),
        },
        "records": [asdict(rec) for rec in records],
    }
    (config.output_dir / "optimization_archive.json").write_text(json.dumps(archive, indent=2), encoding="utf-8")
    progress(_format_progress("train", completed, total_evals, perf_counter() - start_time))
    progress("[train] completed successfully")
    return archive


def _choose_best_record(records: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    feasible = [
        (idx, rec) for idx, rec in enumerate(records)
        if rec["constraints"][0] <= 0.0 and rec["constraints"][1] <= 0.0
    ]
    ranked = feasible if feasible else list(enumerate(records))
    best = max(
        ranked,
        key=lambda item: (item[1]["objectives"][0], item[1]["objectives"][1]),
    )
    return best



def generate_operation_report(
    archive_path: str | Path,
    current_day: int,
    climate_history_path: str | Path,
    *,
    randomize_future: bool = False,
    random_seed: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ExploitationReport:
    progress = progress_callback or _default_progress
    start_time = perf_counter()
    archive_path = Path(archive_path)
    progress("[operate] loading optimization archive")
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    idx, best = _choose_best_record(archive["records"])

    cfg_state = archive["config"]["season_state"]
    state = SeasonState(
        current_day=current_day,
        season_start_day=int(cfg_state["season_start_day"]),
        season_end_day=int(cfg_state["season_end_day"]),
    )

    progress("[operate] loading climate history")
    climate_history = read_climate_history(climate_history_path)
    if randomize_future:
        progress("[operate] generating randomized future climate regime")
        climate_history = generate_random_future_climate(climate_history, state, seed=random_seed)

    progress("[operate] computing optimal-vs-realized regime gap")
    decision = CandidateDecision(**best["decision"])
    full_target, realized_past, required_future = compute_target_triplet(climate_history, state, decision)
    full_target_d = aggregation_dict(full_target)
    past_d = aggregation_dict(realized_past)
    future_d = aggregation_dict(required_future)
    delta = {
        key: float(full_target_d[key]) - float(past_d.get(key, 0.0))
        for key in full_target_d
        if key != "n_days"
    }
    delta["n_days"] = int(full_target_d["n_days"] - past_d.get("n_days", 0))

    notes = [
        "required_future_regime represents the cumulative regime still needed from now until season end.",
        "delta_to_optimal_full_regime equals optimal full-season target minus realized past regime.",
        "The selected target is the best feasible archive point under a lexicographic ranking on meat production then feed efficiency.",
        f"random_future_generation={'enabled' if randomize_future else 'disabled'}",
    ]
    report = ExploitationReport(
        season_state=asdict(state),
        current_target_id=int(idx),
        objective_estimate={
            "beef_production_herd_kg": float(best["objectives"][0]),
            "feed_efficiency_herd_g_per_kg_dm": float(best["objectives"][1]),
        },
        optimal_full_cumulative_regime=full_target_d,
        realized_past_regime=past_d,
        required_future_regime=future_d,
        delta_to_optimal_full_regime=delta,
        recommended_decision=best["decision"],
        notes=notes,
    )
    progress(_format_progress("operate", 1, 1, perf_counter() - start_time))
    progress("[operate] completed successfully")
    return report
