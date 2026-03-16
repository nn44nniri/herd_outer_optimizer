from __future__ import annotations

import argparse
import json
from pathlib import Path

from herd_optimizer.ligaps_sync import LiGAPSSynchronizer
from herd_optimizer.schemas import CandidateDecision, HerdCaseConfig, SeasonState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize LiGAPSBeef20180301_herd_worked.py with the outer-loop optimizer")
    parser.add_argument("--simulator", required=True)
    parser.add_argument("--climate", required=True)
    parser.add_argument("--current-day", type=int, default=1)
    parser.add_argument("--season-end-day", type=int, default=365)
    parser.add_argument("--genotype", type=int, default=1)
    parser.add_argument("--location", default="FRANCE1")
    parser.add_argument("--housing1", type=int, default=0)
    parser.add_argument("--housing2", type=int, default=1)
    parser.add_argument("--housing3", type=int, default=0)
    parser.add_argument("--feed1", type=float, default=20.0)
    parser.add_argument("--feed2", type=float, default=0.0)
    parser.add_argument("--feednr", type=int, default=1)
    parser.add_argument("--slaughter-weight", type=float, default=936.1493)
    parser.add_argument("--rad-scale", type=float, default=1.0)
    parser.add_argument("--mint-shift", type=float, default=0.0)
    parser.add_argument("--maxt-shift", type=float, default=0.0)
    parser.add_argument("--vpr-scale", type=float, default=1.0)
    parser.add_argument("--wind-scale", type=float, default=1.0)
    parser.add_argument("--rain-scale", type=float, default=1.0)
    parser.add_argument("--okta-shift", type=float, default=0.0)
    parser.add_argument("--feed1-scale", type=float, default=1.0)
    parser.add_argument("--feed2-scale", type=float, default=1.0)
    parser.add_argument("--slweight-scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sync = LiGAPSSynchronizer(args.simulator, args.climate)
    decision = CandidateDecision(
        rad_scale=args.rad_scale,
        mint_shift=args.mint_shift,
        maxt_shift=args.maxt_shift,
        vpr_scale=args.vpr_scale,
        wind_scale=args.wind_scale,
        rain_scale=args.rain_scale,
        okta_shift=args.okta_shift,
        feed1_scale=args.feed1_scale,
        feed2_scale=args.feed2_scale,
        slweight_scale=args.slweight_scale,
    )
    case = HerdCaseConfig(
        genotype=args.genotype,
        location=args.location,
        housing1=args.housing1,
        housing2=args.housing2,
        housing3=args.housing3,
        feed1=args.feed1 * args.feed1_scale,
        feed2=args.feed2 * args.feed2_scale,
        feednr=args.feednr,
        slaughter_weight=args.slaughter_weight * args.slweight_scale,
    )
    state = SeasonState(current_day=args.current_day, season_end_day=args.season_end_day)
    result = sync.evaluate(decision=decision, state=state, case=case)
    print(json.dumps({
        "success": result.success,
        "case_number": result.case_number,
        "objectives": result.objective_vector(),
        "constraints": result.constraint_vector(),
        "plot_path": result.plot_path,
        "metadata": result.metadata,
    }, indent=2))


if __name__ == "__main__":
    main()
