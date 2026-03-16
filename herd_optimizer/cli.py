from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import HerdCaseConfig, OptimizationConfig, SeasonState
from .reporting import create_operation_report, create_training_report
from .workflow import generate_operation_report, train_optimizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Outer-loop herd optimizer for LiGAPS-Beef")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the outer-loop optimizer")
    train.add_argument("--simulator", required=True)
    train.add_argument("--climate", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--current-day", type=int, default=1)
    train.add_argument("--season-end-day", type=int, default=365)
    train.add_argument("--n-initial", type=int, default=6)
    train.add_argument("--n-iterations", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--genotype", type=int, default=1)
    train.add_argument("--location", default="FRANCE1")
    train.add_argument("--housing1", type=int, default=0)
    train.add_argument("--housing2", type=int, default=1)
    train.add_argument("--housing3", type=int, default=0)
    train.add_argument("--feed1", type=float, default=20.0)
    train.add_argument("--feed2", type=float, default=0.0)
    train.add_argument("--feednr", type=int, default=1)
    train.add_argument("--slaughter-weight", type=float, default=936.1493)

    operate = sub.add_parser("operate", help="Generate the exploitation-phase JSON report")
    operate.add_argument("--archive", required=True)
    operate.add_argument("--climate", required=True)
    operate.add_argument("--current-day", type=int, required=True)
    operate.add_argument("--output-json", required=False)
    operate.add_argument("--random-future", action="store_true", help="Bootstrap-generate a random future climate regime for exploitation")
    operate.add_argument("--random-seed", type=int, default=1234)

    report_train = sub.add_parser("report-train", help="Create a graphical training report from an optimization archive JSON")
    report_train.add_argument("--archive", required=True)
    report_train.add_argument("--output-png", required=True)

    report_operate = sub.add_parser("report-operate", help="Create a comparative graphical report from an operation JSON")
    report_operate.add_argument("--operation-json", required=True)
    report_operate.add_argument("--output-png", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        cfg = OptimizationConfig(
            simulator_path=Path(args.simulator),
            climate_history_path=Path(args.climate),
            output_dir=Path(args.output_dir),
            season_state=SeasonState(current_day=args.current_day, season_end_day=args.season_end_day),
            herd_case=HerdCaseConfig(
                genotype=args.genotype,
                location=args.location,
                housing1=args.housing1,
                housing2=args.housing2,
                housing3=args.housing3,
                feed1=args.feed1,
                feed2=args.feed2,
                feednr=args.feednr,
                slaughter_weight=args.slaughter_weight,
            ),
            n_initial=args.n_initial,
            n_iterations=args.n_iterations,
            batch_size=args.batch_size,
        )
        archive = train_optimizer(cfg)
        print(json.dumps({"n_records": len(archive["records"]), "output_dir": str(cfg.output_dir)}, indent=2))
        return

    if args.command == "report-train":
        output = create_training_report(args.archive, args.output_png)
        print(json.dumps({"report_png": str(output), "mode": "train"}, indent=2))
        return

    if args.command == "report-operate":
        output = create_operation_report(args.operation_json, args.output_png)
        print(json.dumps({"report_png": str(output), "mode": "operate"}, indent=2))
        return

    report = generate_operation_report(
        archive_path=args.archive,
        current_day=args.current_day,
        climate_history_path=args.climate,
        randomize_future=args.random_future,
        random_seed=args.random_seed,
    )
    payload = report.to_dict()
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
