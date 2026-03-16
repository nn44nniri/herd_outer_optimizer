# herd_outer_optimizer

This package provides an outer-loop livestock optimizer around `LiGAPSBeef20180301_herd_worked.py`.

## What is included

- `herd_optimizer/ligaps_sync.py`: synchronization layer that patches the original monolithic simulator at runtime and converts its printed table into machine-readable JSON.
- `herd_optimizer/climate.py`: climate-history loading, seasonal slicing, cumulative aggregation, and future-climate adjustment logic.
- `herd_optimizer/botorch_loop.py`: constrained multi-objective BoTorch loop using qNEHVI.
- `herd_optimizer/workflow.py`: training workflow and exploitation-phase JSON report generation.
- `LiGAPSBeef20180301_herd_sync.py`: standalone synchronization entrypoint for the original simulator.
- `assets/`: bundled copies of the original simulator and France climate file.

## Optimization framing

The optimizer maximizes two objectives:

1. herd-level beef production
2. herd-level feed efficiency as a proxy for lower resource consumption

The optimization is time-conditioned. At exploitation time, the package computes:

- the cumulative climatic regime already realized,
- the optimal cumulative full-season target associated with the selected archive point,
- the remaining future cumulative regime still required to approach that optimum.

## Required dependencies

Runtime training requires:

- Python 3.10+
- `numpy`
- `pandas`
- `torch`
- `botorch`
- `gpytorch`
- `matplotlib`

The source code is import-safe if BoTorch is not installed, but the training command will raise a clear error until those packages are available.

## Example training command

```bash
python -m herd_optimizer.cli train \
  --simulator ./assets/LiGAPSBeef20180301_herd_worked.py \
  --climate ./assets/FRACHA19982012.csv \
  --output-dir ./runs/france_case1 \
  --current-day 120 \
  --season-end-day 365 \
  --n-initial 4 \
  --n-iterations 4 \
  --batch-size 1 \
  --genotype 1 \
  --location FRANCE1 \
  --housing1 0 --housing2 1 --housing3 0 \
  --feed1 20 --feed2 0 --feednr 1 \
  --slaughter-weight 936.1493
```

## Example exploitation command

```bash
python -m herd_optimizer.cli operate \
  --archive ./runs/france_case1/optimization_archive.json \
  --climate ./assets/FRACHA19982012.csv \
  --current-day 180 \
  --random-future \
  --random-seed 42 \
  --output-json ./runs/france_case1/operation_day_180.json
```

## Notes

- The original LiGAPS-Beef script is not imported directly because it executes at import time. The synchronization layer uses a temporary patched copy and a subprocess.
- The current design targets the France weather file bundled in this archive. The same interface can be used with other climate histories if the CSV schema matches the expected columns.
- The constraints currently include simulator success and non-negative slaughter weight. You can extend this to add feed-budget or biological-feasibility constraints.


In exploitation mode, `--random-future` bootstrap-generates the remaining future climate segment from the historical seasonal window and feeds that randomized future climate to the optimizer report logic.


## Example graphical reports

Create a training report from the generated archive JSON:

```bash
python -m herd_optimizer.cli report-train \
  --archive ./runs/france_case1/optimization_archive.json \
  --output-png ./runs/france_case1/training_report.png
```

Create a comparative exploitation report from the generated operation JSON:

```bash
python -m herd_optimizer.cli report-operate \
  --operation-json ./runs/france_case1/operation_day_180.json \
  --output-png ./runs/france_case1/operation_day_180_report.png
```

The training report shows the objective history and archive trade-off.
The operation report compares the optimal full-season cumulative regime, the realized past regime, and the remaining required future regime.
