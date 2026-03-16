from __future__ import annotations

import json
from pathlib import Path

from herd_optimizer.reporting import create_operation_report, create_training_report


def test_create_training_report(tmp_path: Path) -> None:
    archive = {
        'config': {'season_state': {'current_day': 120, 'season_start_day': 1, 'season_end_day': 365}},
        'records': [
            {'objectives': [500.0, 50.0], 'constraints': [0.0, 0.0]},
            {'objectives': [510.0, 52.0], 'constraints': [0.0, 0.0]},
            {'objectives': [505.0, 51.0], 'constraints': [1.0, 0.0]},
        ],
    }
    archive_path = tmp_path / 'archive.json'
    archive_path.write_text(json.dumps(archive), encoding='utf-8')
    out = tmp_path / 'training_report.png'
    created = create_training_report(archive_path, out)
    assert created.exists()
    assert created.stat().st_size > 0


def test_create_operation_report(tmp_path: Path) -> None:
    payload = {
        'season_state': {'current_day': 180, 'season_start_day': 1, 'season_end_day': 365},
        'current_target_id': 5,
        'optimal_full_cumulative_regime': {'rad': 10, 'mint': 20, 'maxt': 30, 'vpr': 40, 'wind': 50, 'rain': 60, 'okta': 70, 'n_days': 365},
        'realized_past_regime': {'rad': 4, 'mint': 5, 'maxt': 6, 'vpr': 7, 'wind': 8, 'rain': 9, 'okta': 10, 'n_days': 180},
        'required_future_regime': {'rad': 6, 'mint': 15, 'maxt': 24, 'vpr': 33, 'wind': 42, 'rain': 51, 'okta': 60, 'n_days': 185},
        'delta_to_optimal_full_regime': {'rad': 6, 'mint': 15, 'maxt': 24, 'vpr': 33, 'wind': 42, 'rain': 51, 'okta': 60, 'n_days': 185},
    }
    operation_path = tmp_path / 'operation.json'
    operation_path.write_text(json.dumps(payload), encoding='utf-8')
    out = tmp_path / 'operation_report.png'
    created = create_operation_report(operation_path, out)
    assert created.exists()
    assert created.stat().st_size > 0
