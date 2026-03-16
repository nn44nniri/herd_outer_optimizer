from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLIMATE_KEYS = ["rad", "mint", "maxt", "vpr", "wind", "rain", "okta"]


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out



def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))



def create_training_report(archive_json: str | Path, output_png: str | Path) -> Path:
    archive = _load_json(archive_json)
    records = archive.get('records', [])
    if not records:
        raise ValueError('optimization archive contains no records')

    df = pd.DataFrame(
        {
            'trial': np.arange(1, len(records) + 1, dtype=int),
            'meat': [float(r['objectives'][0]) for r in records],
            'feed_eff': [float(r['objectives'][1]) for r in records],
            'failure': [float(r['constraints'][0]) for r in records],
            'neg_weight': [float(r['constraints'][1]) for r in records],
        }
    )
    df['best_meat_so_far'] = df['meat'].cummax()
    df['best_feed_eff_so_far'] = df['feed_eff'].cummax()
    feasible = (df['failure'] <= 0.0) & (df['neg_weight'] <= 0.0)

    out = _ensure_parent(output_png)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)

    ax = axes[0]
    ax.plot(df['trial'], df['meat'], marker='o', label='Beef production')
    ax.plot(df['trial'], df['best_meat_so_far'], linestyle='--', label='Best so far')
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('kg herd beef')
    ax.set_title('Training objective: beef production')
    ax.legend()

    ax = axes[1]
    ax.plot(df['trial'], df['feed_eff'], marker='o', label='Feed efficiency')
    ax.plot(df['trial'], df['best_feed_eff_so_far'], linestyle='--', label='Best so far')
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('g kg-1 DM')
    ax.set_title('Training objective: herd feed efficiency')
    ax.legend()

    ax = axes[2]
    colors = np.where(feasible, 1.0, 0.0)
    ax.scatter(df['meat'], df['feed_eff'], c=colors)
    for _, row in df.iterrows():
        ax.annotate(int(row['trial']), (row['meat'], row['feed_eff']), fontsize=8)
    ax.set_xlabel('Beef production (kg herd)')
    ax.set_ylabel('Feed efficiency (g kg-1 DM)')
    ax.set_title('Archive objective trade-off (bright=feasible, dark=infeasible)')

    fig.suptitle('herd_optimizer training report', fontsize=14)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out



def create_operation_report(operation_json: str | Path, output_png: str | Path) -> Path:
    report = _load_json(operation_json)
    optimal = report['optimal_full_cumulative_regime']
    past = report['realized_past_regime']
    future = report['required_future_regime']
    delta = report['delta_to_optimal_full_regime']

    labels = CLIMATE_KEYS
    optimal_vals = np.array([float(optimal[k]) for k in labels], dtype=float)
    past_vals = np.array([float(past[k]) for k in labels], dtype=float)
    future_vals = np.array([float(future[k]) for k in labels], dtype=float)
    delta_vals = np.array([float(delta[k]) for k in labels], dtype=float)
    ratio_vals = np.divide(
        past_vals,
        optimal_vals,
        out=np.zeros_like(past_vals),
        where=np.abs(optimal_vals) > 1e-12,
    )

    out = _ensure_parent(output_png)
    fig, axes = plt.subplots(len(labels), 3, figsize=(16, 3.8 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.array([axes])

    for i, label in enumerate(labels):
        ax0, ax1, ax2 = axes[i]

        ax0.bar(['optimal', 'past', 'future'], [optimal_vals[i], past_vals[i], future_vals[i]])
        ax0.set_title(f'{label}: cumulative regime')
        ax0.set_ylabel('Cumulative value')
        ax0.tick_params(axis='x', rotation=20)

        ax1.bar(['gap'], [delta_vals[i]])
        ax1.set_title(f'{label}: remaining gap')
        ax1.set_ylabel('Gap to optimum')
        ax1.tick_params(axis='x', rotation=20)

        progress_pct = float(ratio_vals[i] * 100.0)
        ax2.bar(['realized %'], [progress_pct])
        ax2.set_title(f'{label}: progress at day {report["season_state"]["current_day"]}')
        ax2.set_ylabel('% realized')
        ax2.set_ylim(0.0, max(110.0, progress_pct + 10.0))
        ax2.tick_params(axis='x', rotation=20)

    fig.suptitle(
        'herd_optimizer operation comparison report\n'
        f'separate parameter windows | target id={report["current_target_id"]}',
        fontsize=14,
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
