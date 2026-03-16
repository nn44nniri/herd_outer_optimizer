from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .climate import aggregation_dict, compute_target_triplet, read_climate_history, save_adjusted_climate_csv
from .schemas import CandidateDecision, HerdCaseConfig, SeasonState, SimulationResult

JSON_MARKER = "__LIGAPS_JSON__"


class LiGAPSSynchronizer:
    """Synchronizes the monolithic LiGAPS-Beef script with the outer-loop optimizer."""

    def __init__(self, simulator_path: str | Path, climate_history_path: str | Path) -> None:
        self.simulator_path = Path(simulator_path)
        self.climate_history_path = Path(climate_history_path)
        if not self.simulator_path.exists():
            raise FileNotFoundError(self.simulator_path)
        if not self.climate_history_path.exists():
            raise FileNotFoundError(self.climate_history_path)

    def evaluate(
        self,
        decision: CandidateDecision,
        state: SeasonState,
        case: HerdCaseConfig,
        work_dir: str | Path | None = None,
    ) -> SimulationResult:
        history = read_climate_history(self.climate_history_path)
        full_target, realized_past, required_future = compute_target_triplet(history, state, decision)

        temp_context = tempfile.TemporaryDirectory(prefix="ligaps_sync_") if work_dir is None else None
        root = Path(temp_context.name) if temp_context is not None else Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)

        try:
            climate_name = "FRACHA19982012.csv" if case.location.upper().startswith("FRANCE") else "AUSTRALIA1992A.csv"
            climate_path = root / climate_name
            save_adjusted_climate_csv(history, state, decision, climate_path)

            patched_source = self._build_patched_source(case, climate_name)
            patched_script = root / self.simulator_path.name
            patched_script.write_text(patched_source, encoding="utf-8")

            start = time.perf_counter()
            process = subprocess.run(
                [os.environ.get("PYTHON", "python"), str(patched_script)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            runtime = time.perf_counter() - start
            result = self._parse_result(
                stdout=process.stdout,
                stderr=process.stderr,
                runtime_seconds=runtime,
                root=root,
                full_target=aggregation_dict(full_target),
                realized_past=aggregation_dict(realized_past),
                required_future=aggregation_dict(required_future),
                decision=asdict(decision),
                case=asdict(case),
            )
            return result
        finally:
            if temp_context is not None:
                temp_context.cleanup()

    def _build_patched_source(self, case: HerdCaseConfig, climate_name: str) -> str:
        source = self.simulator_path.read_text(encoding="utf-8", errors="ignore")
        source = source.replace("import matplotlib.pyplot as plt", "import matplotlib.pyplot as plt\nimport json")
        source = self._replace_literal(source, r"SHOW_PROGRESS\s*=\s*True", "SHOW_PROGRESS = False")
        source = self._replace_literal(source, r"DEBUG_LOOP\s*=\s*False", "DEBUG_LOOP = False")
        source = self._replace_block(
            source,
            r"ill_genotype\s*=\s*np\.array\(\[[\s\S]*?dtype=int\)",
            f"ill_genotype = np.array([{case.genotype}], dtype=int)",
        )
        source = self._replace_block(
            source,
            r"ill_slweight\s*=\s*np\.array\([\s\S]*?dtype=float\)",
            f"ill_slweight = np.array([{case.slaughter_weight}], dtype=float)",
        )
        source = self._replace_block(
            source,
            r"ill_location\s*=\s*\[[\s\S]*?\]\n",
            f"ill_location = ['{case.location}']\n",
        )
        source = self._replace_block(
            source,
            r"ill_housing1\s*=\s*np\.array\(\[[\s\S]*?dtype=int\)",
            f"ill_housing1 = np.array([{case.housing1}], dtype=int)",
        )
        source = self._replace_block(
            source,
            r"ill_housing2\s*=\s*np\.array\(\[[\s\S]*?dtype=int\)",
            f"ill_housing2 = np.array([{case.housing2}], dtype=int)",
        )
        source = self._replace_block(
            source,
            r"ill_housing3\s*=\s*np\.array\(\[[\s\S]*?dtype=int\)",
            f"ill_housing3 = np.array([{case.housing3}], dtype=int)",
        )
        source = self._replace_block(
            source,
            r"ill_f1\s*=\s*np\.array\(\[[\s\S]*?dtype=float\)",
            f"ill_f1 = np.array([{case.feed1}], dtype=float)",
        )
        source = self._replace_block(
            source,
            r"ill_f2\s*=\s*np\.array\(\[[\s\S]*?dtype=float\)",
            f"ill_f2 = np.array([{case.feed2}], dtype=float)",
        )
        source = self._replace_block(
            source,
            r"FEEDNR\s*=\s*np\.array\(\[[\s\S]*?dtype=int\)",
            f"FEEDNR = np.array([{case.feednr}], dtype=int)",
        )
        source = self._replace_block(source, r"case_ids\s*=\s*\[[^\]]*\]", "case_ids = [1]")
        source = source.replace('weather_file = BASE_DIR / "FRACHA19982012.csv"', f'weather_file = BASE_DIR / "{climate_name}"')
        source = source.replace('weather_file = BASE_DIR / "AUSTRALIA1992A.csv"', f'weather_file = BASE_DIR / "{climate_name}"')
        source = source.replace(
            'print(TABLEDATA)',
            'print(TABLEDATA)\n'
            'payload = {\n'
            '    "case_number": int(z),\n'
            '    "feed_efficiency_herd_g_per_kg_dm": float(TABLEDATA.loc["Feed efficiency herd unit (g beef kg-1 DM)", "Herd level"]),\n'
            '    "feed_efficiency_repr_g_per_kg_dm": float(TABLEDATA.loc["Feed efficiency repr. cow (g beef kg-1 DM)", "Herd level"]),\n'
            '    "feed_efficiency_bull_g_per_kg_dm": float(TABLEDATA.loc["Feed efficiency bull calf (g beef kg-1 DM)", "Herd level"]),\n'
            '    "feed_fraction_repr_cow": float(TABLEDATA.loc["Feed fraction repr. cow (-)", "Herd level"]),\n'
            '    "beef_production_herd_kg": float(TABLEDATA.loc["Beef production herd unit (kg)", "Herd level"]),\n'
            '    "beef_production_repr_cow_kg": float(TABLEDATA.loc["Beef production repr. cow (kg)", "Herd level"]),\n'
            '    "beef_production_bull_calf_kg": float(TABLEDATA.loc["Beef production bull calf (kg)", "Herd level"]),\n'
            '    "slaughter_weight_bull_calf_kg": float(TABLEDATA.loc["Slaughter weight bull calf (kg)", "Herd level"]),\n'
            '    "plot_path": str(BASE_DIR / f"ligaps_case_{z}_calves.png"),\n'
            '}\n'
            f'print("{JSON_MARKER}" + json.dumps(payload))',
            1,
        )
        return source

    @staticmethod
    def _replace_block(source: str, pattern: str, replacement: str) -> str:
        new_source, n = re.subn(pattern, replacement, source, count=1)
        if n != 1:
            raise ValueError(f"Pattern not found or ambiguous: {pattern}")
        return new_source

    @staticmethod
    def _replace_literal(source: str, pattern: str, replacement: str) -> str:
        return re.sub(pattern, replacement, source, count=1)

    @staticmethod
    def _parse_result(
        *,
        stdout: str,
        stderr: str,
        runtime_seconds: float,
        root: Path,
        full_target: dict[str, Any],
        realized_past: dict[str, Any],
        required_future: dict[str, Any],
        decision: dict[str, Any],
        case: dict[str, Any],
    ) -> SimulationResult:
        payload = None
        for line in stdout.splitlines():
            if line.startswith(JSON_MARKER):
                payload = json.loads(line[len(JSON_MARKER):])
        if payload is None:
            return SimulationResult(
                success=False,
                case_number=1,
                beef_production_herd_kg=float("nan"),
                beef_production_repr_cow_kg=float("nan"),
                beef_production_bull_calf_kg=float("nan"),
                feed_efficiency_herd_g_per_kg_dm=float("nan"),
                feed_efficiency_repr_g_per_kg_dm=float("nan"),
                feed_efficiency_bull_g_per_kg_dm=float("nan"),
                slaughter_weight_bull_calf_kg=float("nan"),
                feed_fraction_repr_cow=float("nan"),
                runtime_seconds=runtime_seconds,
                stdout=stdout,
                stderr=stderr,
                metadata={
                    "decision": decision,
                    "case": case,
                    "full_target": full_target,
                    "realized_past": realized_past,
                    "required_future": required_future,
                },
            )

        original_plot = Path(payload["plot_path"])
        archived_plot = None
        if original_plot.exists():
            archived_plot = root / f"archived_{original_plot.name}"
            shutil.copy2(original_plot, archived_plot)

        return SimulationResult(
            success=True,
            case_number=int(payload["case_number"]),
            beef_production_herd_kg=float(payload["beef_production_herd_kg"]),
            beef_production_repr_cow_kg=float(payload["beef_production_repr_cow_kg"]),
            beef_production_bull_calf_kg=float(payload["beef_production_bull_calf_kg"]),
            feed_efficiency_herd_g_per_kg_dm=float(payload["feed_efficiency_herd_g_per_kg_dm"]),
            feed_efficiency_repr_g_per_kg_dm=float(payload["feed_efficiency_repr_g_per_kg_dm"]),
            feed_efficiency_bull_g_per_kg_dm=float(payload["feed_efficiency_bull_g_per_kg_dm"]),
            slaughter_weight_bull_calf_kg=float(payload["slaughter_weight_bull_calf_kg"]),
            feed_fraction_repr_cow=float(payload["feed_fraction_repr_cow"]),
            runtime_seconds=runtime_seconds,
            stdout=stdout,
            stderr=stderr,
            plot_path=str(archived_plot) if archived_plot is not None else None,
            metadata={
                "decision": decision,
                "case": case,
                "full_target": full_target,
                "realized_past": realized_past,
                "required_future": required_future,
            },
        )
