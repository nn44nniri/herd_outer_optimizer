from __future__ import annotations

from dataclasses import asdict
from typing import Iterable
import warnings

import numpy as np

from .schemas import CandidateDecision, OptimizationConfig, OptimizationRecord


def _import_botorch() -> None:
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "BoTorch, GPyTorch, and PyTorch must be installed to train the herd optimizer."
        ) from exc


def sobol_initial_design(config: OptimizationConfig) -> np.ndarray:
    _import_botorch()
    import torch
    from torch.quasirandom import SobolEngine

    dim = len(config.lower_bounds)
    sobol = SobolEngine(dimension=dim, scramble=True, seed=config.random_seed)
    X_unit = sobol.draw(config.n_initial).cpu().numpy()
    lb = np.asarray(config.lower_bounds, dtype=float)
    ub = np.asarray(config.upper_bounds, dtype=float)
    return lb + (ub - lb) * X_unit


def _array_has_variation(values: np.ndarray, tol: float = 1e-12) -> bool:
    if values.size == 0:
        return False
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    for col in range(arr.shape[1]):
        finite = arr[np.isfinite(arr[:, col]), col]
        if finite.size >= 2 and np.nanstd(finite) > tol:
            return True
    return False


def _safe_column(values: np.ndarray, *, jitter_scale: float = 1e-6) -> np.ndarray:
    col = np.asarray(values, dtype=float).reshape(-1, 1).copy()
    mask = ~np.isfinite(col[:, 0])
    if mask.any():
        finite = col[~mask, 0]
        fill = float(np.nanmean(finite)) if finite.size else 0.0
        col[mask, 0] = fill
    std = float(np.std(col[:, 0]))
    if std <= 1e-12:
        n = col.shape[0]
        if n > 1:
            ramp = np.linspace(-0.5, 0.5, n, dtype=float).reshape(-1, 1)
            scale = max(abs(float(np.mean(col[:, 0]))), 1.0) * jitter_scale
            col = col + ramp * scale
    return col


class BotorchMOBO:
    def __init__(self, config: OptimizationConfig) -> None:
        _import_botorch()
        self.config = config
        self._fallback_calls = 0
        self.last_strategy = 'uninitialized'

    def _fallback_batch(self) -> np.ndarray:
        self.last_strategy = 'fallback_random'
        seed = int(self.config.random_seed + 1000 + self._fallback_calls)
        self._fallback_calls += 1
        rng = np.random.default_rng(seed)
        lb = np.asarray(self.config.lower_bounds, dtype=float)
        ub = np.asarray(self.config.upper_bounds, dtype=float)
        unit = rng.random((self.config.batch_size, len(lb)))
        return lb + (ub - lb) * unit

    def propose(
        self,
        train_X: np.ndarray,
        train_Y: np.ndarray,
        train_C: np.ndarray,
    ) -> np.ndarray:
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

        train_X = np.asarray(train_X, dtype=float)
        train_Y = np.asarray(train_Y, dtype=float)
        train_C = np.asarray(train_C, dtype=float)

        if train_X.ndim != 2 or train_X.shape[0] < 2:
            return self._fallback_batch()
        if not _array_has_variation(train_Y):
            return self._fallback_batch()

        stacked = np.concatenate([_safe_column(train_Y[:, i]) for i in range(train_Y.shape[1])] + [_safe_column(train_C[:, i]) for i in range(train_C.shape[1])], axis=1)
        train_X_t = torch.tensor(train_X, dtype=torch.double)
        train_Y_t = torch.tensor(stacked, dtype=torch.double)

        models = []
        dim = train_X.shape[1]
        for idx in range(train_Y_t.shape[1]):
            yi = train_Y_t[:, idx:idx + 1]
            models.append(
                SingleTaskGP(
                    train_X_t,
                    yi,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'Data \(outcome observations\) is not standardized.*')
            fit_gpytorch_mll(mll)

        ref_point = list(self.config.ref_point) + [0.5, 0.5]
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))

        def _constraint_0(Z: torch.Tensor) -> torch.Tensor:
            return Z[..., 2]

        def _constraint_1(Z: torch.Tensor) -> torch.Tensor:
            return Z[..., 3]

        self.last_strategy = 'qlognehvi'
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X_t,
            sampler=sampler,
            objective=None,
            constraints=[_constraint_0, _constraint_1],
            prune_baseline=True,
        )

        bounds = torch.tensor(
            np.vstack([self.config.lower_bounds, self.config.upper_bounds]),
            dtype=torch.double,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unable to find non-zero acquisition function values.*')
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=self.config.batch_size,
                num_restarts=10,
                raw_samples=128,
            )
        cand_np = candidates.detach().cpu().numpy()
        if not np.all(np.isfinite(cand_np)):
            return self._fallback_batch()
        return cand_np


def build_record(
    *,
    x: Iterable[float],
    objectives: list[float],
    constraints: list[float],
    decision: CandidateDecision,
    result: dict,
    full_target: dict,
    realized_past: dict,
    required_future: dict,
) -> OptimizationRecord:
    return OptimizationRecord(
        x=[float(v) for v in x],
        objectives=[float(v) for v in objectives],
        constraints=[float(v) for v in constraints],
        decision=asdict(decision),
        result=result,
        climate_target_full=full_target,
        climate_realized_past=realized_past,
        climate_required_future=required_future,
    )
