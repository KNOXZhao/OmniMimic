"""Shared clock and observable simulation telemetry for the matched experiment.

No CUDA initialization occurs on import. Overflow counts are world/physics-substep
observations, not printf counts. A world with two overflow bits contributes once
to each corresponding bit. Simulator flags are sampled before environment reset.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

OVERFLOW_NAMES = (
    "NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD",
    "CONTACT_MATCH", "NVMAX", "EPA_HORIZON", "ITERATIONS", "LS_ITERATIONS", "TACTILE",
)
CAPACITY_MASK = 15  # NEFC | NJMAX_NNZ | BROADPHASE | NARROWPHASE


class OverflowCounter:
    def __init__(self, overflow: torch.Tensor, nefc: torch.Tensor,
                 nacon: torch.Tensor | None = None, ncollision: torch.Tensor | None = None):
        if overflow.ndim != 1 or nefc.shape != overflow.shape:
            raise ValueError("overflow/nefc must have one entry per simulation world")
        self.flags, self.nefc = overflow, nefc
        self.nacon, self.ncollision = nacon, ncollision
        self.masks = (2 ** torch.arange(len(OVERFLOW_NAMES), device=overflow.device)).to(torch.int32)
        self.counts = torch.zeros(len(OVERFLOW_NAMES), dtype=torch.int64, device=overflow.device)
        self.any_count = torch.zeros((), dtype=torch.int64, device=overflow.device)
        self.max_nefc = torch.zeros((), dtype=torch.int64, device=overflow.device)
        self.max_nacon = torch.zeros_like(self.max_nefc)
        self.max_ncollision = torch.zeros_like(self.max_nefc)
        self.capacity_seen = torch.zeros((), dtype=torch.int32, device=overflow.device)
        self.physics_substeps = 0

    def before_step(self):
        # Clear only diagnostic state, never a physical state or solver input.
        self.flags.zero_()

    def after_step(self):
        self.counts.add_(((self.flags[:, None] & self.masks) != 0).sum(0))
        self.any_count.add_((self.flags != 0).sum())
        self.max_nefc.copy_(torch.maximum(self.max_nefc, self.nefc.max().to(torch.int64)))
        if self.nacon is not None:
            self.max_nacon.copy_(torch.maximum(self.max_nacon, self.nacon.max().to(torch.int64)))
        if self.ncollision is not None:
            self.max_ncollision.copy_(torch.maximum(self.max_ncollision, self.ncollision.max().to(torch.int64)))
        self.capacity_seen.bitwise_or_(((self.flags & CAPACITY_MASK) != 0).any().to(torch.int32))
        self.physics_substeps += 1

    def snapshot(self):
        return {
            "schema_version": 1,
            "scope": "per-rank world/physics-substep flags, sampled immediately after each sim.step before resets",
            "physics_substeps": self.physics_substeps,
            "world_substep_observations": self.physics_substeps * self.flags.numel(),
            "any_flag_world_substeps": int(self.any_count),
            "max_nefc_observed": int(self.max_nefc),
            "max_nacon_observed": int(self.max_nacon),
            "max_ncollision_observed": int(self.max_ncollision),
            "strict_capacity_mask": CAPACITY_MASK,
            "capacity_overflow_seen": bool(self.capacity_seen),
            "by_flag_world_substeps": dict(zip(OVERFLOW_NAMES, self.counts.cpu().tolist())),
            "solver_limit_flags_are_not_capacity_overflows": ["ITERATIONS", "LS_ITERATIONS"],
        }


def install_overflow_counter(sim):
    """Hook mjlab Simulation.step; return the same monitor for either trainer."""
    if hasattr(sim, "matched_overflow_counter"):
        raise ValueError("overflow counter already installed")
    counter = OverflowCounter(sim.data.overflow[:], sim.data.nefc[:],
                              sim.data.nacon[:], sim.data.ncollision[:])
    original_step = sim.step

    def counted_step(*args, **kwargs):
        counter.before_step()
        result = original_step(*args, **kwargs)
        counter.after_step()
        return result

    sim.step = counted_step
    sim.matched_overflow_counter = counter
    return counter


def capacity_overflow_on_any_rank(counter: OverflowCounter, world: int) -> bool:
    """Synchronize the abort decision before either trainer starts optimizer work."""
    flag = counter.capacity_seen.clone()
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def nonfinite_metrics(metrics: dict) -> dict[str, str]:
    """Check actual numeric update outputs, excluding console-only placeholders."""
    bad = {}
    for name, value in metrics.items():
        number = float(value)
        if not math.isfinite(number):
            bad[name] = "NaN" if math.isnan(number) else ("+Inf" if number > 0 else "-Inf")
    return bad


def any_rank_flag(value: bool, world: int, device) -> bool:
    flag = torch.tensor([int(value)], device=device, dtype=torch.int32)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_fingerprints(modules: dict) -> dict:
    """Hash exact model parameters/buffers; intentionally exclude rank-local sampler."""
    result = {}
    for label, module in modules.items():
        digest = hashlib.sha256()
        nonfinite = []
        state = module.state_dict()
        for name, tensor in sorted(state.items()):
            value = tensor.detach().cpu().contiguous()
            if not bool(torch.isfinite(value).all()):
                nonfinite.append(name)
            digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[label] = {"sha256": digest.hexdigest(), "tensor_count": len(state),
                         "nonfinite": nonfinite}
    return result


def rank0_deadline_reached(elapsed_seconds: float, max_hours: float | None,
                           *, rank: int, world: int, device) -> bool:
    if max_hours is None:
        return False
    stop = rank == 0 and elapsed_seconds >= max_hours * 3600
    if world > 1:
        flag = torch.tensor([int(stop)], device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        stop = bool(flag.item())
    return stop


def learning_rate_step(iteration: int, credited_collection_steps: int, clock: str) -> int:
    if clock == "iteration":
        return iteration
    if clock == "credited_collection_steps":
        # AllMimic uses its pre-increment main-loop counter for the current update.
        return max(0, credited_collection_steps - 1)
    raise ValueError(f"unknown learning-rate clock: {clock}")
