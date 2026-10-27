"""Pure tensor helpers for finite-horizon motion reset sampling.

Attainment is discounted reward divided by the reward available over a horizon
chosen from the reference and configured episode limit *before* failure. The
normalizer must never use the number of steps the robot actually survived.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

import torch


def sampler_layout_sha256(clips: Iterable[tuple[str, int, int]], bin_frames: int) -> str:
    """Identify bin layout using ordered (name, start, length) metadata.

    This is intentionally not a motion-content hash: modified poses with the
    same names and frame counts require separate dataset provenance checks.
    """
    payload = {
        "bin_frames": int(bin_frames),
        "clips": [[str(name), int(start), int(length)] for name, start, length in clips],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def eligible_start_ranges(
    bin_start: torch.Tensor,
    bin_end: torch.Tensor,
    clip_last: torch.Tensor,
    min_start_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exclusive bin ends and eligibility after reserving reference steps.

    Resetting to frame f provides ``clip_last - f`` distinct reference transitions.
    The simulator's extra final-pose hold is deliberately not counted. Short clips
    and partial tail bins can therefore have no eligible starts.
    """
    if min_start_steps < 1:
        raise ValueError("min_start_steps must be at least 1")
    ends = torch.minimum(bin_end, clip_last - min_start_steps + 1)
    ends = torch.maximum(ends, bin_start)
    return ends, ends > bin_start


def discounted_capacity(
    planned_steps: torch.Tensor, gamma: float, max_step_reward: float
) -> torch.Tensor:
    """Maximum finite discounted reward; zero for a zero-length horizon."""
    if not 0 < gamma <= 1 or not math.isfinite(max_step_reward) or max_step_reward <= 0:
        raise ValueError("require 0 < gamma <= 1 and finite max_step_reward > 0")
    steps = planned_steps.to(torch.float64).clamp_min(0)
    if gamma == 1:
        return steps * max_step_reward
    # expm1 avoids cancellation for gamma close to one or short horizons.
    return -torch.expm1(steps * math.log(gamma)) * (max_step_reward / (1 - gamma))


def normalized_attainment(
    returns: torch.Tensor, planned_steps: torch.Tensor, gamma: float, max_step_reward: float
) -> torch.Tensor:
    """Bounded attainment: perfect=1, zero/negative return=0.

    Zero horizons are not observations and are filtered by the caller. Returning
    zero for them keeps masked tensor operations finite. Nonfinite reward values
    remain nonfinite so the estimator can ignore invalid observations.
    """
    capacity = discounted_capacity(planned_steps, gamma, max_step_reward)
    score = returns.double() / capacity.clamp_min(torch.finfo(torch.float64).tiny)
    score = torch.where(capacity > 0, score, torch.zeros_like(score))
    return torch.where(torch.isfinite(score), score.clamp(0, 1), score)


def update_bin_estimates(
    estimates: torch.Tensor,
    seen: torch.Tensor,
    bins: torch.Tensor,
    scores: torch.Tensor,
    ema_alpha: float,
) -> None:
    """Aggregate a completed batch and update in place, initializing first visits."""
    if not 0 < ema_alpha <= 1:
        raise ValueError("ema_alpha must be in (0, 1]")
    finite = torch.isfinite(scores)
    bins, scores = bins[finite], scores[finite].to(estimates.dtype)
    sums = torch.zeros_like(estimates).index_add_(0, bins, scores)
    counts = torch.zeros_like(estimates).index_add_(0, bins, torch.ones_like(scores))
    touched = counts > 0
    means = sums[touched] / counts[touched]
    estimates[touched] = torch.where(
        seen[touched],
        (1 - ema_alpha) * estimates[touched] + ema_alpha * means,
        means,
    )
    seen[touched] = True


def reset_probabilities(
    attainment: torch.Tensor,
    memory: torch.Tensor,
    eligible: torch.Tensor,
    *,
    softmax_temp: float = 25.0,
    uniform_ratio: float = 0.5,
    hard_ratio: float = 0.4,
    memory_ratio: float = 0.1,
    memory_alpha: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked 50/40/10 mixture, returning probabilities and updated memory.

    All three components have support only on eligible bins. Memory is renormalized
    after masking, including when eligibility changes on resume. Nonfinite stored
    values cannot poison multinomial probabilities.
    """
    ratios = (uniform_ratio, hard_ratio, memory_ratio)
    if (not all(math.isfinite(v) and v >= 0 for v in ratios) or sum(ratios) <= 0
            or not math.isfinite(softmax_temp) or softmax_temp < 0
            or not 0 <= memory_alpha <= 1):
        raise ValueError("invalid reset mixture weights, temperature, or memory alpha")
    if not bool(eligible.any()):
        raise ValueError("No motion reset has the required remaining reference steps")
    uniform = eligible.to(attainment.dtype)
    uniform = uniform / uniform.sum()
    score = torch.nan_to_num(attainment, nan=1.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    hard = torch.softmax((-softmax_temp * score).masked_fill(~eligible, -torch.inf), dim=0)
    old_memory = torch.nan_to_num(memory, nan=0.0, posinf=0.0, neginf=0.0)
    old_memory = old_memory.clamp_min(0) * eligible
    # Also avoid overflow when individually finite but corrupted entries sum
    # beyond the dtype's range (including when memory_alpha is zero).
    old_memory = old_memory / old_memory.amax().clamp_min(1.0)
    mass = old_memory.sum()
    old_memory = torch.where(mass > 0, old_memory / mass.clamp_min(1e-30), uniform)
    new_memory = (1 - memory_alpha) * old_memory + memory_alpha * hard
    new_memory = new_memory / new_memory.sum()
    probabilities = uniform_ratio * uniform + hard_ratio * hard + memory_ratio * new_memory
    return probabilities / probabilities.sum(), new_memory


def sampled_horizon_mass(
    probabilities: torch.Tensor,
    bin_start: torch.Tensor,
    draw_end: torch.Tensor,
    clip_last: torch.Tensor,
    max_remaining_steps: int,
    *,
    frame_cap: torch.Tensor | None = None,
    capped_clip_last: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact mass of integer frame draws with at most the given reference steps.

    Bins use half-open [start, end) intervals. Optional frame caps describe the
    legacy post-draw clamp; capped_clip_last also handles its uniform sampler
    crossing into a previous clip when a clip is shorter than three frames.
    This function does not sample or mutate memory.
    """
    width = (draw_end - bin_start).clamp_min(0)
    uncapped_end = draw_end if frame_cap is None else torch.minimum(draw_end, frame_cap + 1)
    uncapped_end = torch.maximum(uncapped_end, bin_start)
    first_tail_frame = torch.maximum(bin_start, clip_last - max_remaining_steps)
    count = (uncapped_end - first_tail_frame).clamp_min(0)
    if frame_cap is not None:
        cap_last = clip_last if capped_clip_last is None else capped_clip_last
        count = count + (draw_end - uncapped_end).clamp_min(0) * (
            cap_last - frame_cap <= max_remaining_steps
        )
    fraction = count.double() / width.clamp_min(1).double()
    return (probabilities.double() * fraction).sum()
