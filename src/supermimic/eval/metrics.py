"""Explicit benchmark return denominators, with no simulator or tensor dependencies.

The paper's prose does not fully specify how an early failure enters the return
denominator. We report both conventions. The primary full-clip metric counts the
unexecuted part of a failed clip as zero reward; the conditional metric describes
only the steps that ran. Neither score is clipped, so penalties remain visible.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ReturnScores:
    planned_steps: int
    executed_steps: int
    full_clip_norm_return: float
    executed_steps_norm_return: float
    completion_fraction: float


def return_scores(
    reward_sum: float, frames: int, executed_steps: int, max_step_reward: float = 0.1
) -> ReturnScores:
    """Score transitions from frame 0 through frame ``frames - 1``.

    ``executed_steps`` counts rewarded transitions inside this horizon, excluding
    any simulator bookkeeping step that holds the final target before auto-reset.
    """
    if frames < 2:
        raise ValueError("a benchmark clip must contain at least two frames")
    planned = frames - 1
    if not 0 <= executed_steps <= planned:
        raise ValueError("executed_steps must be within the clip's transition horizon")
    if not isfinite(max_step_reward) or max_step_reward <= 0:
        raise ValueError("max_step_reward must be positive and finite")
    if not isfinite(reward_sum):
        raise ValueError("reward_sum must be finite")
    if executed_steps == 0 and reward_sum != 0:
        raise ValueError("zero executed steps cannot have nonzero accumulated reward")
    return ReturnScores(
        planned_steps=planned,
        executed_steps=executed_steps,
        full_clip_norm_return=reward_sum / (max_step_reward * planned),
        executed_steps_norm_return=(
            reward_sum / (max_step_reward * executed_steps) if executed_steps else 0.0
        ),
        completion_fraction=executed_steps / planned,
    )
