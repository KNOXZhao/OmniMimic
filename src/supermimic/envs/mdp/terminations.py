"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25, state_source: str = "post_physics") -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    if state_source == "post_physics":
        height = env.sim.data.qpos[:, 2] - env.scene.env_origins[:, 2]
        return (cmd._ref.root_pos_w[:, 2] - height).abs() > threshold
    if state_source == "legacy":
        return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold
    raise ValueError(f"Unknown termination state source: {state_source}")


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8, state_source: str = "post_physics") -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    if state_source == "post_physics":
        root_quat = env.sim.data.qpos[:, 3:7]
    elif state_source == "legacy":
        root_quat = cmd.robot_root_quat_w
    else:
        raise ValueError(f"Unknown termination state source: {state_source}")
    rob = quat_apply_inverse(root_quat, g)[:, 2]
    return (ref - rob).abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end
