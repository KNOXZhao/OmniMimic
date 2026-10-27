"""Analytic BeyondMimic tracking reward in torch (plan D4/D7).

One differentiable implementation shared by the simulator and imagination
(world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
``tracking_reward_terms`` returns weighted, dt-scaled values. The simulator's
``PostPhysicsTrackingReward`` requests unit weights/dt, then lets RewardManager
apply the configured weights and timestep once. ``reward_mode="legacy"`` keeps
the historical mjlab reward functions for old-checkpoint comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms


class _PostPhysicsRewardCache:
    """One FK/reward evaluation shared by all eight manager terms.

    MuJoCo's derived body quantities lag the final integration substep. Read raw
    qpos/qvel instead, and re-anchor against that same state. The current command's
    cached *reference* bodies are valid; its cached robot-relative bodies are not.
    """

    def __init__(self, env, command_name: str, compile_rewards: bool):
        from supermimic.data.motion_convert import load_g1_model
        from supermimic.kinematics.g1_fk import G1Kinematics

        self.command = env.command_manager.get_term(command_name)
        self.fk = getattr(self.command.library, "_fk", None)
        if self.fk is None:
            self.fk = G1Kinematics(load_g1_model(), device=env.device)
        self.tracked_idx = self.fk.body_indices(self.command.cfg.body_names)
        self.anchor_index = self.command.cfg.body_names.index(self.command.cfg.anchor_body_name)
        # RewardManager applies the original term weights and dt exactly once.
        self.raw_cfg = TrackingRewardCfg(
            w_root_pos=1.0, w_root_ori=1.0, w_body_pos=1.0, w_body_ori=1.0,
            w_body_lin_vel=1.0, w_body_ang_vel=1.0, w_action_rate=1.0,
            w_joint_limit=1.0, dt=1.0,
        )
        self.compute = self._compute
        if compile_rewards and torch.device(env.device).type == "cuda":
            self.compute = torch.compile(self._compute, dynamic=False)
        self.computations = 0
        self.invalidate()

    def invalidate(self):
        self.key = None
        self.terms = None
        self.served = set()

    @staticmethod
    def _signature(tensor):
        # Warp updates do not increment Torch versions, hence the physics counter
        # below. Versions additionally catch manual StateIO/action writes in tests
        # and diagnostic code; inference tensors have no version counter.
        return (tensor.data_ptr(), None if tensor.is_inference() else tensor._version)

    def _compute(self, qpos, qvel, ref_pos, ref_quat, ref_lin, ref_ang, action, prev_action, limits):
        b = self.fk(qpos, qvel)
        i = self.tracked_idx
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"][:, i], robot_body_quat_w=b["quat"][:, i],
            robot_body_lin_vel_w=b["lin_vel"][:, i], robot_body_ang_vel_w=b["ang_vel"][:, i],
            ref_body_pos_w=ref_pos, ref_body_quat_w=ref_quat,
            ref_body_lin_vel_w=ref_lin, ref_body_ang_vel_w=ref_ang,
            anchor_index=self.anchor_index, action=action, prev_action=prev_action,
            joint_pos=qpos[:, 7:36], soft_limits=limits, cfg=self.raw_cfg,
        )

    @torch.no_grad()
    def get(self, env, term_name):
        cmd = self.command
        qpos, qvel = env.sim.data.qpos[:], env.sim.data.qvel[:]
        action, prev = env.action_manager.action, env.action_manager.prev_action
        key = (env._sim_step_counter, id(cmd._bodies), self._signature(cmd.frame),
               self._signature(qpos), self._signature(qvel),
               self._signature(action), self._signature(prev))
        # A repeated term starts a fresh evaluation even when called twice between
        # physics steps. Reset hooks invalidate all slots, including partial resets.
        if key != self.key or term_name in self.served:
            qpos = qpos.clone()
            qpos[:, :3] -= env.scene.env_origins
            ref = cmd._bodies  # Local-frame reference cache, never cached re-anchoring.
            self.terms = self.compute(qpos, qvel, ref.pos_w, ref.quat_w,
                                      ref.lin_vel_w, ref.ang_vel_w, action, prev,
                                      cmd.robot.data.soft_joint_pos_limits)
            self.key = key
            self.served.clear()
            self.computations += 1
        self.served.add(term_name)
        return self.terms[term_name]


class PostPhysicsTrackingReward:
    """RewardManager term selecting one raw term from a shared post-physics result."""

    def __init__(self, cfg, env):
        command_name = cfg.params.get("command_name", "motion")
        caches = getattr(env, "_post_physics_reward_caches", None)
        if caches is None:
            caches = env._post_physics_reward_caches = {}
        if command_name not in caches:
            caches[command_name] = _PostPhysicsRewardCache(
                env, command_name, cfg.params.get("compile_rewards", True)
            )
        self.cache = caches[command_name]

    def __call__(self, env, term_name: str, command_name: str = "motion", compile_rewards: bool = True):
        del command_name, compile_rewards
        return self.cache.get(env, term_name)

    def reset(self, env_ids=None):
        del env_ids
        self.cache.invalidate()
