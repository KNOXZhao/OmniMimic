"""Regression coverage for reward timing, re-anchoring and raw-state termination.

The CUDA case checks the reward returned by an actual env.step, before its target
advances. Comparing reward functions after sim.forward would miss the original
one-substep lag and stale command re-anchoring.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.managers.reward_manager import RewardManager

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.rewards import PostPhysicsTrackingReward
from supermimic.envs.mdp.terminations import bad_root_height, bad_root_tilt
from supermimic.envs.tracking_env_cfg import (
    ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES, SuperMimicEnvCfgArgs, make_env_cfg,
)
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_rotvec
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame


def _reference(cmd):
    b, r = cmd._bodies, cmd._ref
    return ReferenceFrame(b.pos_w.clone(), b.quat_w.clone(), b.lin_vel_w.clone(),
                          b.ang_vel_w.clone(), r.root_pos_w.clone(), r.root_quat_w.clone())


@pytest.fixture
def cpu_case():
    """Real MuJoCo integration, with different environment origins and rapid motion."""
    model = load_g1_model()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    states, stale_roots, stale_quats = [], [], []
    for i in range(4):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = (0.17 * i, -0.23 * i, 1.7 + 0.03 * i)
        data.qpos[3:7] = quat_from_rotvec(torch.tensor([0.1, -0.2, 0.7 * i])).numpy()
        data.qvel[:] = np.sin(np.arange(model.nv) + i) * 2.0
        data.qvel[:6] = (2.0, -1.0, 0.7, 1.0, -2.0, 4.0)
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        states.append(np.r_[data.qpos.copy(), data.qvel.copy()])
        stale_roots.append(data.xpos[1].copy())
        stale_quats.append(data.xquat[1].copy())
    state = torch.tensor(np.array(states), dtype=torch.float32)
    fk = G1Kinematics(model, device="cpu")
    limits = torch.tensor(model.jnt_range[1:], dtype=torch.float32)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, limits, state[0, 7:36])
    reference_state = state.clone()
    reference_state[:, :3] += torch.tensor([0.15, -0.22, -0.04])
    reference_state[:, 3:7] = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.25]])).expand(4, -1)
    bodies = heads.robot_bodies(reference_state)
    origins = torch.tensor([[0., 0., 0.], [3., -2., 0.7], [-3., 4., -0.2], [6., 1., 0.3]])
    cmd = SimpleNamespace(
        library=SimpleNamespace(_fk=fk),
        cfg=SimpleNamespace(body_names=TRACKED_BODIES, anchor_body_name=ANCHOR_BODY),
        frame=torch.zeros(4, dtype=torch.long),
        _bodies=SimpleNamespace(pos_w=bodies["pos"], quat_w=bodies["quat"],
                                lin_vel_w=bodies["lin_vel"], ang_vel_w=bodies["ang_vel"]),
        _ref=SimpleNamespace(root_pos_w=reference_state[:, :3], root_quat_w=reference_state[:, 3:7]),
        robot=SimpleNamespace(data=SimpleNamespace(
            soft_joint_pos_limits=limits.expand(4, -1, -1),
            gravity_vec_w=torch.tensor([[0., 0., -1.]]).expand(4, -1))),
        root_pos_w=reference_state[:, :3] + origins,
        root_quat_w=reference_state[:, 3:7],
        robot_root_pos_w=torch.tensor(np.array(stale_roots), dtype=torch.float32) + origins,
        robot_root_quat_w=torch.tensor(np.array(stale_quats), dtype=torch.float32),
    )
    qp = state[:, :36].clone()
    qp[:, :3] += origins
    env = SimpleNamespace(
        num_envs=4, device="cpu", max_episode_length_s=10.0, _sim_step_counter=4,
        scene=SimpleNamespace(env_origins=origins),
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qp, qvel=state[:, 36:].clone())),
        action_manager=SimpleNamespace(action=torch.full((4, 29), 0.2), prev_action=torch.zeros(4, 29)),
    )
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[]))
    manager = RewardManager(cfg.rewards, env)
    return env, cmd, heads, state, manager


def test_real_cpu_step_uses_post_integration_state_and_current_reanchor(cpu_case):
    env, cmd, heads, state, manager = cpu_case
    # mj_step really left the derived root state stale, so this exercises the gap.
    stale = cmd.robot_root_pos_w - env.scene.env_origins
    assert (stale - state[:, :3]).abs().max() > 0.005
    expected = heads.reward(state, env.action_manager.action, env.action_manager.prev_action, _reference(cmd))
    actual = manager.compute(dt=0.02)
    torch.testing.assert_close(actual, expected["total"], atol=2e-7, rtol=2e-5)
    for name in manager.active_terms:
        torch.testing.assert_close(manager.get_term(name), expected[name], atol=2e-7, rtol=2e-5)
    assert env._post_physics_reward_caches["motion"].computations == 1


def test_cache_recomputes_on_manual_state_action_and_reference_writes(cpu_case):
    env, cmd, _, _, manager = cpu_case
    term = manager.get_term_cfg("motion_global_root_pos").func
    first = term(env, "motion_global_root_pos").clone()
    assert term.cache.computations == 1
    # A different term after a manual write must not reuse the old result.
    env.sim.data.qpos[:, 2] += 0.3
    term(env, "motion_global_root_ori")
    assert term.cache.computations == 2
    assert not torch.allclose(first, term.cache.terms["motion_global_root_pos"])
    env.action_manager.action.add_(0.1)
    term(env, "action_rate_l2")
    assert term.cache.computations == 3
    cmd.frame.add_(1)
    term(env, "joint_limit")
    assert term.cache.computations == 4
    env._sim_step_counter += 4  # Warp writes do not increment Torch versions.
    term(env, "motion_body_pos")
    assert term.cache.computations == 5
    term.reset(env_ids=torch.tensor([1]))
    term(env, "motion_body_ori")
    assert term.cache.computations == 6


def test_termination_uses_raw_height_and_tilt_even_with_stale_derived_pose(cpu_case):
    env, cmd, heads, state, _ = cpu_case
    # Make derived body data look perfectly on target while raw state crosses both
    # thresholds. Different origin heights also catch world/local frame mixing.
    cmd.robot_root_pos_w = cmd.root_pos_w.clone()
    cmd.robot_root_quat_w = cmd.root_quat_w.clone()
    state[:, :3] = cmd._ref.root_pos_w
    state[:, 2] += torch.tensor([0.251, 0.249, -0.251, 0.0])
    state[:, 3:7] = quat_from_rotvec(torch.tensor([[0., 0., 0.], [1.7, 0., 0.], [0., 0., 0.], [0., 0., 0.]]))
    env.sim.data.qpos.copy_(state[:, :36])
    env.sim.data.qpos[:, :3] += env.scene.env_origins
    ref = _reference(cmd)
    features = heads.termination_features(state[:, :3], state[:, 3:7], ref)
    assert torch.equal(bad_root_height(env, "motion"), features[:, 0].abs() > 0.25)
    assert torch.equal(bad_root_tilt(env, "motion"), features[:, 1].abs() > 0.8)
    assert bad_root_height(env, "motion").tolist() == [True, False, True, False]
    assert bad_root_tilt(env, "motion").tolist() == [False, True, False, False]
    assert not bad_root_height(env, "motion", state_source="legacy").any()
    assert not bad_root_tilt(env, "motion", state_source="legacy").any()


def test_legacy_protocol_keeps_original_functions_weights_and_anchor():
    old = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], reward_mode="legacy"))
    new = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[]))
    assert new.commands["motion"].anchor_body_name == old.commands["motion"].anchor_body_name == "torso_link"
    assert new.rewards.keys() == old.rewards.keys()
    for name in new.rewards:
        assert new.rewards[name].weight == old.rewards[name].weight
        assert new.rewards[name].func is PostPhysicsTrackingReward
        assert old.rewards[name].func is not PostPhysicsTrackingReward
    assert old.terminations["root_height"].params["state_source"] == "legacy"
    with pytest.raises(ValueError, match="reward_mode"):
        make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], reward_mode="typo"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_actual_env_step_rewards_and_terminations_match_analytic_heads(tmp_path):
    """Self-contained CUDA test; no downloaded motion data or checkpoints needed."""
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.envs.state_io import StateIO

    model = load_g1_model()
    q = torch.tensor(model.qpos0, dtype=torch.float32)
    reference = torch.cat([q, torch.zeros(35)]).repeat(64, 1)
    path = tmp_path / "tiny.pt"
    torch.save(dict(ref=reference, names=["synthetic"], starts=[0], lengths=[64]), path)
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(path)], compact_library=True, num_envs=4,
        obs_noise=False, reset_noise=False, domain_randomization=False,
        action_noise_std=0.0, record_transitions=False, sampling_mode="start",
    ))
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        io = StateIO(env)
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(model, env.device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0], robot.data.default_joint_pos[0])
        cmd = env.command_manager.get_term("motion")
        for iteration in range(2):
            env.reset()
            state = io.read()
            state[:, :2] += torch.tensor([[0.0, 0.0], [0.2, -0.15], [-0.3, 0.4], [0.1, 0.2]], device=env.device)
            tilt = 1.7 if iteration == 0 else np.pi
            state[:, 3:7] = quat_from_rotvec(torch.tensor([[0., 0., 0.], [0., 0., 0.6], [0., 0., -0.8], [tilt, 0., 0.]], device=env.device))
            state[2, 2] += 0.4
            state[:, 36:42] = torch.tensor([2., -1., 0.7, 1., -2., 4.], device=env.device) * (iteration + 1)
            io.write(torch.arange(4, device=env.device), state)
            # Do not refresh command re-anchoring: that reproduces the real timing.
            ref = _reference(cmd)
            prev = env.action_manager.action.clone()
            action = torch.sin(torch.arange(116, device=env.device).reshape(4, 29)) * 0.15
            cache = env._post_physics_reward_caches["motion"]
            count = cache.computations
            _, actual, terminated, timeout, _ = env.step(action)
            expected = heads.reward(io.read(), action, prev, ref)
            torch.testing.assert_close(actual, expected["total"], atol=2e-6, rtol=2e-4)
            for name in env.reward_manager.active_terms:
                torch.testing.assert_close(env.reward_manager.get_term(name), expected[name], atol=2e-6, rtol=2e-4)
            assert cache.computations == count + 1
            assert torch.equal(terminated, heads.terminated(io.read(), ref))
            assert terminated.any(), "exercise the positive termination path"
            done = terminated | timeout
            assert (~done).any(), "exercise continuing slots during a partial reset"
            env.reset(env_ids=done.nonzero(as_tuple=False).squeeze(-1))
            ref = _reference(cmd)
            prev = env.action_manager.action.clone()
            count = cache.computations
            action = action * 0.7
            _, actual, terminated, _, _ = env.step(action)
            expected = heads.reward(io.read(), action, prev, ref)
            torch.testing.assert_close(actual, expected["total"], atol=2e-6, rtol=2e-4)
            for name in env.reward_manager.active_terms:
                torch.testing.assert_close(env.reward_manager.get_term(name), expected[name], atol=2e-6, rtol=2e-4)
            assert cache.computations == count + 1
            assert torch.equal(terminated, heads.terminated(io.read(), ref))
    finally:
        env.close()
