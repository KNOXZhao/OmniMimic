"""The explicit 9D feedback variant must expose drift and match real observation terms.

These CPU tests run MuJoCo FK and mjlab's actual observation functions, plus the
full imagination path. They do not claim to replace the GPU environment tests.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import (
    ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES, SuperMimicEnvCfgArgs, make_env_cfg,
)
from supermimic.imagination.rollout import ANCHOR_NOISE_HALF, imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_rotvec, quat_from_yaw, quat_mul
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, anchor_relative_obs
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(723)
    model = load_g1_model()
    fk = G1Kinematics(model, device="cpu")
    q0 = torch.zeros(29)
    s = torch.zeros(3, 71)
    s[:, 2] = 0.9
    s[:, 3:7] = quat_from_rotvec(torch.tensor([[0.12, -0.18, 0.25]])).repeat(3, 1)
    s[1, :2] += torch.tensor([0.4, -0.2])
    s[2, 3:7] = quat_mul(quat_from_yaw(torch.tensor(0.8)), s[0, 3:7])

    ref = s[0:1].repeat(6, 1)
    ref[:, 0] = torch.arange(6) * 0.02
    ref[:, 3:7] = quat_mul(quat_from_yaw(torch.arange(6) * 0.04), ref[:, 3:7])
    bodies = fk(ref[:, :36], ref[:, 36:])
    motion_file = tmp_path / "feedback_reference.npz"
    np.savez(motion_file, joint_pos=ref[:, 7:36].numpy(), joint_vel=ref[:, 42:71].numpy(),
             body_names=np.array(fk.body_names), body_pos_w=bodies["pos"].numpy(),
             body_quat_w=bodies["quat"].numpy(), body_lin_vel_w=bodies["lin_vel"].numpy(),
             body_ang_vel_w=bodies["ang_vel"].numpy())
    lib = MotionLibrary([motion_file], TRACKED_BODIES, q0, device="cpu")
    limits = torch.tensor(model.jnt_range[1:], dtype=torch.float32)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, limits, q0)
    wm = WorldModel(WorldModelCfg(hidden=(16,), state_spec=StateSpecCfg(explicit_integration=False)), heads, "cpu")
    start = StartBatch(s, torch.zeros(3, 33), torch.zeros(3, dtype=torch.long),
                       torch.zeros(3, 29), s[None].repeat(10, 1, 1), torch.zeros(10, 3, 29))
    return SimpleNamespace(model=model, lib=lib, wm=wm, start=start, q0=q0)


def _mjlab_observation_from_cpu_fk(model, states, lib, frames):
    """Independent CPU FK supplies the fields read by mjlab's real observation terms."""
    from mjlab.tasks.tracking.mdp.observations import motion_anchor_ori_b, motion_anchor_pos_b

    data = mujoco.MjData(model)
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
    positions, quaternions = [], []
    for s in states.detach().numpy():
        data.qpos[:] = s[:36]
        data.qvel[:] = s[36:]
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[anchor_id].copy())
        quaternions.append(data.xquat[anchor_id].copy())
    # Nonzero env origins test that simulator-world versus origin-free states agree.
    origin = torch.tensor([7.0, -3.0, 0.0])
    ai = TRACKED_BODIES.index(ANCHOR_BODY)
    reference = lib.bodies(frames)
    command = SimpleNamespace(
        robot_anchor_pos_w=torch.tensor(np.array(positions), dtype=torch.float32) + origin,
        robot_anchor_quat_w=torch.tensor(np.array(quaternions), dtype=torch.float32),
        anchor_pos_w=reference.pos_w[:, ai] + origin,
        anchor_quat_w=reference.quat_w[:, ai],
    )
    env = SimpleNamespace(num_envs=len(states), command_manager=SimpleNamespace(get_term=lambda _: command))
    return torch.cat([motion_anchor_pos_b(env, "motion"), motion_anchor_ori_b(env, "motion")], -1)


def _policy(goal_dim):
    return Policy(PolicyCfg(goal_dim=goal_dim, encoder_hidden=(16,), decoder_hidden=(16,)))


def test_anchor_feedback_resolves_position_and_yaw_blindness(setup):
    st = setup.start
    proprio = analytic_observation(st.state, st.prev_action, st.xi, setup.q0)
    # Translating or yawing a stationary robot changes no paper proprioceptive input.
    torch.testing.assert_close(proprio[0], proprio[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(proprio[0], proprio[2], atol=1e-6, rtol=0)
    original = imagine(setup.wm, _policy(680), setup.lib, st, 2, obs_noise=False, anchor_obs=False)
    torch.testing.assert_close(original.goals[:, 0], original.goals[:, 1])
    torch.testing.assert_close(original.goals[:, 0], original.goals[:, 2])

    feedback = imagine(setup.wm, _policy(689), setup.lib, st, 2, obs_noise=False, anchor_obs=True)
    assert feedback.goals.shape == (3, 3, 689)
    torch.testing.assert_close(feedback.goals[..., :680], original.goals)
    assert torch.linalg.norm(feedback.goals[0, 0, 680:683] - feedback.goals[0, 1, 680:683]) > 0.4
    assert torch.linalg.norm(feedback.goals[0, 0, 683:] - feedback.goals[0, 2, 683:]) > 0.5
    # Check every imagined timestamp, including advanced reference frames, against
    # independent MuJoCo poses passed to mjlab's actual observation functions.
    for t in range(3):
        expected = _mjlab_observation_from_cpu_fk(setup.model, feedback.states[t], setup.lib, feedback.frames[t])
        torch.testing.assert_close(feedback.goals[t, :, 680:], expected, atol=2e-6, rtol=1e-5)


def test_anchor_noise_contract_matches_simulator_and_imagination(setup):
    args = SuperMimicEnvCfgArgs(npz_files=["unused"], anchor_obs=True, obs_noise=True)
    terms = make_env_cfg(args).observations["goal"].terms
    halves = torch.tensor([terms["anchor_pos"].noise.n_max] * 3 + [terms["anchor_ori"].noise.n_max] * 6)
    lows = torch.tensor([terms["anchor_pos"].noise.n_min] * 3 + [terms["anchor_ori"].noise.n_min] * 6)
    torch.testing.assert_close(halves, ANCHOR_NOISE_HALF)
    torch.testing.assert_close(lows, -ANCHOR_NOISE_HALF)
    assert terms["future_frames"].noise is None
    clean_terms = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused"], anchor_obs=True, obs_noise=False)).observations["goal"].terms
    assert clean_terms["anchor_pos"].noise is None and clean_terms["anchor_ori"].noise is None

    # With a zero-output identity WM, noise cannot change the state or reference.
    # Compare the actual goals returned by imagination with/without corruption.
    st = setup.start
    n = 256
    repeated = StartBatch(st.state[:1].repeat(n, 1), st.xi[:1].repeat(n, 1),
                         st.frame[:1].repeat(n), st.prev_action[:1].repeat(n, 1),
                         st.hist_states[:, :1].repeat(1, n, 1), st.hist_actions[:, :1].repeat(1, n, 1))
    policy = _policy(689)
    clean = imagine(setup.wm, policy, setup.lib, repeated, 1, obs_noise=False, anchor_obs=True)
    noisy = imagine(setup.wm, policy, setup.lib, repeated, 1, obs_noise=True, anchor_obs=True)
    torch.testing.assert_close(clean.goals[..., :680], noisy.goals[..., :680])
    error = noisy.goals[..., 680:] - clean.goals[..., 680:]
    assert (error.abs() <= halves + 1e-6).all()
    expected_std = halves / np.sqrt(3)
    torch.testing.assert_close(error.flatten(0, 1).std(dim=0), expected_std, atol=0.015, rtol=0.1)


def test_anchor_orientation_gradient_is_finite_at_vertical_pitch():
    # Full relative rotation has no heading extraction or vertical-pitch singularity.
    rotvec = torch.tensor([[0.0, np.pi / 2, 0.0]], dtype=torch.float64, requires_grad=True)
    q = quat_from_rotvec(rotvec)
    pos = torch.tensor([[0.2, -0.3, 0.9]], dtype=torch.float64, requires_grad=True)
    ref_pos = torch.tensor([[0.5, 0.1, 0.8]], dtype=torch.float64)
    ref_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    output = anchor_relative_obs(pos, q, ref_pos, ref_q)
    weights = torch.arange(1, 10, dtype=torch.float64)
    g_pos, g_rot = torch.autograd.grad((output * weights).sum(), (pos, rotvec))
    assert torch.isfinite(g_pos).all() and torch.isfinite(g_rot).all()
    assert 0 < g_rot.abs().max() < 50
    assert torch.autograd.gradcheck(lambda p, v: anchor_relative_obs(p, quat_from_rotvec(v), ref_pos, ref_q),
                                   (pos, rotvec), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("coordinate_mode", ["heading", "body_gravity"])
def test_anchor_policy_update_has_finite_feedback_gradients(setup, coordinate_mode):
    """Run the production actor/critic update on CPU with a small differentiable WM.

    No action-rate penalty is used, so the actor signal must pass through the WM
    and state-dependent reward/value. This checks gradient plumbing, not learning
    quality or the accuracy of this synthetic, untrained world model.
    """
    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
    from supermimic.policy.nets import Critic, CriticCfg

    setup.wm.spec.cfg.coordinate_mode = coordinate_mode
    setup.wm.heads.reward_cfg.w_action_rate = 0.0
    setup.wm.heads.reward_cfg.w_joint_limit = 0.0
    # An untrained zero-output dynamics layer has no action derivative. Seed a
    # small nonzero layer to exercise BPTT instead of only direct action penalties.
    with torch.no_grad():
        setup.wm.dynamics.mlp.net[-1].weight.normal_(std=0.005)
    policy = _policy(689)
    critic = Critic(CriticCfg(hidden=(16,)), pcfg=policy.cfg)
    cfg = PolicyTrainerCfg(horizon=3, batch_size=3, critic_epochs=1, critic_minibatches=1,
                           compile=None, anchor_obs=True, obs_noise=True)
    trainer = PolicyTrainer(setup.wm, policy, critic, setup.lib, cfg)
    frozen_wm = {name: parameter.detach().clone() for name, parameter in setup.wm.named_parameters()}
    actor_before = policy.encoders["motion"].mlp.net[0].weight.detach().clone()
    critic_before = critic.mlp.net[0].weight.detach().clone()
    replay = SimpleNamespace(sample_starts=lambda batch, hist: setup.start)
    metrics = trainer.update(replay)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["policy_grad_norm"] > 0
    feedback_weight = policy.encoders["motion"].mlp.net[0].weight
    assert torch.isfinite(feedback_weight.grad).all()
    assert feedback_weight.grad[:, -9:].abs().sum() > 0
    assert not torch.equal(feedback_weight, actor_before)
    assert not torch.equal(critic.mlp.net[0].weight, critic_before)
    for name, parameter in setup.wm.named_parameters():
        torch.testing.assert_close(parameter, frozen_wm[name], rtol=0, atol=0)
        assert parameter.grad is None
