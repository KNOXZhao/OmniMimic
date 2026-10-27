"""Regression checks for the explicit alternative to singular heading coordinates."""

import math

import pytest
import torch

from supermimic.kinematics.rotations import (
    heading_quat, quat_apply, quat_apply_inverse, quat_canonical,
    quat_from_rotvec, quat_from_yaw, quat_mul, quat_normalize, rot6d_from_quat,
)
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg, quat_apply_inverse_quat


def _states(n=4):
    torch.manual_seed(195)
    state = torch.randn(n, 71, dtype=torch.float64)
    state[:, 3:7] = quat_normalize(state[:, 3:7])
    return state


def _rotate_world(state, yaw, shift):
    result = state.clone()
    rotation = quat_from_yaw(yaw)
    result[:, :3] = quat_apply(rotation, state[:, :3]) + shift
    result[:, 3:7] = quat_mul(rotation, state[:, 3:7])
    result[:, 36:39] = quat_apply(rotation, state[:, 36:39])
    return result


def test_legacy_heading_features_and_integration_are_unchanged():
    # Evaluate the original formulas, including old pickled config instances that
    # lack an instance value for the new field (class default must be heading).
    cfg = StateSpecCfg()
    del cfg.__dict__["coordinate_mode"]
    spec = StateSpec(cfg, "cpu", torch.float64)
    state = _states()
    p, q, joints, v, w, joint_v = spec.split(state)
    q = quat_normalize(q)
    heading = heading_quat(q)
    orientation = rot6d_from_quat(quat_canonical(quat_mul(quat_apply_inverse_quat(heading), q)))
    expected = torch.cat([p[:, 2:3], orientation, quat_apply_inverse(heading, v), w, joints, joint_v], -1) / spec.in_scale
    torch.testing.assert_close(spec.canonical(state), expected, rtol=0, atol=0)
    delta = torch.randn(4, 70, dtype=torch.float64)
    d = delta * spec.out_scale
    vn = v + quat_apply(heading, d[:, :3])
    wn, qdn = w + d[:, 3:6], joint_v + d[:, 6:35]
    pn = p + cfg.dt * vn + quat_apply(heading, d[:, 35:38])
    qn = quat_normalize(quat_mul(q, quat_from_rotvec(cfg.dt * wn + d[:, 38:41])))
    jn = joints + cfg.dt * qdn + d[:, 41:70]
    torch.testing.assert_close(spec.integrate(state, delta), torch.cat([pn, qn, jn, vn, wn, qdn], -1), rtol=0, atol=0)


def test_body_gravity_is_invariant_and_integrates_equivariantly():
    spec = StateSpec(StateSpecCfg(coordinate_mode="body_gravity"), "cpu", torch.float64)
    state = _states()
    yaw = torch.tensor([0.8, -2.3, 3.0, -0.1], dtype=torch.float64)
    shift = torch.tensor([2.0, -3.0, 0.0], dtype=torch.float64)
    transformed = _rotate_world(state, yaw, shift)
    torch.testing.assert_close(spec.canonical(state), spec.canonical(transformed), atol=1e-12, rtol=1e-12)
    delta = torch.randn(4, 70, dtype=torch.float64)
    expected = _rotate_world(spec.integrate(state, delta), yaw, shift)
    torch.testing.assert_close(spec.integrate(transformed, delta), expected, atol=1e-12, rtol=1e-12)
    zero_delta = spec.integrate(state, torch.zeros_like(delta))
    torch.testing.assert_close(zero_delta[:, :3], state[:, :3] + 0.02 * state[:, 36:39])
    # The quaternion double cover must not change features or physical increments.
    opposite = state.clone()
    opposite[:, 3:7] *= -1
    torch.testing.assert_close(spec.canonical(state), spec.canonical(opposite), atol=1e-12, rtol=0)
    output = spec.integrate(opposite, delta)
    output[:, 3:7] *= -1
    torch.testing.assert_close(output, spec.integrate(state, delta), atol=1e-12, rtol=0)


@pytest.mark.parametrize("pitch", [89.99, 90.0, 90.01, 180.0])
def test_body_gravity_has_bounded_jacobian_through_vertical_pitch(pitch, monkeypatch):
    import supermimic.world_model.state_spec as module

    def forbidden_heading(_):
        raise AssertionError("body_gravity must not extract Euler heading")

    monkeypatch.setattr(module, "heading_quat", forbidden_heading)
    spec = StateSpec(StateSpecCfg(coordinate_mode="body_gravity"), "cpu", torch.float64)
    state = torch.zeros(1, 71, dtype=torch.float64)
    state[:, 2], state[:, 36] = 0.8, 1.0
    state[:, 3:7] = quat_from_rotvec(torch.tensor([[0.0, math.radians(pitch), 0.0]], dtype=torch.float64))
    delta = torch.linspace(-0.5, 0.5, 70, dtype=torch.float64)[None]
    jac_features = torch.autograd.functional.jacobian(spec.canonical, state, vectorize=True)
    jac_integration = torch.autograd.functional.jacobian(lambda x: spec.integrate(x, delta), state, vectorize=True)
    assert torch.isfinite(jac_features).all() and jac_features.abs().max() < 5
    assert torch.isfinite(jac_integration).all() and jac_integration.abs().max() < 5


def test_body_gravity_continuity_and_finite_difference_gradient():
    spec = StateSpec(StateSpecCfg(coordinate_mode="body_gravity"), "cpu", torch.float64)
    states = torch.zeros(2, 71, dtype=torch.float64)
    states[:, 2], states[:, 36] = 0.8, 1.0
    states[:, 3:7] = quat_from_rotvec(torch.tensor([[0.0, math.radians(89.99), 0.0],
                                                 [0.0, math.radians(90.01), 0.0]], dtype=torch.float64))
    assert (spec.canonical(states)[1] - spec.canonical(states)[0]).norm() < 0.001
    s = states[:1].clone().requires_grad_()
    delta = torch.zeros(1, 70, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(spec.integrate, (s, delta), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_invalid_coordinate_mode_is_rejected():
    with pytest.raises(ValueError, match="coordinate mode"):
        StateSpec(StateSpecCfg(coordinate_mode="unknown"), "cpu")
