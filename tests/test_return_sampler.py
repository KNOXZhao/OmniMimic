"""Behavioral regressions for finite-horizon return reset sampling (CPU only)."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.data.sampler import (
    discounted_capacity,
    eligible_start_ranges,
    normalized_attainment,
    reset_probabilities,
    sampled_horizon_mass,
    sampler_layout_sha256,
    update_bin_estimates,
)


def perfect_return(steps, gamma=.99):
    return sum(.1 * gamma ** t for t in range(steps))


def test_perfect_short_long_and_time_limited_episodes_have_equal_difficulty():
    horizons = torch.tensor([10, 32, 50, 500])
    actual = torch.tensor([perfect_return(int(h)) for h in horizons], dtype=torch.float64)
    score = normalized_attainment(actual, horizons, .99, .1)
    torch.testing.assert_close(score, torch.ones_like(score), atol=1e-12, rtol=1e-12)


def test_early_failure_keeps_planned_horizon_instead_of_survived_steps():
    # Same ten rewards: one finishes a short reference, the other fails a long one.
    rewards = torch.tensor([perfect_return(10)] * 2, dtype=torch.float64)
    score = normalized_attainment(rewards, torch.tensor([10, 500]), .99, .1)
    assert float(score[0]) == pytest.approx(1.)
    assert 0 < float(score[1]) < .1


@pytest.mark.parametrize("gamma", [1., .999999999, .99])
def test_finite_capacity_handles_zero_and_short_horizons(gamma):
    result = discounted_capacity(torch.tensor([0, 1, 10]), gamma, .1)
    torch.testing.assert_close(result, torch.tensor(
        [0., .1, perfect_return(10, gamma)], dtype=torch.float64
    ))
    assert torch.isfinite(normalized_attainment(torch.tensor([0.]), torch.tensor([0]), gamma, .1)).all()


def test_first_observation_replaces_optimistic_prior_then_uses_ema():
    values = torch.ones(3)
    seen = torch.zeros(3, dtype=torch.bool)
    update_bin_estimates(values, seen, torch.tensor([0, 0, 1]), torch.tensor([.1, .3, float("nan")]), .05)
    torch.testing.assert_close(values, torch.tensor([.2, 1., 1.]))
    assert seen.tolist() == [True, False, False]
    update_bin_estimates(values, seen, torch.tensor([0, 1]), torch.tensor([1., .4]), .05)
    torch.testing.assert_close(values, torch.tensor([.24, .4, 1.]))


def library_for_lengths(lengths, bin_frames=50, names=None):
    starts, last, bin_starts, bin_ends, bin_clips = [], [], [], [], []
    offset = 0
    for index, length in enumerate(lengths):
        starts.append(offset)
        last.append(offset + length - 1)
        for local in range(0, length, bin_frames):
            bin_starts.append(offset + local)
            bin_ends.append(offset + min(local + bin_frames, length))
            bin_clips.append(index)
        offset += length
    lib = SimpleNamespace(
        num_clips=len(lengths), num_frames=offset, num_bins=len(bin_starts),
        clip_start=torch.tensor(starts), clip_last=torch.tensor(last),
        bin_start=torch.tensor(bin_starts), bin_end=torch.tensor(bin_ends),
        bin_clip=torch.tensor(bin_clips),
        frame_clip=torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)),
        bin_frames=bin_frames,
        clips=[SimpleNamespace(name=(names or [f"clip_{j}" for j in range(len(lengths))])[i],
                               start=start, length=length)
               for i, (start, length) in enumerate(zip(starts, lengths))],
    )
    lib.last_frame_of = lambda frames: lib.clip_last[lib.frame_clip[frames]]
    lib.bin_of_frame = lambda frames: torch.bucketize(frames, lib.bin_end, right=True)
    return lib


@pytest.mark.parametrize("length, expected_starts", [(1, 0), (9, 0), (10, 0), (11, 1), (50, 40), (60, 50), (61, 51)])
def test_bin_mask_boundary_reserves_distinct_reference_transitions(length, expected_starts):
    lib = library_for_lengths([length])
    ends, valid = eligible_start_ranges(lib.bin_start, lib.bin_end, lib.clip_last[lib.bin_clip], 10)
    assert int((ends - lib.bin_start).sum()) == expected_starts
    for b in valid.nonzero().flatten():
        frames = torch.arange(lib.bin_start[b], ends[b])
        assert (lib.last_frame_of(frames) - frames >= 10).all()
    # At exactly 60 frames the final ten-frame bin is unusable, not clamped to
    # a start in the previous bin; with 61 frames it contains one valid start.
    if length == 60:
        assert valid.tolist() == [True, False]
    if length == 61:
        assert valid.tolist() == [True, True]
        assert int(ends[1] - lib.bin_start[1]) == 1


def test_perfect_partial_tail_cannot_dominate_hard_sampling():
    horizons = torch.tensor([500, 100, 50, 10, 1])
    values = normalized_attainment(
        torch.tensor([perfect_return(int(h)) for h in horizons]), horizons, .99, .1
    ).float()
    valid = horizons >= 10
    # Even stale memory concentrated on the ineligible final frame is masked.
    probabilities, memory = reset_probabilities(values, torch.tensor([0., 0., 0., 0., 1.]), valid)
    torch.testing.assert_close(probabilities, torch.tensor([.25, .25, .25, .25, 0.]))
    assert memory[-1] == 0


def test_invalid_memory_cannot_make_probabilities_nan_or_add_ineligible_mass():
    p, memory = reset_probabilities(
        torch.tensor([float("nan"), .2, float("inf")]),
        torch.tensor([float("nan"), -1., float("inf")]),
        torch.tensor([True, True, False]),
    )
    assert torch.isfinite(p).all() and torch.isfinite(memory).all()
    assert (p >= 0).all() and p[-1] == 0
    assert float(p.sum()) == pytest.approx(1.)
    with pytest.raises(ValueError, match="No motion reset"):
        reset_probabilities(torch.ones(2), torch.ones(2), torch.zeros(2, dtype=torch.bool))
    p, _ = reset_probabilities(
        torch.ones(10), torch.full((10,), 1e38), torch.ones(10, dtype=torch.bool),
        uniform_ratio=0, hard_ratio=0, memory_ratio=1, memory_alpha=0,
    )
    torch.testing.assert_close(p, torch.full((10,), .1))


def test_tail_diagnostic_counts_partial_bin_draws_and_legacy_clamp():
    p = torch.tensor([.4, .6], dtype=torch.float64)
    start, end, last = torch.tensor([0, 50]), torch.tensor([50, 61]), torch.tensor([60, 60])
    assert sampled_horizon_mass(p, start, end, last, 10) == pytest.approx(.6)
    # Legacy draws at 58,59,60 are all clamped to 58 (two steps remain).
    assert sampled_horizon_mass(p, start, end, last, 1, frame_cap=last - 2) == 0
    assert sampled_horizon_mass(p, start, end, last, 2, frame_cap=last - 2) == pytest.approx(.6 * 3 / 11)
    # A one-frame clip in the legacy uniform sampler can clamp into its predecessor.
    assert sampled_horizon_mass(
        torch.ones(1), torch.tensor([10]), torch.tensor([11]), torch.tensor([10]), 1,
        frame_cap=torch.tensor([8]), capped_clip_last=torch.tensor([9]),
    ) == 1


@pytest.fixture
def command_factory():
    # Import lazily: pure helper tests do not require simulator imports. No
    # simulator is constructed. Headless CPU invocation uses MUJOCO_GL=disable.
    from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg

    def build(lengths=(1001,), episode_steps=500, version="finite_horizon_v1", mode="return", names=None, bin_frames=50):
        cmd = object.__new__(MultiClipMotionCommand)
        cmd.cfg = MultiClipMotionCommandCfg(
            npz_files=[], anchor_body_name="torso_link", body_names=("torso_link",),
            return_sampler_version=version, sampling_mode=mode,
        )
        cmd._env = SimpleNamespace(num_envs=1, device="cpu", reward_buf=torch.tensor([.1]),
                                   termination_manager=SimpleNamespace(terminated=torch.tensor([False])))
        cmd.library = library_for_lengths(lengths, bin_frames, names)
        lib = cmd.library
        cmd._sampler_layout_sha256 = sampler_layout_sha256(
            ((clip.name, clip.start, clip.length) for clip in lib.clips), lib.bin_frames
        )
        cmd._eligible_bin_end, cmd._eligible_bins = eligible_start_ranges(
            lib.bin_start, lib.bin_end, lib.clip_last[lib.bin_clip], cmd.cfg.min_start_steps
        )
        cmd.episode_steps = episode_steps
        cmd.max_ep_steps = episode_steps + 2
        cmd.bin_return = torch.empty(lib.num_bins)
        cmd.bin_memory = torch.empty(lib.num_bins)
        cmd.bin_seen = torch.zeros(lib.num_bins, dtype=torch.bool)
        cmd._reset_sampler_state()
        cmd.ep_len = torch.zeros(1, dtype=torch.long)
        cmd.ep_rewards = torch.zeros(1, cmd.max_ep_steps)
        cmd.ep_bins = torch.zeros(1, cmd.max_ep_steps, dtype=torch.long)
        cmd.ep_horizon = torch.zeros_like(cmd.ep_bins)
        cmd._gamma_pow = cmd.cfg.gamma ** torch.arange(cmd.max_ep_steps, dtype=torch.float64)
        cmd._episode_active = torch.tensor([False])
        cmd.frame = torch.zeros(1, dtype=torch.long)
        cmd.clip_end = torch.tensor([False])
        cmd.bootstrap_fn = None
        cmd.metrics = {name: torch.zeros(1) for name in ("sampling_entropy", "episode_return_to_go")}
        return cmd
    return build


def test_bookkeeping_excludes_initial_zero_and_extra_final_hold(command_factory):
    cmd = command_factory(lengths=[11])
    cmd._log_step_rewards()
    assert cmd.ep_len.item() == 0
    cmd._episode_active[:] = True
    for target in range(1, 11):
        cmd.frame[:] = target
        cmd._log_step_rewards()
    cmd.clip_end[:] = True
    cmd._log_step_rewards()  # Simulator's extra hold earns no sampler credit.
    assert cmd.ep_len.item() == 10
    cmd._finalize_episodes(torch.tensor([0]))
    torch.testing.assert_close(cmd.bin_return, torch.ones(1))
    assert cmd.bin_seen.tolist() == [True]


def test_return_is_credited_to_reset_source_bin_at_target_boundary(command_factory):
    cmd = command_factory(lengths=[100])
    cmd._episode_active[:] = True
    cmd.frame[:] = 50  # Transition starts at reference49, the last start in bin0.
    cmd._log_step_rewards()
    assert cmd.ep_bins[0, 0] == 0
    assert cmd.ep_horizon[0, 0] == 50  # Targets50 through99 remain.
    cmd.frame[:] = 51
    cmd._log_step_rewards()
    assert cmd.ep_bins[0, 1] == 1
    legacy = command_factory(lengths=[100], version="legacy")
    legacy.frame[:] = 50
    legacy._log_step_rewards()
    assert legacy.ep_bins[0, 0] == 1


@pytest.mark.parametrize("survived, failed", [(20, False), (5, True)])
def test_timeout_contract_and_early_failure_through_command_bookkeeping(command_factory, survived, failed):
    cmd = command_factory(episode_steps=20)
    cmd._episode_active[:] = True
    for target in range(1, survived + 1):
        cmd.frame[:] = target
        cmd._log_step_rewards()
    assert cmd.ep_horizon[0, 0] == 20
    cmd._env.termination_manager.terminated[:] = failed
    def forbidden_bootstrap(_):
        raise AssertionError("finite-horizon sampler must not bootstrap timeouts")
    cmd.bootstrap_fn = forbidden_bootstrap
    cmd._finalize_episodes(torch.tensor([0]))
    if failed:
        assert 0 < cmd.bin_return[0] < .3
    else:
        assert cmd.bin_return[0].item() == pytest.approx(1.)


@pytest.mark.parametrize("mode", ["return", "uniform"])
def test_draws_never_enter_short_clips_or_unusable_tail(command_factory, mode):
    cmd = command_factory(lengths=[1, 10, 11, 60, 61], mode=mode)
    frames = cmd._sample_frames(5000)
    assert (cmd.library.last_frame_of(frames) - frames >= 10).all()
    assert set(cmd.library.frame_clip[frames].tolist()) == {2, 3, 4}


def test_start_mode_keeps_short_clips_available_for_evaluation(command_factory):
    cmd = command_factory(lengths=[1, 5], mode="start")
    assert set(cmd._sample_frames(100).tolist()) == {0, 1}


def test_diagnostic_is_read_only_and_detects_legacy_short_draws(command_factory):
    corrected = command_factory(lengths=[61])
    before = corrected.bin_memory.clone()
    info = corrected.sampler_diagnostics()
    assert info["reset_probability_remaining_lt_8"] == 0
    assert info["reset_probability_remaining_le_10"] == pytest.approx(.5)
    torch.testing.assert_close(corrected.bin_memory, before)
    legacy = command_factory(lengths=[61], version="legacy")
    before = legacy.bin_memory.clone()
    info = legacy.sampler_diagnostics()
    assert info["reset_probability_remaining_lt_8"] == pytest.approx(.5 * 8 / 11)
    torch.testing.assert_close(legacy.bin_memory, before)


def test_long_evaluation_horizon_and_all_short_clips_do_not_allocate_huge_logs(monkeypatch):
    import supermimic.envs.mdp.commands as module

    lib = library_for_lengths([1, 5])
    lib.summary = lambda: "tiny evaluation library"
    lib.ref = lib.bodies = lambda frames: None
    monkeypatch.setattr(module, "MotionLibrary", lambda *args, **kwargs: lib)
    robot = SimpleNamespace(
        body_names=["torso_link", "pelvis"],
        find_bodies=lambda names, preserve_order: ([0], names),
        data=SimpleNamespace(default_joint_pos=torch.zeros(1, 29)),
    )
    env = SimpleNamespace(
        device="cpu", num_envs=1, step_dt=.02,
        cfg=SimpleNamespace(episode_length_s=1e5), scene={"robot": robot},
    )
    cfg = module.MultiClipMotionCommandCfg(
        npz_files=[], anchor_body_name="torso_link", body_names=("torso_link",),
        sampling_mode="start",
    )
    cmd = module.MultiClipMotionCommand(cfg, env)
    assert cmd.episode_steps == 5_000_000
    assert cmd.ep_rewards.shape == (1, 1024)
    assert cmd.ep_horizon.shape == (1, 1024)
    assert torch.isfinite(cmd.bin_memory).all()
    assert cmd.sampler_diagnostics()["reset_probability_remaining_lt_8"] == 1
    cfg.sampling_mode = "return"
    with pytest.raises(ValueError, match="No clips provide 10"):
        module.MultiClipMotionCommand(cfg, env)


def test_versioned_checkpoint_rejects_old_raw_semantics_but_legacy_accepts(command_factory):
    cmd = command_factory(lengths=[60])
    old = {"bin_return": torch.full_like(cmd.bin_return, 10.), "bin_memory": cmd.bin_memory.clone()}
    assert not cmd.load_sampler_state_dict(old)
    assert not cmd.bin_seen.any()
    torch.testing.assert_close(cmd.bin_return, torch.ones_like(cmd.bin_return))
    legacy = command_factory(lengths=[60], version="legacy")
    with pytest.warns(RuntimeWarning, match="without verifiable library layout identity"):
        assert legacy.load_sampler_state_dict(old)
    torch.testing.assert_close(legacy.bin_return, old["bin_return"])
    cmd.bin_return[0] = .3
    cmd.bin_seen[0] = True
    saved = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cmd.sampler_state_dict().items()}
    assert cmd.load_sampler_state_dict(saved)
    assert cmd.bin_return[0].item() == pytest.approx(.3) and cmd.bin_seen[0]
    assert cmd.load_sampler_state_dict(cmd.sampler_state_dict())
    assert cmd.bin_return[0].item() == pytest.approx(.3) and cmd.bin_seen[0]
    saved["episode_steps"] += 1
    assert not cmd.load_sampler_state_dict(saved)
    assert not cmd.bin_seen.any()


@pytest.mark.parametrize("version", ["finite_horizon_v1", "legacy"])
@pytest.mark.parametrize("changed", ["order", "lengths", "bin_frames"])
def test_same_bin_count_cannot_hide_changed_library_layout(command_factory, version, changed):
    source = command_factory(lengths=[61, 59], names=["a", "b"], version=version)
    source.bin_return.fill_(.25)
    source.bin_seen.fill_(True)
    state = {k: v.clone() if isinstance(v, torch.Tensor) else v
             for k, v in source.sampler_state_dict().items()}
    lengths = [60, 60] if changed == "lengths" else [61, 59]
    names = ["b", "a"] if changed == "order" else ["a", "b"]
    width = 51 if changed == "bin_frames" else 50
    target = command_factory(lengths=lengths, names=names, bin_frames=width, version=version)
    assert target.library.num_bins == source.library.num_bins == 4
    assert not target.load_sampler_state_dict(state)
    assert not target.bin_seen.any()
    assert target.bin_return[0] == (10. if version == "legacy" else 1.)
    identical = command_factory(lengths=[61, 59], names=["a", "b"], version=version)
    assert identical.load_sampler_state_dict(state)
    torch.testing.assert_close(identical.bin_return, source.bin_return)


def test_layout_hash_includes_starts_and_is_metadata_only():
    original = [("motion", 0, 61), ("other", 61, 59)]
    shifted = [("motion", 0, 61), ("other", 62, 59)]
    assert sampler_layout_sha256(original, 50) != sampler_layout_sha256(shifted, 50)
    assert sampler_layout_sha256(original, 50) == sampler_layout_sha256(iter(original), 50)


@pytest.mark.parametrize("version", ["finite_horizon_v1", "legacy"])
def test_pre_identity_schema_resets_sampler_in_both_modes(command_factory, version):
    cmd = command_factory(lengths=[61, 59], version=version)
    old = {k: v.clone() if isinstance(v, torch.Tensor) else v
           for k, v in cmd.sampler_state_dict().items()}
    old["schema_version"] = 1
    old.pop("library_layout_sha256")
    old.pop("bin_frames")
    old["bin_return"].fill_(.25)
    old["bin_seen"].fill_(True)
    assert not cmd.load_sampler_state_dict(old)
    assert not cmd.bin_seen.any()
