"""Meaningful CPU tests for the isolated matched-protocol additions."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.training.matched import (
    OverflowCounter, capacity_overflow_on_any_rank, install_overflow_counter,
    learning_rate_step, rank0_deadline_reached,
)
from test_training_budget import _loop_trainer


def test_four_substep_hook_keeps_early_overflow_before_later_reset():
    flags = torch.zeros(2, dtype=torch.int32)
    nefc = torch.zeros(2, dtype=torch.int32)
    nacon, ncollision = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    patterns = [(1, 0), (512, 1024), (0, 8), (0, 0)]
    calls = []
    def step():
        # Diagnostic bits are clear at each fresh physics substep.
        assert not bool(flags.any())
        flags.copy_(torch.tensor(patterns[len(calls)]))
        nefc.fill_(100 + len(calls))
        nacon.fill_(10 + len(calls))
        ncollision.fill_(20 + len(calls))
        calls.append(1)
    sim = SimpleNamespace(data=SimpleNamespace(overflow=flags, nefc=nefc, nacon=nacon, ncollision=ncollision), step=step)
    counter = install_overflow_counter(sim)
    for _ in range(4):
        sim.step()
    flags.zero_()  # an environment reset cannot erase prior observations
    s = counter.snapshot()
    assert s['physics_substeps'] == 4 and s['world_substep_observations'] == 8
    assert s['by_flag_world_substeps']['NEFC'] == 1
    assert s['by_flag_world_substeps']['NARROWPHASE'] == 1
    assert s['by_flag_world_substeps']['ITERATIONS'] == 1
    assert s['by_flag_world_substeps']['LS_ITERATIONS'] == 1
    assert (s['max_nefc_observed'], s['max_nacon_observed'], s['max_ncollision_observed']) == (103, 13, 23)
    assert capacity_overflow_on_any_rank(counter, 1)
    with pytest.raises(ValueError, match='already installed'):
        install_overflow_counter(sim)


def test_solver_limit_flags_are_recorded_but_do_not_abort():
    flags = torch.tensor([512, 1024], dtype=torch.int32)
    c = OverflowCounter(flags, torch.tensor([90, 91]))
    c.after_step()
    assert c.snapshot()['any_flag_world_substeps'] == 2
    assert not capacity_overflow_on_any_rank(c, 1)


def test_remote_rank_overflow_is_broadcast_to_clean_rank(monkeypatch):
    c = OverflowCounter(torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32))
    monkeypatch.setattr('supermimic.training.matched.dist.all_reduce', lambda flag, op: flag.fill_(1))
    assert capacity_overflow_on_any_rank(c, 2)


def test_deadline_uses_rank_zero_and_includes_boundary(monkeypatch):
    assert rank0_deadline_reached(3600, 1, rank=0, world=1, device='cpu')
    assert not rank0_deadline_reached(3599, 1, rank=0, world=1, device='cpu')
    seen = []
    def broadcast(flag, src):
        seen.append(int(flag))
        flag.fill_(1)
    monkeypatch.setattr('supermimic.training.matched.dist.broadcast', broadcast)
    assert rank0_deadline_reached(1, 1, rank=1, world=2, device='cpu')
    assert seen == [0]


def test_warmup_credit_samples_lr_and_actor_checkpoints_match_allmimic(monkeypatch):
    f = _loop_trainer(monkeypatch, ratio=1, utd=.4, budget=100, warmup=0)
    t = f.trainer
    t.cfg.schedule.warmup_iterations = 65
    t.cfg.wm_trainer.window = 8
    t.cfg.schedule.policy_checkpoint_interval = 25
    t.cfg.schedule.lr_clock = 'credited_collection_steps'
    t.run()
    assert (t.iteration, t.credited_collection_steps, t.num_policy_updates, t.num_updates) == (314, 250, 100, 100)
    assert [x[0] for x in f.saved if x[0].startswith('ckpt_policy_')] == [
        'ckpt_policy_00000025.pt', 'ckpt_policy_00000050.pt',
        'ckpt_policy_00000075.pt', 'ckpt_policy_00000100.pt',
    ]
    assert next(i for event, i in f.events if event == 'policy') == 67
    assert learning_rate_step(67, 3, 'credited_collection_steps') == 2
    assert learning_rate_step(67, 3, 'iteration') == 67


def test_clock_includes_initial_env_reset_and_stops_without_waiting_for_log(monkeypatch):
    import supermimic.training.loop as module
    f = _loop_trainer(monkeypatch, ratio=1, utd=.4, budget=100, warmup=0)
    t = f.trainer
    clock = [100.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    def reset():
        clock[0] += 42
        return {}, {}
    t.env.reset = reset
    t.cfg.schedule.max_hours = .01
    t.run()
    assert t.iteration == 1 and t.num_policy_updates == 0
    assert t._completed_training_seconds == 42


def test_capacity_failure_aborts_before_optimizer_updates(monkeypatch, tmp_path):
    f = _loop_trainer(monkeypatch, ratio=1, utd=1, budget=100, warmup=0)
    t = f.trainer
    t.log_dir = tmp_path
    t.env.overflow_counter = OverflowCounter(torch.tensor([1], dtype=torch.int32), torch.tensor([600]))
    t.env.overflow_counter.after_step()
    with pytest.raises(RuntimeError, match='Strict capacity'):
        t.run()
    assert t.num_updates == t.num_policy_updates == 0
    assert f.saved == [('ckpt_capacity_failure.pt', 1, 0, 0)]
    assert (tmp_path / 'CAPACITY_FAILURE.json').exists()


def test_fp32_and_capacity_configuration_reach_environment_before_construction():
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
    a = SuperMimicEnvCfgArgs(npz_files=['not_loaded_in_config.pt'], compact_library=True,
                            library_fp16=False, njmax=512, nconmax=128, warn_overflow=True)
    cfg = make_env_cfg(a)
    assert cfg.commands['motion'].library_fp16 is False
    assert (cfg.sim.njmax, cfg.sim.nconmax) == (512, 128)
    opt = SimpleNamespace(warn_overflow=4095)
    cfg.sim.apply_wp_opt(opt)
    assert opt.warn_overflow == 15


def test_compact_cpu_loader_keeps_unquantized_reference_values(tmp_path):
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
    ref = torch.zeros(12, 71, dtype=torch.float32)
    ref[:, 0] = 12.345678
    ref[:, 2] = .789123
    ref[:, 3] = 1
    ref[:, 7] = .1234567
    path = tmp_path / 'fp32.pt'
    torch.save({'ref': ref, 'names': ['test'], 'starts': [0], 'lengths': [12]}, path)
    lib = MotionLibrary([path], TRACKED_BODIES, torch.zeros(29), device='cpu', compact=True, store_dtype=torch.float32)
    assert lib._ref.dtype == torch.float32
    assert torch.equal(lib._ref, ref)
    assert not torch.equal(ref, ref.half().float())


def test_actual_update_nan_guard_aborts_before_actor_and_records_context(monkeypatch, tmp_path):
    import json
    f = _loop_trainer(monkeypatch, ratio=1, utd=1, budget=100, warmup=0)
    t = f.trainer
    t.rank = 0
    t.log_dir = tmp_path
    t.wm_trainer.update = lambda _: {'loss': 1., 'grad_norm': float('inf')}
    with pytest.raises(RuntimeError, match='Nonfinite actual world_model'):
        t.run()
    assert not any(event == 'policy' for event, _ in f.events)
    assert t.num_policy_updates == 0
    failure = json.loads((tmp_path / 'NONFINITE_FAILURE_rank0.json').read_text())
    assert failure['local_nonfinite_metrics'] == {'grad_norm': '+Inf'}
    assert f.saved[-1][0] == 'ckpt_nonfinite_failure.pt'


def test_progress_contains_real_empty_warmup_metrics_and_final_counters(monkeypatch, tmp_path):
    import json
    f = _loop_trainer(monkeypatch, ratio=1, utd=.4, budget=1, warmup=0, log_interval=1)
    t = f.trainer
    t.rank = 0
    t.log_dir = tmp_path
    t.cfg.schedule.warmup_iterations = 3
    t.run()
    p = json.loads((tmp_path / 'progress_rank0.json').read_text())
    assert p['stage'] == 'complete' and p['num_policy_updates'] == 1
    assert p['credited_collection_steps'] == 3 and p['iteration'] == 5
    assert p['actual_update_metrics'] == {}
    assert p['total_transitions_all_ranks'] == 5 * 4096


def test_model_fingerprint_detects_parameter_buffer_difference_and_nan(tmp_path):
    from supermimic.training.matched import model_fingerprints
    a = torch.nn.Linear(2, 2)
    a.register_buffer('normalizer', torch.ones(2))
    b = torch.nn.Linear(2, 2)
    b.register_buffer('normalizer', torch.ones(2))
    b.load_state_dict(a.state_dict())
    assert model_fingerprints({'policy': a}) == model_fingerprints({'policy': b})
    b.normalizer[0] = 2
    assert model_fingerprints({'policy': a}) != model_fingerprints({'policy': b})
    b.normalizer[0] = float('nan')
    assert model_fingerprints({'policy': b})['policy']['nonfinite'] == ['normalizer']


def test_training_clock_stops_before_final_checkpoint_and_audit(monkeypatch):
    import supermimic.training.loop as module
    f = _loop_trainer(monkeypatch, ratio=1, utd=1, budget=1, warmup=0)
    t, clock = f.trainer, [100.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    old_step = t.env.step
    def step(action):
        clock[0] += 2.
        return old_step(action)
    t.env.step = step
    old_save = t.save
    def save(name):
        clock[0] += 20.
        old_save(name)
    t.save = save
    t.audit_final_models = lambda: clock.__setitem__(0, clock[0] + 10.)
    t.run()
    assert t._completed_training_seconds == t.iteration * 2.
    assert clock[0] - 100. == t._completed_training_seconds + 30.


def test_loaded_library_provenance_comes_from_live_library(monkeypatch):
    f = _loop_trainer(monkeypatch)
    f.trainer.command.library = SimpleNamespace(num_clips=129943, num_frames=47475680,
        num_bins=1234, compact=True, _ref=torch.zeros(1, 71, dtype=torch.float32))
    assert f.trainer.library_snapshot() == {'num_clips': 129943, 'num_frames': 47475680,
        'num_bins': 1234, 'compact': True, 'stored_ref_dtype': 'torch.float32'}
