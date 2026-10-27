"""CPU control-flow regressions for absolute actor budgets and resume semantics."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer, validate_resume_config


def _loop_trainer(monkeypatch, *, ratio=3, utd=1.0, budget=5, warmup=2,
                  iteration=0, wm_updates=0, policy_updates=0, log_interval=10000):
    import supermimic.training.loop as module

    cfg = TrainConfig(device="cpu")
    cfg.schedule.total_iterations = 10000
    cfg.schedule.warmup_iterations = 2
    cfg.wm_trainer.window = 1
    cfg.schedule.wm_only_updates = warmup
    cfg.schedule.policy_updates_per_wm_update = ratio
    cfg.schedule.updates_per_env_step = utd
    cfg.schedule.max_policy_updates = budget
    cfg.schedule.log_interval = log_interval
    cfg.schedule.checkpoint_interval = 10000
    trainer = object.__new__(Trainer)
    trainer.cfg = cfg
    trainer.iteration, trainer.num_updates = iteration, wm_updates
    trainer.num_policy_updates = policy_updates
    trainer.world, trainer.is_main = 1, True
    trainer.max_step_reward = 0.1
    trainer.ep_reward_sum, trainer.ep_len = torch.zeros(1), torch.zeros(1)
    trainer._completed_training_seconds = 0.0
    trainer._completed_gpu_hours = 0.0
    trainer._compute_history_complete = True
    trainer._run_started_at = None
    trainer.buffer = SimpleNamespace(size=0)
    events, scalars, saved, resets = [], {}, [], []
    trainer.command = SimpleNamespace(metrics={"sampling_entropy": torch.ones(1)}, bin_return=torch.zeros(1))
    trainer.writer = SimpleNamespace(add_scalar=lambda name, value, step: scalars.update({name: value}), flush=lambda: None)

    def reset():
        resets.append(True)
        return {}, {}

    def step(_):
        events.append(("sim", trainer.iteration + 1))
        trainer.buffer.size += 1
        return {}, torch.zeros(1), torch.tensor([False]), torch.tensor([False]), {}

    def wm_update(buffer):
        assert buffer.size >= 2, "replay must be refilled even after resume"
        events.append(("wm", trainer.iteration))
        return {"loss": 1.0}

    def actor_update(_):
        events.append(("policy", trainer.iteration))
        return {"imag_return_undisc": float(trainer.num_policy_updates + 1)}

    trainer.env = SimpleNamespace(reset=reset, step=step)
    trainer.act = lambda _: torch.zeros(1, 29)
    trainer.wm_trainer = SimpleNamespace(set_lr=lambda _: None, update=wm_update)
    trainer.policy_trainer = SimpleNamespace(set_lr=lambda _: None, update=actor_update)
    trainer.save = lambda name: saved.append((name, trainer.iteration, trainer.num_updates, trainer.num_policy_updates))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module, "finalize", lambda: None)
    return SimpleNamespace(trainer=trainer, events=events, scalars=scalars, saved=saved, resets=resets)


@pytest.mark.parametrize("utd,expected_steps", [(1.0, 5), (2.5, 3)])
def test_two_wm_only_rounds_then_ratio_three_stops_at_exact_actor_budget(monkeypatch, utd, expected_steps):
    fixture = _loop_trainer(monkeypatch, utd=utd)
    fixture.trainer.run()
    trainer = fixture.trainer
    assert trainer.num_updates == 4
    assert trainer.num_policy_updates == 5  # final round is capped to two, not three
    assert trainer.iteration == expected_steps
    updates = [event for event, _ in fixture.events if event != "sim"]
    assert updates == ["wm", "wm", "wm", "policy", "policy", "policy", "wm", "policy", "policy"]
    assert fixture.saved[-1] == ("ckpt_latest.pt", expected_steps, 4, 5)


@pytest.mark.parametrize("completed", [5, 8])
def test_resumed_absolute_budget_already_reached_has_no_extra_steps(monkeypatch, completed):
    fixture = _loop_trainer(monkeypatch, iteration=123, wm_updates=6, policy_updates=completed, budget=5)
    fixture.trainer.run()
    assert fixture.events == []
    assert fixture.trainer.iteration == 123
    assert fixture.trainer.num_updates == 6
    assert fixture.trainer.num_policy_updates == completed
    assert fixture.saved[-1] == ("ckpt_latest.pt", 123, 6, completed)


def test_resumed_warmup_is_global_but_replay_refill_is_per_invocation(monkeypatch):
    fixture = _loop_trainer(monkeypatch, iteration=1000, wm_updates=7, policy_updates=14, budget=16)
    fixture.trainer.run()
    assert fixture.events == [("sim", 1001), ("sim", 1002), ("wm", 1002), ("policy", 1002), ("policy", 1002)]
    assert fixture.trainer.num_updates == 8
    assert fixture.trainer.num_policy_updates == 16


def test_logged_actor_metrics_average_actual_updates_excluding_wm_only_rounds(monkeypatch):
    fixture = _loop_trainer(monkeypatch, log_interval=5)
    fixture.trainer.run()
    # Actor updates return metrics 1,2,3 then4,5; WM-only rounds have no metric.
    assert fixture.scalars["policy/imag_return_undisc"] == pytest.approx(3.0)
    assert fixture.scalars["wm/loss"] == pytest.approx(1.0)


@pytest.mark.parametrize("saved", [{}, {"env": {}, "world_model": {"state_spec": {}}}])
def test_old_checkpoint_missing_coordinate_mode_keeps_heading(saved):
    cfg = TrainConfig()
    assert cfg.world_model.state_spec.coordinate_mode == "heading"
    validate_resume_config(saved, cfg)


@pytest.mark.parametrize("old_mode,new_mode,old_anchor,new_anchor,expected", [
    ("heading", "body_gravity", False, False, "coordinates"),
    ("body_gravity", "heading", False, False, "coordinates"),
    ("heading", "heading", False, True, "anchor_obs"),
    ("heading", "heading", True, False, "anchor_obs"),
])
def test_load_rejects_semantic_mismatch_before_touching_weights(tmp_path, old_mode, new_mode, old_anchor, new_anchor, expected):
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.cfg = TrainConfig()
    trainer.cfg.world_model.state_spec.coordinate_mode = new_mode
    trainer.cfg.env.anchor_obs = new_anchor
    saved = {"world_model": {"state_spec": {"coordinate_mode": old_mode}}, "env": {"anchor_obs": old_anchor}}
    path = tmp_path / "config_only.pt"
    torch.save({"config": saved}, path)
    # There are deliberately no weights or optimizer fields. A semantic mismatch
    # must fail before attempting to deserialize/load any network state.
    with pytest.raises(ValueError, match=expected):
        trainer.load(path)


def test_load_legacy_actor_counter_uses_saved_ratio_not_new_ratio(tmp_path):
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.cfg = TrainConfig()
    trainer.cfg.schedule.policy_updates_per_wm_update = 3
    trainer.wm, trainer.policy, trainer.critic = (torch.nn.Linear(1, 1) for _ in range(3))
    trainer.wm_trainer = SimpleNamespace(opt=torch.optim.Adam(trainer.wm.parameters()))
    trainer.policy_trainer = SimpleNamespace(opt=torch.optim.Adam(trainer.policy.parameters()),
                                             critic_opt=torch.optim.Adam(trainer.critic.parameters()))
    restored = []
    trainer.command = SimpleNamespace(load_sampler_state_dict=lambda state: restored.append(state) or True)
    config = asdict(trainer.cfg)
    config["schedule"]["policy_updates_per_wm_update"] = 2
    del config["world_model"]["state_spec"]["coordinate_mode"]
    ck = {"config": config, "iteration": 1000, "num_updates": 7,
          "wm": trainer.wm.state_dict(), "policy": trainer.policy.state_dict(), "critic": trainer.critic.state_dict(),
          "wm_opt": trainer.wm_trainer.opt.state_dict(), "policy_opt": trainer.policy_trainer.opt.state_dict(),
          "critic_opt": trainer.policy_trainer.critic_opt.state_dict(),
          "bin_return": torch.ones(2), "bin_memory": torch.full((2,), 0.5)}
    path = tmp_path / "legacy.pt"
    torch.save(ck, path)
    trainer.load(path)
    assert trainer.num_policy_updates == 14
    assert trainer.num_updates == 7 and trainer.iteration == 1000
    assert len(restored) == 1
    torch.testing.assert_close(restored[0]["bin_return"], ck["bin_return"])
    assert trainer._compute_history_complete is False


@pytest.mark.parametrize("field,value", [("wm_only_updates", -1), ("policy_updates_per_wm_update", 0), ("max_policy_updates", 0)])
def test_invalid_update_budget_fails_before_simulator_construction(monkeypatch, field, value):
    import supermimic.training.loop as module

    def forbidden_init():
        raise AssertionError("invalid budget must fail before distributed/simulator setup")

    monkeypatch.setattr(module, "init_distributed", forbidden_init)
    cfg = TrainConfig()
    setattr(cfg.schedule, field, value)
    with pytest.raises(ValueError, match=field):
        Trainer(cfg)
