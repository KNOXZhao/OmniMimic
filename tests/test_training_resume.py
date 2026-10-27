"""Resume must refill uncheckpointed replay without rewinding the learning schedule."""

from types import SimpleNamespace

import pytest
import torch


def test_resumed_loop_refills_replay_and_keeps_global_lr(monkeypatch):
    import supermimic.training.loop as module
    from supermimic.training.schedule import cosine_lr

    trainer = object.__new__(module.Trainer)
    trainer.cfg = SimpleNamespace(
        schedule=SimpleNamespace(
            total_iterations=1045, warmup_iterations=40, updates_per_env_step=0.4,
            lr_final_ratio=0.03, lr_decay_fraction=0.8,
            policy_updates_per_wm_update=1, log_interval=10000,
            checkpoint_interval=10000, max_hours=None,
        ),
        wm_trainer=SimpleNamespace(lr=2e-4, window=8),
        policy_trainer=SimpleNamespace(lr=2e-4),
    )
    trainer.iteration = 1000
    trainer.num_updates = 123
    trainer._utd_acc = 0.7
    trainer._completed_training_seconds = 100.0
    trainer._completed_gpu_hours = 0.2
    trainer._compute_history_complete = True
    trainer.world = 1
    trainer.writer = None
    trainer.max_step_reward = 0.1
    trainer.ep_reward_sum = torch.zeros(1)
    trainer.ep_len = torch.zeros(1)
    trainer.buffer = []
    update_steps, learning_rates = [], []

    def step(_):
        trainer.buffer.append(1)
        return {}, torch.zeros(1), torch.tensor([False]), torch.tensor([False]), {}

    def update(_):
        assert len(trainer.buffer) >= 40, "resumed replay was used before warmup"
        update_steps.append((len(trainer.buffer), trainer.iteration))
        return {}

    trainer.env = SimpleNamespace(reset=lambda: ({}, {}), step=step)
    trainer.act = lambda _: torch.zeros(1, 29)
    trainer.wm_trainer = SimpleNamespace(set_lr=learning_rates.append, update=update)
    trainer.policy_trainer = SimpleNamespace(set_lr=lambda _: None, update=lambda _: {})
    trainer.save = lambda _: None
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module, "finalize", lambda: None)

    trainer.run()

    assert update_steps == [(42, 1042), (44, 1044)]
    assert trainer.num_updates == 125
    assert learning_rates[0] == pytest.approx(cosine_lr(1042, 1045, 2e-4, 0.03, 0.8))
    assert learning_rates[0] < cosine_lr(42, 1045, 2e-4, 0.03, 0.8)


def test_compute_accumulates_prior_gpu_cost_when_world_size_changes(monkeypatch):
    import supermimic.training.loop as module

    trainer = object.__new__(module.Trainer)
    trainer._completed_training_seconds = 3600.0
    trainer._completed_gpu_hours = 2.0  # prior one-hour run on two GPUs
    trainer._compute_history_complete = True
    trainer._run_started_at = 100.0
    trainer.world = 8
    monkeypatch.setattr(module.time, "perf_counter", lambda: 1900.0)

    compute = trainer.compute_snapshot()
    assert compute["schema_version"] == 1
    assert compute["training_seconds"] == pytest.approx(5400.0)
    assert compute["gpu_hours"] == pytest.approx(6.0)
    assert compute["world_size_at_save"] == 8
    assert compute["history_complete"] is True
