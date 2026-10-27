"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import cosine_lr
from supermimic.training.matched import (
    any_rank_flag, atomic_json, capacity_overflow_on_any_rank, learning_rate_step,
    model_fingerprints, nonfinite_metrics, rank0_deadline_reached,
)
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


def validate_resume_config(saved: dict, cfg: TrainConfig) -> None:
    """Reject equal-shaped weights whose input/output meanings have changed."""
    old_mode = saved.get("world_model", {}).get("state_spec", {}).get("coordinate_mode", "heading")
    new_mode = cfg.world_model.state_spec.coordinate_mode
    if old_mode != new_mode:
        raise ValueError(f"Cannot resume world-model coordinates {old_mode!r} as {new_mode!r}; train a new run")
    if saved.get("env", {}).get("anchor_obs", False) != cfg.env.anchor_obs:
        raise ValueError("Cannot change anchor_obs while resuming; train a new policy")


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.schedule.wm_only_updates < 0:
            raise ValueError("wm_only_updates must be nonnegative")
        if cfg.schedule.policy_updates_per_wm_update < 1:
            raise ValueError("policy_updates_per_wm_update must be positive")
        if cfg.schedule.max_policy_updates is not None and cfg.schedule.max_policy_updates < 1:
            raise ValueError("max_policy_updates must be positive")
        if cfg.data.library is not None:
            cfg.env.compact_library = True
        cfg.policy.goal_dim = cfg.env.future_frames * 68 + (9 if cfg.env.anchor_obs else 0)
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                library_fp16=cfg.env.library_fp16,
                nconmax=cfg.env.nconmax,
                njmax=cfg.env.njmax,
                warn_overflow=cfg.env.warn_overflow,
                monitor_overflow=cfg.env.monitor_overflow,
                anchor_obs=cfg.env.anchor_obs,
                reward_mode=cfg.env.reward_mode,
                extra_command_kwargs={
                    "return_sampler_version": cfg.env.return_sampler_version,
                    "min_start_steps": cfg.env.min_start_steps,
                },
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(load_g1_model(), device=self.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.num_updates = 0  # world-model update rounds
        self.num_policy_updates = 0
        self.credited_collection_steps = 0
        self._utd_acc = 0.0
        self._completed_training_seconds = 0.0
        self._completed_gpu_hours = 0.0
        self._compute_history_complete = True
        self._run_started_at: float | None = None
        self._run_stopped_at: float | None = None
        if cfg.resume is not None:
            self.load(cfg.resume)

    # ------------------------------------------------------------------ checkpoints

    def compute_snapshot(self) -> dict:
        """Cumulative measured loop cost, retaining each segment's GPU count on resume.

        Environment construction and checkpoint evaluation are outside this timer.
        Legacy checkpoints without timing retain an explicit incomplete-history flag.
        """
        stopped = getattr(self, "_run_stopped_at", None)
        seconds = 0.0 if self._run_started_at is None else (stopped if stopped is not None else time.perf_counter()) - self._run_started_at
        return {
            "schema_version": 1,
            "training_seconds": self._completed_training_seconds + seconds,
            "gpu_hours": self._completed_gpu_hours + seconds * self.world / 3600.0,
            "world_size_at_save": self.world,
            "history_complete": self._compute_history_complete,
            "scope": "barrier_synchronized_training_loop_including_warmup_and_periodic_checkpoints_excluding_construction_evaluation_and_finalization",
        }

    def library_snapshot(self) -> dict | None:
        library = getattr(self.command, "library", None)
        if library is None:
            return None
        return {"num_clips": library.num_clips, "num_frames": library.num_frames,
                "num_bins": library.num_bins, "compact": library.compact,
                "stored_ref_dtype": str(library._ref.dtype)}

    def write_progress(self, stage: str, metrics: dict | None = None) -> None:
        if not hasattr(self, "log_dir"):
            return  # CPU loop fixtures do not create output directories.
        counter = getattr(self.env, "overflow_counter", None)
        atomic_json(self.log_dir / f"progress_rank{self.rank}.json", {
            "stage": stage, "rank": self.rank, "world_size": self.world,
            "iteration": self.iteration, "credited_collection_steps": self.credited_collection_steps,
            "num_wm_updates": self.num_updates, "num_policy_updates": self.num_policy_updates,
            "total_transitions_per_rank": self.iteration * self.cfg.env.num_envs,
            "total_transitions_all_ranks": self.iteration * self.cfg.env.num_envs * self.world,
            "compute": self.compute_snapshot(), "actual_update_metrics": metrics or {},
            "loaded_library": self.library_snapshot(),
            "finalization_seconds_so_far": (time.perf_counter() - self._run_stopped_at)
                if getattr(self, "_run_stopped_at", None) is not None else 0.0,
            "overflow": counter.snapshot() if counter is not None else None,
            "updated_unix_seconds": time.time(),
        })

    def check_update_finite(self, metrics: dict, stage: str) -> None:
        bad = nonfinite_metrics(metrics)
        if any_rank_flag(bool(bad), self.world, getattr(self, "device", "cpu")):
            if hasattr(self, "log_dir"):
                atomic_json(self.log_dir / f"NONFINITE_FAILURE_rank{self.rank}.json", {
                    "stage": stage, "iteration": self.iteration,
                    "num_wm_updates": self.num_updates,
                    "num_policy_updates": self.num_policy_updates,
                    "local_nonfinite_metrics": bad,
                })
            self.save("ckpt_nonfinite_failure.pt")
            raise RuntimeError(f"Nonfinite actual {stage} update metrics on at least one rank")

    def audit_final_models(self) -> None:
        if not hasattr(self, "wm"):
            return  # CPU loop fixtures use stub trainers.
        local = model_fingerprints({"wm": self.wm, "policy": self.policy, "critic": self.critic})
        ranks = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(ranks, local)
        else:
            ranks[0] = local
        passed = all(item == ranks[0] for item in ranks) and not any(
            v["nonfinite"] for item in ranks for v in item.values()
        )
        atomic_json(self.log_dir / f"final_model_audit_rank{self.rank}.json", {
            "passed": passed, "rank": self.rank, "world_size": self.world,
            "iteration": self.iteration, "num_policy_updates": self.num_policy_updates,
            "model_fingerprints_by_rank": ranks,
            "scope": "exact parameters/buffers; sampler intentionally excluded",
        })
        if not passed:
            raise RuntimeError("Final model finiteness or rank agreement failed")

    def save(self, name: str = "ckpt_latest.pt") -> None:
        counter = getattr(self.env, "overflow_counter", None)
        overflow = counter.snapshot() if counter is not None else None
        if overflow is not None:
            (self.log_dir / f"overflow_rank{self.rank}_{Path(name).stem}.json").write_text(
                json.dumps(overflow, indent=2)
            )
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "num_policy_updates": self.num_policy_updates,
                "credited_collection_steps": self.credited_collection_steps,
                "collection": {
                    "total_collection_steps_per_rank": self.iteration,
                    "credited_collection_steps_per_rank": self.credited_collection_steps,
                    "total_transitions_per_rank": self.iteration * self.cfg.env.num_envs,
                    "total_transitions_all_ranks": self.iteration * self.cfg.env.num_envs * self.world,
                    "world_size": self.world,
                },
                "overflow_rank0": overflow,
                "loaded_library": self.library_snapshot(),
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "sampler": self.command.sampler_state_dict(),
                "config": asdict(self.cfg),
                "compute": self.compute_snapshot(),
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        validate_resume_config(ck.get("config", {}), self.cfg)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        sampler = ck.get("sampler")
        if sampler is None and "bin_return" in ck:
            sampler = {k: ck[k] for k in ("bin_return", "bin_memory")}
        restored = self.command.load_sampler_state_dict(sampler)
        if not restored:
            print("[INFO] sampler reset: checkpoint uses an incompatible sampler protocol or library")
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        old_ratio = ck.get("config", {}).get("schedule", {}).get("policy_updates_per_wm_update", 1)
        self.num_policy_updates = ck.get("num_policy_updates", self.num_updates * old_ratio)
        self.credited_collection_steps = ck.get("credited_collection_steps", 0)
        compute = ck.get("compute")
        self._completed_training_seconds = float(compute["training_seconds"]) if compute else 0.0
        self._completed_gpu_hours = float(compute["gpu_hours"]) if compute else 0.0
        self._compute_history_complete = bool(compute.get("history_complete", True)) if compute else False
        self._run_started_at = None
        self._run_stopped_at = None
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        if self.world > 1:
            dist.barrier()
        self._run_stopped_at = None
        self._run_started_at = time.perf_counter()
        obs, _ = env.reset()
        t_start = self._run_started_at
        # Replay is not checkpointed: every invocation must refill it, including a
        # resumed run whose global iteration already exceeds the warmup threshold.
        collected_steps = 0
        warmup_steps = max(sch.warmup_iterations, cfg.wm_trainer.window + 1)
        self._utd_acc = 0.0
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        agg_policy_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()

        # Absolute actor-update budget supports comparisons with different WM warmup.
        policy_budget = getattr(sch, "max_policy_updates", None)
        wm_only_updates = getattr(sch, "wm_only_updates", 0)
        if not hasattr(self, "num_policy_updates"):
            self.num_policy_updates = self.num_updates * sch.policy_updates_per_wm_update
        if not hasattr(self, "credited_collection_steps"):
            self.credited_collection_steps = 0
        checkpoint_updates = getattr(sch, "policy_checkpoint_interval", None)
        next_update_checkpoint = (
            (self.num_policy_updates // checkpoint_updates + 1) * checkpoint_updates
            if checkpoint_updates else None
        )

        while self.iteration < sch.total_iterations and (
            policy_budget is None or self.num_policy_updates < policy_budget
        ):
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1
            collected_steps += 1
            counter = getattr(env, "overflow_counter", None)
            if counter is not None and capacity_overflow_on_any_rank(counter, self.world):
                self.save("ckpt_capacity_failure.pt")
                if self.is_main:
                    (self.log_dir / "CAPACITY_FAILURE.json").write_text(json.dumps({
                        "iteration": self.iteration,
                        "num_policy_updates": self.num_policy_updates,
                        "reason": "capacity overflow on at least one rank; stopped before optimizer updates",
                    }, indent=2))
                raise RuntimeError("Strict capacity overflow: inspect per-rank failure telemetry")

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if collected_steps >= warmup_steps:
                self.credited_collection_steps += 1
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr_step = learning_rate_step(self.iteration, self.credited_collection_steps,
                                                 getattr(sch, "lr_clock", "iteration"))
                    lr = cosine_lr(lr_step, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    self.check_update_finite(m_wm, "world_model")
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    m_pi = {}
                    policy_count = 0
                    if self.num_updates >= wm_only_updates:
                        count = sch.policy_updates_per_wm_update
                        if policy_budget is not None:
                            count = min(count, policy_budget - self.num_policy_updates)
                        for _ in range(count):
                            metrics = self.policy_trainer.update(self.buffer)
                            self.check_update_finite(metrics, "policy_critic")
                            for key, value in metrics.items():
                                m_pi[key] = m_pi.get(key, 0.0) + value
                            policy_count += 1
                            self.num_policy_updates += 1
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1
                    agg_policy_n += policy_count
                    if policy_budget is not None and self.num_policy_updates >= policy_budget:
                        break

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.perf_counter() - t_start
                actual_metrics = {
                    k: v / (1 if k == "lr" else max(agg_policy_n if k.startswith("policy/") else agg_n, 1))
                    for k, v in agg.items()
                }
                self.write_progress("training", actual_metrics)
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n, agg_policy_n = {}, 0, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        count = agg_policy_n if k.startswith("policy/") else agg_n
                        self.writer.add_scalar(k, v / (max(count, 1) if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    compute = self.compute_snapshot()
                    self.writer.add_scalar("time/hours", compute["training_seconds"] / 3600, it)
                    self.writer.add_scalar("time/gpu_hours", compute["gpu_hours"], it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("time/policy_updates", self.num_policy_updates, it)
                    self.writer.add_scalar("time/credited_collection_steps", self.credited_collection_steps, it)
                    self.writer.add_scalar("time/transitions_all_ranks", self.iteration * cfg.env.num_envs * self.world, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | wm/pi updates {self.num_updates}/{self.num_policy_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_policy_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_policy_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n, agg_policy_n = {}, 0, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if next_update_checkpoint is not None and self.num_policy_updates >= next_update_checkpoint:
                # The matched recipe performs one actor update per WM round.
                # Save the actual count rather than silently labelling an overshoot.
                self.save(f"ckpt_policy_{self.num_policy_updates:08d}.pt")
                next_update_checkpoint = (self.num_policy_updates // checkpoint_updates + 1) * checkpoint_updates
            if rank0_deadline_reached(time.perf_counter() - t_start, sch.max_hours,
                                      rank=getattr(self, "rank", 0), world=self.world,
                                      device=getattr(self, "device", "cpu")):
                if self.is_main:
                    print("[INFO] rank-0 training-entry clock budget reached, stopping")
                break
        # Preserve the common training clock; report final write/audit overhead
        # separately rather than charging it to the primary budget metadata.
        self._run_stopped_at = time.perf_counter()
        self.save("ckpt_latest.pt")
        self.audit_final_models()
        self.write_progress("complete")
        compute = self.compute_snapshot()
        self._completed_training_seconds = compute["training_seconds"]
        self._completed_gpu_hours = compute["gpu_hours"]
        self._run_started_at = None
        self._run_stopped_at = None
        if self.writer is not None:
            self.writer.flush()
        finalize()
