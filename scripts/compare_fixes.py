"""Eight-GPU controlled LAFAN1 comparison: four variants, two seeds.

Each run uses the same absolute actor-update budget and paper-sized networks.
The WM-warmup variant does additional work; measured time/GPU-hours are retained.
All policies are evaluated with the same post-physics reward/termination protocol.
Outputs are created exclusively, so invoking this script cannot overwrite a run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VARIANTS = ("legacy", "corrected", "feedback", "feedback_warmup")
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def fit(args):
    import torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig()
    cfg.run_name = args.run_name
    cfg.log_root = args.output
    cfg.seed = args.seed
    cfg.data.npz_dir = args.data
    cfg.env.num_envs = 4096
    cfg.env.reward_mode = "legacy" if args.variant == "legacy" else "post_physics"
    cfg.env.return_sampler_version = "legacy" if args.variant == "legacy" else "finite_horizon_v1"
    cfg.env.anchor_obs = args.variant in ("feedback", "feedback_warmup")
    cfg.schedule.wm_only_updates = 500 if args.variant == "feedback_warmup" else 0
    cfg.schedule.max_policy_updates = args.policy_updates
    cfg.schedule.log_interval = 250
    cfg.schedule.checkpoint_interval = 100_000
    cfg.schedule.max_hours = args.max_hours
    trainer = Trainer(cfg)
    started = time.monotonic()
    trainer.run()
    trainer.save("ckpt_final.pt")
    write_json(trainer.log_dir / "training_result.json", {
        "variant": args.variant, "seed": args.seed,
        "requested_policy_updates": args.policy_updates,
        "num_policy_updates": trainer.num_policy_updates,
        "num_wm_updates": trainer.num_updates,
        "env_steps": trainer.iteration,
        "compute": trainer.compute_snapshot(),
        "wall_seconds_after_construction": time.monotonic() - started,
        "sampler": trainer.command.sampler_diagnostics(),
        "gpu": torch.cuda.get_device_name(),
        "complete": trainer.num_policy_updates == args.policy_updates,
    })
    trainer.env.close()
    if trainer.num_policy_updates != args.policy_updates:
        raise RuntimeError("Training ended before the requested actor-update budget")


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("src", "scripts", "tests") for p in sorted((ROOT / folder).rglob("*.py"))}
    write_json(args.output / "experiment.json", {
        "variants": VARIANTS, "seeds": [0, 1], "gpus": list(range(8)),
        "actor_updates_per_run": args.policy_updates,
        "wm_warmup_rounds": {v: 500 if v == "feedback_warmup" else 0 for v in VARIANTS},
        "data": args.data.resolve(), "evaluation_reward_mode": "post_physics",
        "source_root": ROOT, "source_sha256": hashes,
        "started_utc_epoch": time.time(),
        "comparison_limits": "Training-set coverage, two seeds, short training; equal actor updates, not equal wall time. Only warmup variant adds WM rounds. No claim of paper reproduction.",
    })

    def job(item):
        gpu, variant, seed = item
        run_name = f"{variant}_seed{seed}"
        run = args.output / run_name
        run.mkdir()
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / "src"),
                   MUJOCO_GL="disable", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1")
        result = {"gpu": gpu, "variant": variant, "seed": seed, "started_utc_epoch": time.time()}
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--fit",
                   "--variant", variant, "--seed", str(seed), "--run-name", run_name,
                   "--output", str(args.output), "--data", str(args.data),
                   "--policy-updates", str(args.policy_updates), "--max-hours", str(args.max_hours)]
        with (run / "train.log").open("w") as log:
            result["train_exit_code"] = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if result["train_exit_code"] == 0:
            with (run / "eval.log").open("w") as log:
                command = [sys.executable, "-u", str(ROOT / "scripts/eval_benchmark.py"),
                           "--ckpt", str(run / "ckpt_final.pt"), "--npz-dir", str(args.data),
                           "--num-envs", "40", "--reward-mode", "post_physics",
                           "--out", str(run / "benchmark.json")]
                result["eval_exit_code"] = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        result["finished_utc_epoch"] = time.time()
        result["complete"] = result.get("train_exit_code") == 0 and result.get("eval_exit_code") == 0
        write_json(run / "status.json", result)
        print(json.dumps(result), flush=True)
        return result

    jobs = [(i, v, seed) for i, (v, seed) in enumerate((v, s) for v in VARIANTS for s in (0, 1))]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(job, jobs))
    write_json(args.output / "status.json", results)
    if not all(r["complete"] for r in results):
        raise SystemExit("One or more runs failed; inspect per-run logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy-updates", type=int, default=1024)
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name")
    options = parser.parse_args()
    if options.policy_updates < 1 or options.max_hours <= 0:
        parser.error("policy-updates and max-hours must be positive")
    (fit if options.fit else supervise)(options)
