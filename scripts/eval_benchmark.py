"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
"""

from __future__ import annotations

from pathlib import Path

import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize
from supermimic.eval.inputs import benchmark_source


def main(ckpt: Path, npz_dir: Path | None = None, clips: list[str] | None = None, clip_list: Path | None = None, manifest: Path | None = None, num_envs: int = 1024, device: str = "cuda:0", out: Path | None = None, library: Path | None = None, reward_mode: str | None = None) -> None:
    source = benchmark_source(library, npz_dir, clips, clip_list, manifest)
    if clip_list is not None:
        clips = [l.strip() for l in open(clip_list) if l.strip()]
    files = ([str(source.path)] if source.compact_library else
             [str(f) for f in discover_npz(source.path, names=clips, manifest=manifest)])
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    reward_mode = reward_mode or cfg["env"].get("reward_mode", "legacy")
    results = run_benchmark(policy, files, device=device, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False), compact_library=source.compact_library, reward_mode=reward_mode)
    s = summarize(results)
    print(f"\n{s['clips']} clips | failed {s['failed']} | full-clip mean {s['mean_full_clip_norm_return']:.4f} | full-clip length-weighted {s['full_clip_step_weighted_norm_return']:.4f}")
    print(f"executed-step mean {s['mean_executed_steps_norm_return']:.4f} | executed-step weighted {s['executed_steps_weighted_norm_return']:.4f} | completion {s['completion_fraction']:.1%}")
    print("Full-clip denominator: 0.1 * (frames - 1). Executed-step denominator: 0.1 * scored steps. Paper early-failure normalization is ambiguous; legacy norm_return is the executed-step score.")
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.full_clip_norm_return)[:10]
    for r in worst:
        print(f"  {r.name:32s} full-clip {r.full_clip_norm_return:.3f} executed {r.executed_steps_norm_return:.3f} steps {r.steps}/{r.planned_steps} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}.json"
    save_results(results, out, metadata={
        "checkpoint": str(ckpt.resolve()),
        "source": str(source.path.resolve()),
        "storage": "compact_fp16" if source.compact_library else "full_npz",
        "requested_num_envs": num_envs,
        "anchor_obs": cfg["env"].get("anchor_obs", False),
        "reward_mode": reward_mode,
        "training_reward_mode": cfg["env"].get("reward_mode", "legacy"),
    })
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
