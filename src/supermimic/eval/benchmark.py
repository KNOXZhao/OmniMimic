"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires within the clip horizon. We report both full-clip and
executed-step return denominators: the paper does not fully resolve that choice
for early failures. Full-clip score is primary; legacy ``norm_return`` retains the
executed-step meaning. Scores exclude any final-target hold step before auto-reset.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.metrics import return_scores
from supermimic.eval.inputs import benchmark_env_count
from supermimic.policy.nets import Policy, PolicyCfg


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float  # legacy alias: executed_steps_norm_return, NOT full-clip score
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    full_clip_norm_return: float = 0.0
    executed_steps_norm_return: float = 0.0
    planned_steps: int = 0
    completion_fraction: float = 0.0
    simulator_steps: int = 0  # may include one unscored auto-reset bookkeeping step


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False, compact_library: bool = False, reward_mode: str = "legacy") -> list[ClipResult]:
    if not npz_files:
        raise ValueError("benchmark requires at least one motion clip")
    packed = len(npz_files) == 1 and Path(npz_files[0]).suffix == ".pt"
    if packed and not compact_library:
        raise ValueError("a packed .pt input requires compact_library=True")
    num_envs = benchmark_env_count(num_envs, len(npz_files), packed)
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        compact_library=compact_library,
        reward_mode=reward_mode,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        if n_clips < 1:
            raise ValueError("benchmark motion library contains no clips")
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        # Packed files can contain fewer clips than requested environments. Extra
        # slots simulate a harmless duplicate but are inactive in the queue/scoring.
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(current, max=n_clips - 1)]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = current < n_clips
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        simulator_steps = torch.zeros_like(steps)
        horizons = torch.tensor([c.length - 1 for c in lib.clips], device=device)
        if bool((horizons < 1).any()):
            raise ValueError("benchmark clips must contain at least two frames")
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            planned = horizons[torch.clamp(current, max=n_clips - 1)]
            scored = active & (steps < planned)
            ret_sum += torch.where(scored, rew, 0.0)
            term_sum += torch.where(scored[:, None], env.reward_manager.term_rewards, 0.0)
            steps += scored.long()
            simulator_steps += active.long()
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        scores = return_scores(float(ret_sum[e]), lib.clips[ci].length, st, max_r)
                        failed = bool(term[e] & scored[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=scores.executed_steps_norm_return,
                            failed=failed,
                            fail_step=st if failed else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            full_clip_norm_return=scores.full_clip_norm_return,
                            executed_steps_norm_return=scores.executed_steps_norm_return,
                            planned_steps=scores.planned_steps,
                            completion_fraction=scores.completion_fraction,
                            simulator_steps=int(simulator_steps[e]),
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
                simulator_steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Legacy executed-step aggregate remains available, with explicit labels below.
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    planned = sum(r.planned_steps for r in results)
    out.update(
        primary_metric="mean_full_clip_norm_return",
        mean_full_clip_norm_return=sum(r.full_clip_norm_return for r in results) / max(n, 1),
        full_clip_step_weighted_norm_return=(
            sum(r.full_clip_norm_return * r.planned_steps for r in results) / max(planned, 1)
        ),
        mean_executed_steps_norm_return=mean_ret,
        executed_steps_weighted_norm_return=w_ret,
        completion_fraction=tot_steps / max(planned, 1),
        scored_steps=tot_steps,
        planned_steps=planned,
        legacy_norm_return_semantics="reward_sum / (0.1 * executed rewarded steps)",
        full_clip_norm_return_semantics="reward_sum / (0.1 * (clip_frames - 1)); unexecuted steps earn zero",
        protocol_note="Both denominators are reported because the paper does not fully specify early-failure normalization; macro and length-weighted averages are separate.",
    )
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path, metadata: dict | None = None) -> None:
    # Hash the exact ordered evaluation selection without rereading multi-GB
    # motion tensors. This identifies clip selection/lengths, not tensor contents.
    manifest = [{"name": r.name, "frames": r.frames} for r in results]
    digest = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    provenance = dict(metadata or {})
    provenance["ordered_clip_manifest_sha256"] = digest
    provenance["manifest_hash_semantics"] = "SHA256 of compact JSON list of ordered {name,frames}; does not hash motion tensor contents"
    Path(path).write_text(json.dumps({"summary": summarize(results), "metadata": provenance, "clips": [asdict(r) for r in results]}, indent=1))
