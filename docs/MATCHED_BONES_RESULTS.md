# Matched full-BONES experiments

OmniMimic is our independent, community-code-derived implementation; it is not the SuperMimic authors' code. These experiments compare it with [AllMimic](https://github.com/Shimingyi/allmimic) under a shared physical evaluation protocol. The Python package in this repository remains `supermimic`.

## Scope and protocol

Each implementation trained from scratch with seeds 0 and 1, using two H200 GPUs per run and the same unquantized FP32 references: **129,943 BONES motions, 47,475,680 frames at 50 Hz**. BONES evaluation covers these training motions. The separate 40-clip LAFAN1 set (441,140 frames) measures transfer to unseen motions. Our filtering retains 429 more BONES motions than the paper's 129,514; this and the H200 hardware differ from the paper's benchmark.

The planned four-run, 9.8-hour comparison is **incomplete**: both AllMimic training runs failed late, while both OmniMimic runs completed their budgets. The secondary comparison below uses the largest common saved checkpoint, **32,000 actor updates**, with **655,884,288 real transitions per run** including warmup. It matches updates and samples, not elapsed time. AllMimic used its recommended learning preset with eager policy rollout and default-mode compiled world model; this was an explicit runtime override, not its unmodified recommended runtime.

All tables use a separate, shared **physical tracking score**, not either implementation's native reward or the paper's score. It measures pose and velocity tracking without action-coordinate penalties. Each clip starts at frame zero; the score sums over its planned `frames − 1` transitions, with zero contribution after failure. Completion is the fraction of those transitions executed. Failures include root-height error above 0.25 m, gravity-z error above 0.8, or nonfinite state. Reported scores and completion are means over clips.

Evaluation disables reset/observation/action noise and domain randomization, uses friction 1.0, FP32 inference with TF32 disabled, 4,096 environments for BONES and 40 for LAFAN, and contact/constraint capacities of 128/512. All accepted evaluations passed the capacity and interface checks. The shared evaluator is separate experiment tooling and is **not shipped in this repository**. `scripts/eval_benchmark.py` provides native evaluation with a different reward and protocol; it cannot be used to equate its scores with these tables.

## Matched 32,000-update results

| Method | Seed | Training hours | BONES score | BONES completion | BONES failed / 129,943 | LAFAN score | LAFAN completion | LAFAN failed / 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniMimic | 0 | 8.069 | 0.917267 | 99.976% | 59 | 0.583409 | 75.283% | 18 |
| OmniMimic | 1 | 8.218 | 0.920305 | 99.975% | 64 | 0.597728 | 74.869% | 19 |
| AllMimic | 0 | 4.885 | 0.865683 | 99.753% | 483 | 0.621651 | 81.403% | 16 |
| AllMimic | 1 | 4.799 | 0.247875 | 66.583% | 72,765 | 0.012727 | 3.653% | 40 |

The two-seed mean BONES score is **0.918786** for OmniMimic (range 0.917267–0.920305) and **0.556779** for AllMimic (0.247875–0.865683). Mean failures are 61.5 and 36,624 respectively. On unseen LAFAN, mean scores are **0.590568** (0.583409–0.597728) and **0.317189** (0.012727–0.621651); mean failures are 18.5 and 28. AllMimic seed 0 performs better on LAFAN than either OmniMimic seed, so the mean alone obscures a substantial seed effect. Two seeds do not support a statistical-significance claim.

The earlier acrobatic failure gap is substantially reduced under this protocol. Among filename-defined groups, OmniMimic seeds 0/1 fail 1/16 of 160 handstands, 1/6 of 120 cartwheels, 0/1 of 54 flips, and 0/0 of 22 side rolls. AllMimic fails 160/160, 112/116, 54/54, and 20/22 respectively. These groups are filename-derived, not official annotations; successful completion does not imply precise tracking.

## Completed OmniMimic 9.8-hour runs

| Seed | Actor updates | Real transitions | BONES score | BONES completion | BONES failed / 129,943 | LAFAN score | LAFAN completion | LAFAN failed / 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38,961 | 798,449,664 | 0.927470 | 99.982% | 44 | 0.593086 | 74.606% | 19 |
| 1 | 38,238 | 783,638,528 | 0.910813 | 99.989% | 29 | 0.597779 | 75.369% | 17 |

Mean BONES score is **0.919141** (range 0.910813–0.927470), with 99.985% completion and 36.5 failures. Mean LAFAN score is **0.595433** (0.593086–0.597779), with 74.987% completion and 18 failures. There is no completed AllMimic 9.8-hour comparator; these are standalone OmniMimic results, not a completed time-matched comparison.

## Training recipe

After `uv sync --frozen`, run from the repository root. Supply your own authorized, unquantized FP32 motion library and a fresh run name:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m supermimic.scripts.train \
  --run-name bones_matched_seed0 --log-root runs --seed 0 \
  --data.library /path/to/bones_canonical_fp32.pt \
  --env.no-library-fp16 --env.num-envs 4096 \
  --env.nconmax 128 --env.njmax 512 \
  --env.warn-overflow --env.monitor-overflow \
  --env.anchor-obs --env.reward-mode post_physics \
  --env.return-sampler-version finite_horizon_v1 \
  --env.min-start-steps 10 \
  --schedule.total-iterations 320000 \
  --schedule.warmup-iterations 65 \
  --schedule.lr-clock credited_collection_steps \
  --schedule.policy-checkpoint-interval 8000 \
  --schedule.checkpoint-interval 1000000 \
  --schedule.log-interval 250 \
  --schedule.max-hours 9.8
```

Repeat with `--seed 1` and a new run name for the second independent seed. The settings are explicit because several repository defaults preserve earlier experiments. Each rank uses 4,096 simulator environments and a policy batch of 4,096; world-model batches contain 2,048 samples, with 10 world-model and 20 critic updates per actor update. The imagined horizon is 32. The first 64 collections populate replay without credited actor-training progress. Relative anchor feedback is enabled here and should be reported as an implementation choice, since the paper does not fully specify the goal-frame convention.

The compact library is a trusted `torch.save` dictionary with `ref` (FP32 tensor shaped `[frames, 71]`), `names` (ordered clip names), `starts`, and `lengths`. Columns are root position (3), root quaternion in WXYZ order (4), joint positions (29), root linear velocity in world coordinates (3), root angular velocity in world coordinates (3), and joint velocities (29), using the implementation's G1 joint order. References must be prepared at 50 Hz. The included `scripts/data/build_library.py` currently writes FP16: **do not use it to claim the matched FP32 protocol**. `--env.no-library-fp16` preserves existing FP32 values; it cannot restore information lost through earlier quantization. Datasets and trained weights are not included here.

For identity checks, the experiment's ordered BONES name/frame manifest SHA-256 was `3abd3a000065d8e77b90886287135de4f4aeffb4e6dcecb1139488af701e6797`; the concatenated FP32 reference-array SHA-256 was `57c011a411a81702243e1f8e6f30edb86eb3e861fc06709e2a79d0304107238f`. Matching counts alone does not establish matching reference arrays.

The synchronized training clock starts at rank entry before the initial reset, includes warmup, compilation, and periodic checkpoint writes, and excludes environment construction, final checkpoint/audit work, and evaluation. Rank zero checks the deadline at each collection and shares the decision across ranks; one collection's work can extend beyond the deadline. Actor checkpoints are saved every 8,000 updates. Replay and RNG state are not saved, so resuming is not a bitwise continuation of these fresh runs.

## Validation limits

Four initial OmniMimic BONES evaluations stopped before scoring because the independent interface oracle reconstructed quaternions unstably. A separately versioned validation-only correction used the actual simulator/reference quaternions at the same tolerance. Controller inputs/actions, simulation, and metric code were unchanged; the four replacement evaluations passed, and the failed attempts were retained in the experiment record.

A separate AllMimic native-path check still showed the seed gap: 492/72,549 BONES failures for seeds 0/1, versus 483/72,765 under the shared evaluator. LAFAN seed 1 failed all 40 clips in both. TF32 and evaluator path changed together in this diagnostic, and many per-clip outcomes differed; it is not a one-factor precision test or evidence of trajectory equivalence. Native reward magnitudes remain incomparable to the common score.

These results show training-set coverage and limited transfer under the stated simulator protocol. They do not establish the paper's exact result, hardware timing, or real-robot performance. Historical continuation experiments and their different native metrics remain documented in [REPRODUCTION_FIXES.md](REPRODUCTION_FIXES.md).
