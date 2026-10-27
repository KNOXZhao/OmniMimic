# Reproduction corrections, October 2026

The gap audit identified a reset-distribution error and a mismatch between the simulator reward and the differentiable reward. These changes correct those implementation problems. They do not establish that the paper's benchmark has been reproduced.

Both eight-GPU comparisons are complete. Numerical correctness checks pass, but the short learning experiments do **not** establish better overall tracking. In the mature BONES comparison, all eight continuations have more failed clips than the preserved source checkpoint. Keep that original checkpoint as the best choice by failure count among these compared policies; the new checkpoints are experimental.

## Corrected defaults

`env.return_sampler_version=finite_horizon_v1` divides each discounted return by the maximum discounted reward over its **planned** remaining reference/time-limit horizon. Failure never shortens that denominator. A perfect short motion and a perfect long motion both score one. The first observation initializes a bin's estimate; subsequent observations use the existing EMA. The 50/40/10 uniform/hard/memory mixture remains.

Training starts must provide at least `env.min_start_steps=10` distinct reference transitions. Ineligible tail bins have no reset probability. The extra final-pose hold is excluded, and reward is credited to the source reset frame rather than the next target frame. Start-mode evaluation still covers short clips. `sampler_diagnostics()` reports the probability of starts with ten or fewer transitions and fewer than eight transitions.

`env.reward_mode=post_physics` uses raw final qpos/qvel and one shared compiled forward-kinematics calculation for all eight simulator reward terms. The reference is reanchored to that same final state, matching the analytic imagination head. Terminations also read the final state. Torso anchoring, weights, widths, thresholds, and timestep are unchanged. This fixes timing consistency; it does not resolve the paper's ambiguous root/anchor convention.

## Explicit variants

- `--env.anchor-obs` includes the existing nine-dimensional robot-to-reference position and orientation feedback. Tests verify real/simulated observation agreement and nonzero actor gradients through that feedback. This is an explicitly changed goal representation; exact equivalence to the authors' unspecified encoding is unproven.
- `--schedule.wm-only-updates 500` fits the world model for 500 update rounds before actor/critic optimization. This is an ablation inspired by the external reproduction, not a newly asserted paper requirement.
- `--world-model.state-spec.coordinate-mode body_gravity` uses smooth body-frame world-model features and increments, avoiding heading extraction in that module. It requires retraining. The default remains `heading`; reward and critic still have separate heading operations. This option alone does not remove every potential rotation singularity from BPTT.
- `--schedule.max-policy-updates N` sets an absolute actor-update budget, including updates already in a resumed checkpoint. It prevents overshoot even when an update round contains several actor updates. World-model and actor counters are saved separately.

## Existing checkpoints

Old checkpoint evaluation automatically uses `legacy` reward/termination behavior when its configuration has no reward-mode field. `scripts/eval_benchmark.py --reward-mode post_physics` explicitly overrides that choice for a common-protocol comparison; both training and evaluation modes are recorded in the result.

To continue an old baseline, explicitly select `--env.reward-mode legacy --env.return-sampler-version legacy`. Corrected sampling must not consume the old raw-return statistics. Versioned sampler checkpoints also verify a hash of ordered clip names, starts, lengths, and bin width before restoring statistics. This identifies layout, not motion tensor contents. Old flat sampler arrays remain an explicit legacy-only compatibility exception with a warning. The initial schema-1 experimental checkpoints reset sampler statistics when loaded by schema 2; network and optimizer weights remain usable. Changing coordinate mode or anchor inputs during resume is rejected before network weights are loaded, including coordinate changes that preserve tensor shapes.

These fixes do not repair already-trained weights. Keep historical checkpoints and retrain or explicitly label a fine-tuning run. Replay and RNG state remain outside checkpoints; distributed resume also retains the existing rank-zero sampler-state limitation.

## Controlled comparison

`scripts/compare_fixes.py` runs four variants with seeds 0 and 1 on GPUs 0 through 7: legacy; corrected sampler/reward; corrected plus feedback; corrected plus feedback and 500 world-model-only rounds. Each uses 4,096 environments, the original network/batch sizes, and 1,024 actor updates by default. Every final checkpoint is evaluated over all 40 LAFAN1 clips under the common post-physics protocol. The warmup variant performs additional WM work, so equal actor updates are not equal compute. GPU time and update counts are retained.

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 .venv/bin/python scripts/compare_fixes.py \
  --output /absolute/new/output/directory \
  --data /efs/knoxzhao/HOI/sim/SuperMimic/data/npz/lafan1 \
  --policy-updates 1024
```

The output directory must not exist. Per-run training/evaluation exit codes, source hashes, configuration, sampler diagnostics, checkpoint, full-clip scores, and failures are preserved. A failure or early budget stop is not reported as completion.

The primary comparison is training-set motion coverage at a short fixed optimization budget. It does not establish held-out generalization, acrobatic recovery on BONES, or the paper's full-budget performance. BONES filtering still differs from the paper by 429 motions, and our H200 timing is not an H100 measurement.

### Completed short LAFAN1 check

All eight runs completed exactly 1,024 actor updates, then evaluated all 40 clips under post-physics scoring. Results below are two-seed means, with seed-specific full-clip returns retained to show variability.

| Variant | Mean full-clip return | Mean failed clips / 40 | Returns, seeds 0 / 1 |
|---|---:|---:|---|
| Legacy | 0.14972 | 29.0 | 0.00550 / 0.29395 |
| Corrected sampler/reward | 0.13317 | 30.5 | 0.25690 / 0.00944 |
| Corrected + feedback | 0.19213 | 23.0 | 0.15615 / 0.22810 |
| Corrected + feedback + WM warmup | 0.24896 | 24.5 | 0.27466 / 0.22326 |

These short runs have large seed variation and do **not** demonstrate that sampler/reward changes alone improve tracking. Feedback plus warmup has the highest observed mean return, while feedback without warmup has fewer mean failures. Neither is a paper-level result or a statistically established winner. Each policy covers the whole motion set; the parallel runs are independent experimental variants, not a policy ensemble.

The exact artifacts are available through `runs/fix_validation_lafan_1024/` and `validation/reproduction_fixes/lafan_1024/`. Source commit: `3fb4e81faa15c8315211e3290173fe06c3fb7106`. The source hashes, both seeds, completed exit statuses, scoring denominators, and compute are checked by the saved summarizer. A separate continuation experiment starts all arms from the same mature BONES checkpoint to examine sampler and reward corrections independently of initial learning.

### Mature BONES continuation protocol

The continuation comparison uses the preserved `bones_2gpu_9h8_seed0/ckpt_final.pt` in all eight runs. Four settings (legacy, sampler only, reward/termination only, both corrections) each receive two continuation seeds and exactly 512 additional actor and world-model update rounds. Every run keeps the original networks, heading coordinates, and goal inputs without anchor feedback. This tests adaptation of the existing policy, not the feedback or body-gravity variants.

Each continuation uses one H200 and 4,096 environments. All three base learning rates become 0.0002 after changing from two training ranks to one; optimizer moments and steps are preserved, and the global cosine schedule continues. Replay is refilled for 40 local environment steps. The absolute actor target is 38,196, starting from 37,684. Incremental time excludes the source checkpoint's approximately 19.61 GPU-hours and any wait for an available GPU.

The full 129,943-clip compact BONES library is evaluated with 8,192 environments under the same post-physics reward/termination protocol, including a fresh evaluation of the source checkpoint. Legacy sampler arms retain their old statistics; corrected sampler arms reset incompatible statistics. Thus the sampler comparison includes that reset, and the reward comparison includes termination timing. Two continuation seeds from one checkpoint are not two independent pretraining runs. Source hashes, input hashes, configurations, exit statuses, per-clip results, and compute records are retained in `validation/reproduction_fixes/bones_finetune_512/`.

The source checkpoint under this common protocol scores 0.917877 mean full-clip return, with 485 failures and 99.778% completed reference steps. The archived 0.917575 / 489-failure result used legacy scoring and 1,024 evaluation environments; its difference from this rerun is not a learning gain. Filename-defined handstand, flip, cartwheel, and roll groups contain only 486 motions but account for 410 of the 485 failures. The high aggregate score therefore does not establish acrobatic coverage. These categories are inferred from names, not ground-truth skill annotations.

### Completed mature BONES results

All eight continuations and full-library evaluations exit successfully. Every checkpoint ends at global environment step 95,569 and actor/world-model round 38,196. Independent CPU checkpoint inspection verifies finite weights and optimizer tensors, the continued learning rate, source identity, library layout identity, and compute accounting. Each run adds 7.36–8.10 minutes of training on one H200; the eight increments total 1.022 GPU-hours, excluding evaluation and setup.

| Setting | Mean full-clip return | Mean failed clips / 129,943 | Returns, seeds 0 / 1 | Failed clips, seeds 0 / 1 |
|---|---:|---:|---|---|
| Preserved source, no continuation | 0.917877 | 485 | Single source checkpoint | 485 |
| Legacy continuation control | 0.922552 | 509 | 0.923532 / 0.921572 | 503 / 515 |
| Sampler correction only | 0.918285 | 508 | 0.911103 / 0.925467 | 498 / 518 |
| Reward/termination correction only | 0.917026 | 543 | 0.919660 / 0.914391 | 564 / 522 |
| Both corrections | 0.914408 | 531 | 0.907059 / 0.921758 | 534 / 528 |

The corrected sampler eliminates reset probability with fewer than eight usable transitions (0%, versus approximately 47% in the continued legacy samplers). This confirms the intended distribution correction. It does **not** establish that a short change of curriculum improves the learned policy. Both-correction checkpoints recover 40 of the source failures each, but introduce 89 and 83 new failures. All 160 handstands and all 54 flips still fail in every arm. Median handstand survival rises from 0.25 seconds in the source to 0.60/0.38 seconds with both corrections, which is measurable partial progress but not successful motion tracking.

The sampler-only arm recovers 27/4 previously failing cartwheels, but those recovered sets have no overlap across its two seeds. Mean return and failure count also move differently in the legacy control. These results do not support promoting a new checkpoint as an overall improvement, nor do they establish how a complete retraining with feedback or body-gravity coordinates would perform. Those representation changes require new training and are outside this continuation comparison.

Review `runs/fix_validation_bones_512/baseline_comparison.png` or `.csv` for the aggregate comparison, `per_clip_analysis.json` and each run's `per_clip_vs_mature.json` for recovered/regressed clips, and `validation/reproduction_fixes/bones_finetune_checkpoint_audit.json` for the independent checkpoint checks. Continuation training/evaluation Python source: commit `6c8fd120ab2ce70bc3f12fcd53d83a797b4b463e`. The source-checkpoint rerun used `3fb4e81` with identical evaluation/reward/termination code; its hashes and protocol are recorded in `bones_mature_baseline_provenance.json`. The source checkpoint and previous benchmark artifacts remain unchanged. All validation GPU jobs are complete.

### Numerical validation

The final CPU suite passes 102 tests (19 hardware/data-dependent skips, plus three passing subtests). A GPU transition test covers nonzero scene origins, yaw changes, high velocities, inverted poses, and partial resets. The separate body-gravity BONES smoke checks 6,336 noisy randomized transitions across six verified compact-library motions before reset, including 162 terminations and nine timeouts: maximum reward disagreement is 3.73e-8 and termination labels match exactly. It completes 24 world-model rounds, 16 actor updates, and six-clip queue evaluation with finite losses and weights. Its six failures are expected for a safety smoke and provide no recovery-performance evidence.
