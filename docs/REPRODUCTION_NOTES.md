# Reproduction notes: local H200 experiments

These notes describe the work performed for this reproduction on 2026-10-22. The implementation is an independent reproduction; the initial smoke checks establish that the training and evaluation pipeline runs. They do not establish the paper's tracking results or compute claims.

## Provenance

| Component | Source and revision |
|---|---|
| Research target | [SuperMimic, arXiv:2610.28480v1](https://arxiv.org/abs/2610.28480v1), submitted 2026-10-20 |
| Authors' project | [supermimic.github.io](https://supermimic.github.io/); no official training source release was found during this audit |
| Starting implementation | [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/d9be385b3eeb13a638fb33fb200d3bfd9be9839f), commit `d9be385b3eeb13a638fb33fb200d3bfd9be9839f`, with the local fixes below |
| Simulator framework | [mujocolab/mjlab](https://github.com/mujocolab/mjlab/tree/8bb2bc9e187797689ec82c78c35019afd2fd0376), commit `8bb2bc9e187797689ec82c78c35019afd2fd0376` |
| LAFAN1 G1 retargets | [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/ce1572906efe6157840e8474d5a0d7aa87481e74/g1), revision `ce1572906efe6157840e8474d5a0d7aa87481e74` |

The inherited README, `docs/EXPERIMENTS.md`, and earlier review/plan documents contain the independent upstream author's RTX 5090 experiments and claims. Those are provenance and background, not measurements from this H200 session. Newly generated evidence is under `runs/` and `results/`.

The downloaded LAFAN1 inputs, conversion parameters, and input/output SHA-256 hashes are recorded in [data/lafan1_provenance.json](../data/lafan1_provenance.json). The conversion produced 40 clips at 50 Hz, with 441,140 output frames; all checked numeric arrays were finite. Source CSV inputs were sampled at 30 Hz and use `xyzw` root quaternions. Runtime package versions and framework revisions are in [environment.json](../environment.json); hardware is recorded separately in [hardware.json](../hardware.json).

## Local correctness fixes

1. **Replay termination labels.** A replay row stores the action executed from its state and the resulting reward and termination flag. Window sampling previously read termination from the next row, shifting a positive label one transition early. Labels now use the same source rows as actions and rewards. Regression tests cover terminal transitions, timeouts, reset boundaries, ring-buffer wrapping, and adjacent state/action/reward/frame alignment.
2. **Resume warmup.** Replay is not saved in checkpoints. A resumed run now collects a fresh warmup before attempting world-model windows; its learning rate still follows the restored global iteration. Resume resets the fractional update accumulator. This preserves usable optimization state without implying a bitwise continuation: replay, simulator state, and random-number-generator state are not restored.
3. **Compute accounting.** New checkpoints store a versioned `compute` object with cumulative measured training seconds, integrated GPU-hours, GPU count at save, and a history-completeness flag. Changing GPU counts across resume segments does not retroactively change previous GPU-hours. Older checkpoints lacking timing are marked incomplete when resumed.
4. **Evaluation scores and horizon.** Evaluation reports full-clip and executed-step denominators separately, with explicit names. Scoring covers exactly `clip_frames - 1` reference transitions. A possible final-target hold required by simulator auto-reset is unscored; a failure after the scored horizon is excluded. Macro averages and length-weighted averages remain separate.

The relevant regression tests are `tests/test_replay_alignment.py`, `tests/test_training_resume.py`, and the evaluation/return-metric tests supplied with the local changes. Test logs are evidence of implementation checks, not tracking performance.

## Score semantics

Let `R` be the sum of rewards collected within the scored clip horizon, `F` the number of reference frames, and `S` the number of scored transitions actually executed. The implementation's maximum positive reward is `0.1` per control step.

| Field | Definition | Interpretation |
|---|---|---|
| `full_clip_norm_return` | `R / (0.1 * (F - 1))` | Primary evaluation score; the denominator stays fixed after an early failure |
| `executed_steps_norm_return` | `R / (0.1 * S)` | Tracking quality over executed transitions; can be high even when almost the entire clip is missed |
| Legacy `norm_return` | Alias for the executed-step score | Retained for compatibility; do not label it the full-clip score |
| `completion_fraction` | `S / (F - 1)` | Fraction of the planned horizon executed |
| `train/episode_norm_return` | Training episode reward divided by `0.1 * actual episode steps` | Sampled, noisy training episodes; distinct from deterministic clip evaluation |

The paper says early termination lowers normalized return, which supports keeping the full-motion denominator. It does not explicitly resolve the denominator formula, the frame/transition off-by-one convention, or macro versus length weighting. Both score definitions are therefore retained rather than declaring a numerically identical paper metric. LAFAN1 evaluation here plays every complete clip from its first frame; equivalence to every detail of the paper's LAFAN1 protocol is not established.

## Initial verified status

This is the initial smoke-test snapshot, not a final experiment report:

- `smoke_single` completed 50 simulator iterations and 4 optimizer-update rounds with 64 environments and reduced training batches. Its checkpoint predates the new compute metadata, so its cumulative GPU cost is unknown.
- `smoke_two_gpu` completed 50 iterations with a two-GPU compute record. This verifies the distributed execution path at smoke-test size; it is not a scaling benchmark.
- Evaluating the `smoke_single` checkpoint on all 40 LAFAN1 clips gave **40 failures**, mean full-clip score **0.001842**, and mean executed-step score **0.434586**. This deliberately short run is not a trained tracker; the difference between the scores illustrates why the denominator must be explicit.
- `lafan1_paper_seed0` was launched with the full configured network and batch sizes, 4,096 simulator environments, and a requested 1.2-hour training-loop budget. No result for that experiment is asserted by this static note. Inspect its generated summary and benchmark JSON for subsequent measurements.

The initial training, replay, and resume regression checks passed. Four LAFAN1 conversion tests passed; their log is [data/lafan1_conversion_tests.log](../data/lafan1_conversion_tests.log). Future summaries must continue to report failed or incomplete experiments as such.

## Remaining limitations and reproduction choices

- The paper does not fully specify the world-model state representation, normalizers, learned versus analytic heads, neural activations, precise reference-goal features, or all reset-sampler details. This implementation makes choices recorded in the inherited plan and each run's `config.json`. A run name containing `paper` denotes an intended configuration, not proof of exact equivalence.
- The selected run disables additional anchor-relative observations. Its 10-frame reference features and 71-dimensional simulator-state design remain independent implementation choices. The upstream results that enable anchor observations are a different input setting.
- The update-to-data ratio is interpreted as 0.4 update rounds per vectorized simulator step; the total-iteration and learning-rate schedule follow that interpretation. The paper leaves room for alternative counting conventions.
- Torso center-of-mass randomization is performed at simulator startup, while friction and encoder bias are resampled at resets. The paper describes episode context. The startup choice avoids repeated simulator constant recomputation and is a recorded difference.
- MuJoCo Warp/mjlab rewards and terminations inspect some kinematic quantities one physics substep before the final integrated state. The analytic imagined reward uses the predicted final state. This small timing discrepancy remains; fixing transition-label indexing does not remove it.
- The exact BONES-SEED filter manifest and all keyword/stationary-gate choices are unavailable. The authorized BONES-SEED download is complete; the local filter retains 129,943 clips, 429 more than the paper. LAFAN1 smoke or training results cannot establish the paper's full BONES-SEED coverage claim.
- These are H200 measurements. They cannot be presented as H100 timings or hardware-matched reproductions of published GPU-hour ratios. Concurrent jobs and compilation can also affect elapsed time; record the run configuration and hardware conditions with comparisons.
- Checkpoint timing covers the training loop, including warmup and compilation reached inside that loop. Environment construction, data preparation, and checkpoint evaluation are outside its scope. A 1.2-hour stop checked at logging boundaries can overshoot slightly. The analysis tool does not estimate work not yet logged.

## Refreshing summaries

From the repository root, use the environment created for this reproduction:

```bash
python scripts/summarize_runs.py --trusted-checkpoints
```

`--trusted-checkpoints` enables pickle deserialization only for local checkpoints within `--runs-root`; use it only for your own generated checkpoints. Without it, the tool reads JSON and TensorBoard logs and leaves checkpoint-only fields empty. Active or incomplete inputs are skipped with messages rather than interpreted as finished runs.

The script writes `results/run_summary.json`, `run_summary.csv`, `benchmark_summary.csv`, `scalar_history.json`, `training_curves.png`, and `evaluation_scores.png`. Training and evaluation have separate figures and labels. `--run NAME` selects runs; `--benchmark PATH` adds evaluation JSON stored elsewhere. A benchmark of mutable `ckpt_latest.pt` is not assigned a checkpoint iteration unless the input filename identifies it. The summaries contain observations only; they do not infer paper-level success or generalization.

## BONES-SEED preparation and complete pipeline check

The authorized HF source is `bones-studio/seed` revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`. The original 23,499,973,647-byte archive and metadata are persistent under `/efs/knoxzhao/supermimic_2610_28480/raw`. Conversion checked source-frame counts against metadata for every clip, reported zero failures, and checked source units and joint ordering on 20 samples. The compact library contains 129,943 retained motions, 47,475,680 frames, and only finite FP16 values. Full provenance is in `data/bones_preparation_manifest.json` and per-motion filter decisions in `data/bones_prepared/manifest_filtered.csv`.

`bones_pipeline_smoke` completed 50 simulator iterations and four optimizer-update rounds, then evaluated all 129,943 motions and generated its analysis with exit codes zero. Its mean full-clip score was 0.09517393, mean executed-step score 0.52021655, completion 0.13772775, and failure count 128,108. This is a pipeline baseline after approximately 20 seconds of measured training, not a trained reproduction. The packed library and original source archive are persistent; temporary CSV/NPZ files on tmpfs are expendable and reproducible from the archive.

The final targeted regression suite passed 28 tests plus three subtests, covering replay alignment, resume timing/warmup, benchmark inputs/metrics, and experiment lifecycle. The log is `validation/lifecycle_and_correctness_tests.log`. Full-size single- and two-GPU experiments use separate run directories. Their status files, final benchmarks and measured compute records are the source of truth for completion and performance; a launch is not a result.
