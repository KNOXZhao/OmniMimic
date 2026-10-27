# OmniMimic

OmniMimic is an **independent implementation** of [SuperMimic](https://arxiv.org/abs/2610.28480), derived from the community implementation below, with audited training fixes and experiments on LAFAN1 and BONES-SEED. It is not the paper authors' training code. The Python package remains named `supermimic`.

Current full-BONES results, limitations, and the exact two-GPU training recipe are in [`docs/MATCHED_BONES_RESULTS.md`](docs/MATCHED_BONES_RESULTS.md). The matched 32,000-update comparison covers two fresh seeds per implementation; both OmniMimic runs also completed 9.8 hours. The planned four-run time-matched comparison remains incomplete because both AllMimic runs failed late. These experiments do not establish an exact reproduction of the paper. Datasets, checkpoints, and per-run result archives are not distributed in this code repository.

## Provenance and corrections

The starting implementation is [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), commit `d9be385b3eeb13a638fb33fb200d3bfd9be9839f` (MIT). The simulator is [mjlab](https://github.com/mujocolab/mjlab), commit `8bb2bc9e187797689ec82c78c35019afd2fd0376`. The original README is preserved in [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md); its experiment claims belong to the upstream author.

Our corrections align replay termination labels with their transitions, refill replay when resuming, preserve cumulative compute accounting, and report full-clip scores separately from scores conditional on the executed portion. Regression tests cover these cases. New training also defaults to finite-horizon reset sampling and post-physics reward/termination consistency; optional feedback, warmup, and stable world-model coordinates are documented in [`docs/REPRODUCTION_FIXES.md`](docs/REPRODUCTION_FIXES.md). The current matched protocol adds FP32 reference storage, explicit contact capacities, overflow/nonfinite checks, synchronized budget accounting, and actor-update checkpoints. Old checkpoints retain their recorded evaluation protocol. See [`docs/REPRODUCTION_NOTES.md`](docs/REPRODUCTION_NOTES.md) for historical limitations and protocol details.

**Historical continuation experiment:** the earlier eight-H200 continuation comparison did **not** demonstrate a better BONES policy: both corrections averaged 531 failures versus 509 for an equal-update legacy control and 485 for the preserved source checkpoint, over 129,943 motions. The acrobatic gap remained in that experiment. Its results and the short-run LAFAN comparison remain in [`docs/REPRODUCTION_FIXES.md`](docs/REPRODUCTION_FIXES.md). The later fresh-training results use a different, shared physical evaluation protocol and are reported separately in the current results document.

## Environment

Python 3.12, CUDA-capable Linux, and an NVIDIA GPU are required. The supplied `uv.lock` pins the upstream environment:

```bash
uv sync --frozen
MUJOCO_GL=disable OMP_NUM_THREADS=1 uv run pytest tests -q
```

The experiment runtime is recorded in [`environment.json`](environment.json). [`hardware.json`](hardware.json) records an earlier machine probe; its software versions are not the final runtime specification. Run commands from the repository root using `uv run` or the installed `.venv`. For `scripts/run_experiment.sh`, set `PYTHON_BIN` to override its interpreter selection.

## Data

LAFAN1 preparation produces 40 G1 clips, 441,140 frames at 50 Hz, or 2.45056 hours. The preparation script records provenance alongside the local data; converted clips can be stored under `data/npz/lafan1`.

```bash
uv run python scripts/prepare_lafan1.py --help
```

BONES-SEED requires already-approved dataset access. The downloader reads `HF_TOKEN` or a hidden prompt and never saves credentials:

```bash
uv run python scripts/download_bones.py --destination /path/to/bones/raw
uv run python scripts/prepare_bones.py --help
```

Our preparation converted all 142,220 source clips and retained 129,943 motions: 47,475,680 finite frames at 50 Hz, or 263.75 hours by frame count. Preserve the generated conversion/filter manifests with your data. The original G1 archive was 23.50 GB; the earlier FP16 compact library was 6.75 GB.

The current matched experiments use unquantized **FP32** references. The included `scripts/data/build_library.py` writes FP16; it does not produce that matched FP32 library. The current recipe therefore requires a separately prepared FP32 library, with its schema and precision requirement documented in [`docs/MATCHED_BONES_RESULTS.md`](docs/MATCHED_BONES_RESULTS.md).

The independent filtering retains **429 more motions** than the paper's 129,514, so these are not identical benchmark sets. Dataset licenses remain separate from the code license.

## Training

One GPU, original observation encoding and corrected sampler/rewards, all 40 LAFAN1 clips:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_experiment.sh \
  --run-name lafan1_paper_seed0 --data.npz-dir data/npz/lafan1 \
  --seed 0 --env.num-envs 4096 --policy-trainer.batch-size 4096 \
  --schedule.max-hours 1.2 --schedule.checkpoint-interval 500
```

Basic two-GPU example using a compact BONES-SEED library (the current matched protocol requires the additional explicit settings in the [results and training recipe](docs/MATCHED_BONES_RESULTS.md#training-recipe)):

```bash
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 scripts/run_experiment.sh \
  --run-name bones_paper_2gpu --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 4096 --policy-trainer.batch-size 4096 \
  --schedule.max-hours 9.8 --schedule.checkpoint-interval 500
```

Each GPU has its own simulator and replay; gradients are averaged and learning rates scale by the square root of GPU count. The paper specifies one- and two-GPU configurations. Additional GPUs support independent seeds and evaluation. `--env.anchor-obs` adds relative pose feedback to our goal encoding and must be reported as a separate variant; the paper does not fully specify its goal-frame convention.

Checkpoints omit replay and random-number-generator state. Resuming preserves weights, optimizer state and learning-rate position, then recollects replay; it is not a bitwise continuation. `--schedule.max-hours` caps each invocation, while compute totals accumulate across resumed segments. The synchronized training clock includes warmup, first-use compilation, and periodic checkpoints; it excludes environment construction, final checkpoint/audit work, and separate evaluation. Rank zero checks the deadline at each collection step and shares the stop decision across ranks.

## Evaluation

```bash
MUJOCO_GL=disable CUDA_VISIBLE_DEVICES=6 uv run python scripts/eval_benchmark.py \
  --ckpt runs/lafan1_paper_seed0/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 40
```

For the full dataset use `--library data/library/bones_seed_compact.pt` instead of `--npz-dir`. Every motion starts at frame zero without noise or domain randomization, with foot friction 1.0. The primary score divides earned rewards by the maximum over the full clip; unfinished steps contribute zero. The executed-step score is also saved for diagnosis. They differ sharply when a controller falls early. Report failure count, completion, macro mean and length-weighted mean together. This measures coverage of training motions, not held-out generalization.

This is the shipped **native evaluation**. The current cross-implementation tables use a separate shared physical evaluator and a tracking-only score, not this native reward or the paper's score. That experiment tooling is not included here; the native command above does not reproduce those comparison scores.

H200 timings are recorded as H200 timings, not substitutes for the paper's H100 measurements. No real-robot validation is included.

## Bounded experiments with automatic evaluation

The supervisor runs training to a stated budget, preserves an immutable final checkpoint, evaluates every motion, and writes analysis and a machine-readable status. Choose a new run name for every experiment:

```bash
uv run python scripts/run_and_evaluate.py \
  --run-name bones_1gpu_3h_seed0 --gpus 2 --hours 3 \
  --library data/library/bones_seed_compact.pt

uv run python scripts/run_and_evaluate.py \
  --run-name bones_2gpu_9h8_seed0 --gpus 3,4 --hours 9.8 \
  --library data/library/bones_seed_compact.pt
```

These commands run in the foreground; use a persistent terminal or service for a long unattended run. Each selected GPU uses 4,096 environments and a policy batch of 4,096 by default. A two-GPU 9.8-hour run budgets approximately 19.6 training GPU-hours, plus evaluation. The launcher refuses to overwrite an existing run. Rank zero checks the training deadline at each collection step; the run can overshoot by the work in that collection.

Read `runs/<run-name>/status.json`, `train.log`, `eval.log`, and `analysis.log`. A newly launched child reaches `stage: complete` only after training exits successfully, reaches its budget, and evaluation and analysis both succeed. `failed` and `incomplete_training_evaluated` are distinct outcomes. When attaching a watcher to an already-running process, the launcher verifies its command/configuration and checkpoint freshness but cannot retrieve its exit code; the corresponding success-like status is explicitly `evaluated_budget_met_exit_unverified`.

The historical `bones_pipeline_smoke` run tested all 129,943 motions after only four optimizer-update rounds. It had 128,108 failures and mean full-clip return 0.09517, confirming the pipeline worked but not establishing tracking performance. The `imag_ret` training-log field is a raw sum over 32 imagined steps; it is not on the same scale as normalized `train_ret`.
