#!/usr/bin/env python3
"""Run a bounded experiment and deterministic benchmark, recording durable status.

Uses only the already-prepared local data and environment. No credentials are read.
Can attach to a same-host running trainer by PID and Linux process start time.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    try:
        tail = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if tail[0] == 'Z' else tail[19]
    except (OSError, IndexError):
        return None


def option_value(argv, name):
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(name + '='):
            return item.split('=', 1)[1]
    return None


def resolve_from(path, cwd):
    path = Path(path)
    return (path if path.is_absolute() else cwd / path).resolve()


def validate_attached_trainer(pid, run, data_path, data_key, hours, proc_root=Path('/proc')):
    """Verify a live trainer's identity against the persisted run configuration."""
    proc = proc_root / str(pid)
    argv = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
    if not any(argv[i:i + 2] == ['-m', 'supermimic.scripts.train'] for i in range(len(argv) - 1)):
        raise RuntimeError('attach PID is not running the supermimic training module')
    if option_value(argv, '--run-name') != run.name:
        raise RuntimeError('attach PID run name does not match --run-name')
    # Resolve relative command/config paths using the trainer's working directory,
    # not the supervisor's checkout. Failure to inspect it is a verification error.
    cwd = (proc / 'cwd').resolve(strict=True)
    command_log_root = option_value(argv, '--log-root')
    if command_log_root is None or (resolve_from(command_log_root, cwd) / run.name).resolve() != run.resolve():
        raise RuntimeError('attach PID log root does not match the requested run directory')
    config = json.loads((run / 'config.json').read_text())
    if config.get('run_name') != run.name:
        raise RuntimeError('persisted configuration run name does not match the attach target')
    if (resolve_from(config['log_root'], cwd) / run.name).resolve() != run.resolve():
        raise RuntimeError('persisted configuration log root does not match the attach target')
    configured_data = config.get('data', {}).get(data_key)
    command_data = option_value(argv, '--data.' + data_key.replace('_', '-'))
    if configured_data is None or resolve_from(configured_data, cwd) != data_path:
        raise RuntimeError('persisted training data does not match the requested evaluation data')
    if command_data is None or resolve_from(command_data, cwd) != data_path:
        raise RuntimeError('attach PID data argument does not match the requested evaluation data')
    limit = config.get('schedule', {}).get('max_hours')
    if limit is None or not math.isfinite(float(limit)) or not 0 < float(limit) <= hours:
        raise RuntimeError('attached trainer must have a finite max-hours limit no greater than --hours')
    return {'command': argv, 'cwd': str(cwd), 'configured_max_hours': float(limit)}


def checkpoint_stat(path):
    try:
        stat = path.stat()
        return {'mtime_ns': stat.st_mtime_ns, 'size': stat.st_size, 'inode': stat.st_ino}
    except FileNotFoundError:
        return None


def checkpoint_is_fresh(initial, final, attached_at_ns):
    return final is not None and final != initial and final['mtime_ns'] >= attached_at_ns


def completed_stage(attached, budget_met, fresh_checkpoint):
    if attached:
        return 'evaluated_budget_met_exit_unverified' if budget_met and fresh_checkpoint else 'incomplete_training_evaluated'
    return 'complete' if budget_met else 'incomplete_training_evaluated'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--gpus', default='0')
    p.add_argument('--eval-gpus', default=None)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--seed', type=int, default=0)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument('--library', type=Path)
    data.add_argument('--npz-dir', type=Path)
    p.add_argument('--eval-num-envs', type=int, default=1024)
    p.add_argument('--attach-pid', type=int)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--policy-batch', type=int, default=4096)
    p.add_argument('--checkpoint-interval', type=int, default=1000)
    args = p.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0 or '/' in args.run_name or args.run_name in ('.', '..'):
        p.error('positive hours and a single-component run name are required')
    gpu_list = args.gpus.split(',')
    if not all(g.isdigit() for g in gpu_list) or len(set(gpu_list)) != len(gpu_list):
        p.error('--gpus must be unique integer GPU indices')
    if args.attach_pid is not None and args.attach_pid <= 0:
        p.error('--attach-pid must be positive')
    data_path = (args.library or args.npz_dir).resolve()
    if not data_path.exists():
        p.error(f'data path does not exist: {data_path}')
    run = ROOT / 'runs' / args.run_name
    if run.exists() and args.attach_pid is None:
        p.error('run already exists; choose a new name instead of overwriting it')
    run.mkdir(parents=True, exist_ok=True)
    state = {'run': args.run_name, 'started_at_utc': now(), 'stage': 'preparing',
             'requested_training_hours': args.hours, 'gpus': gpu_list,
             'data': str(data_path), 'source_root': str(ROOT), 'supervisor_pid': os.getpid()}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    state['source_commit'] = git.stdout.strip() if git.returncode == 0 else None
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
    state['source_dirty'] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    state['source_commit_scope'] = ('supervisor checkout; attached trainer source revision is unverified'
                                    if args.attach_pid else 'checkout used to launch training and evaluation')
    state_path = run / 'status.json'

    def save():
        state['updated_at_utc'] = now()
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_path)

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, MUJOCO_GL='disable',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', NCCL_DEBUG='WARN')
    env.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    env.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.cache/torchinductor'))
    latest = run / 'ckpt_latest.pt'
    attached_at_ns = None
    initial_checkpoint = None
    try:
        if args.attach_pid:
            identity = process_identity(args.attach_pid)
            if identity is None:
                raise RuntimeError('trainer PID is not alive; refusing to infer completion')
            verified = validate_attached_trainer(args.attach_pid, run, data_path,
                                                'library' if args.library else 'npz_dir', args.hours)
            if process_identity(args.attach_pid) != identity:
                raise RuntimeError('trainer changed or exited during attach verification')
            attached_at_ns = time.time_ns()
            initial_checkpoint = checkpoint_stat(latest)
            state.update(stage='training_attached', trainer_pid=args.attach_pid,
                         process_start_ticks=identity,
                         attach_verification=verified, attached_at_ns=attached_at_ns,
                         initial_checkpoint_stat=initial_checkpoint,
                         training_exit_code=None, training_exit_verified=False,
                         note='PID disappearance does not verify a clean exit; checkpoint freshness and budget are checked separately.')
            save()
            while process_identity(args.attach_pid) == identity:
                time.sleep(10)
        else:
            cmd = [sys.executable]
            if len(gpu_list) > 1:
                cmd += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(len(gpu_list))]
            cmd += ['-m', 'supermimic.scripts.train', '--run-name', args.run_name,
                    '--log-root', str(ROOT / 'runs'), '--seed', str(args.seed),
                    '--env.num-envs', str(args.num_envs), '--policy-trainer.batch-size', str(args.policy_batch),
                    '--schedule.max-hours', str(args.hours), '--schedule.checkpoint-interval', str(args.checkpoint_interval)]
            cmd += ['--data.library' if args.library else '--data.npz-dir', str(data_path)]
            state.update(stage='training', training_command=cmd)
            with (run / 'train.log').open('w') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state['trainer_pid'] = child.pid
                save()
                code = child.wait()
            state['training_exit_code'] = code
            state['training_exit_verified'] = True
            if code:
                raise RuntimeError(f'training failed with exit code {code}; see train.log')
        if not latest.exists():
            raise RuntimeError('trainer ended without a checkpoint')
        final_checkpoint = checkpoint_stat(latest)
        fresh_checkpoint = (checkpoint_is_fresh(initial_checkpoint, final_checkpoint, attached_at_ns)
                            if args.attach_pid else True)
        state['final_checkpoint_stat'] = final_checkpoint
        state['checkpoint_fresh_since_attach'] = fresh_checkpoint if args.attach_pid else None
        final = run / 'ckpt_final.pt'
        shutil.copy2(latest, final)
        # This checkpoint was generated by the local trainer, not downloaded.
        import torch
        ck = torch.load(final, map_location='cpu', weights_only=False)
        compute = ck.get('compute', {})
        state['checkpoint_iteration'] = int(ck['iteration'])
        state['compute'] = compute
        total = int(ck['config']['schedule']['total_iterations'])
        elapsed = float(compute.get('training_seconds', 0))
        budget_met = state['checkpoint_iteration'] >= total or elapsed >= args.hours * 3600
        state['checkpoint_budget_met'] = budget_met
        state['training_budget_completed'] = budget_met and fresh_checkpoint
        state['budget_criterion'] = 'checkpoint total iterations or cumulative recorded training seconds'
        state['stage'] = 'evaluating'
        del ck
        env['CUDA_VISIBLE_DEVICES'] = args.eval_gpus or args.gpus
        command = [sys.executable, str(ROOT / 'scripts/eval_benchmark.py'), '--ckpt', str(final),
                   '--library' if args.library else '--npz-dir', str(data_path),
                   '--num-envs', str(args.eval_num_envs), '--out', str(run / 'benchmark_final.json')]
        state['evaluation_command'] = command
        save()
        with (run / 'eval.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['evaluation_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'evaluation failed with exit code {result.returncode}; see eval.log')
        state['benchmark'] = json.loads((run / 'benchmark_final.json').read_text())['summary']
        state['stage'] = 'analyzing'
        save()
        # Per-run output avoids races between concurrent experiment reporters.
        with (run / 'analysis.log').open('w') as log:
            analysis = subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_runs.py'), '--trusted-checkpoints',
                            '--run', args.run_name, '--out-dir', str(run / 'analysis')], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
        state['analysis_exit_code'] = analysis.returncode
        if analysis.returncode:
            raise RuntimeError(f'analysis failed with exit code {analysis.returncode}; see analysis.log')
        state['stage'] = completed_stage(bool(args.attach_pid), budget_met, fresh_checkpoint)
        state['finished_at_utc'] = now()
        save()
    except Exception as exc:
        state.update(failed_stage=state['stage'], stage='failed', error=f'{type(exc).__name__}: {exc}', finished_at_utc=now())
        save()
        raise
    finally:
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
