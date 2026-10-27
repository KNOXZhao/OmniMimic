"""CPU-only supervisor regressions; no trainer or GPU subprocess is launched."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_and_evaluate.py"
    spec = importlib.util.spec_from_file_location("runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def attach_case(tmp_path):
    cwd = tmp_path / "trainer_cwd"
    run = cwd / "runs" / "experiment"
    run.mkdir(parents=True)
    data = cwd / "data" / "motions.pt"
    data.parent.mkdir()
    data.write_bytes(b"test fixture")
    config = {
        "run_name": run.name, "log_root": "runs",
        "data": {"library": "data/motions.pt"},
        "schedule": {"max_hours": 3.0},
    }
    (run / "config.json").write_text(json.dumps(config))
    proc = tmp_path / "proc" / "123"
    proc.mkdir(parents=True)
    (proc / "cwd").symlink_to(cwd, target_is_directory=True)
    argv = ["python", "-m", "supermimic.scripts.train", "--run-name", run.name,
            "--log-root", "runs", "--data.library", "data/motions.pt"]
    (proc / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv))
    return SimpleNamespace(run=run, data=data, config=config, proc=proc, argv=argv)


def test_attach_checks_module_run_data_and_resolves_trainer_cwd(runner, attach_case):
    case = attach_case
    verified = runner.validate_attached_trainer(123, case.run, case.data, "library", 3.0, case.proc.parent)
    assert verified["cwd"] == str(case.run.parent.parent)
    assert verified["configured_max_hours"] == 3.0


@pytest.mark.parametrize("change", ["module", "run_name", "log_root", "data", "unbounded"])
def test_attach_refuses_mismatched_or_unbounded_trainer(runner, attach_case, change):
    case = attach_case
    if change == "module":
        case.argv[2] = "unrelated.program"
    elif change == "run_name":
        case.argv[4] = "other_experiment"
    elif change == "log_root":
        case.argv[6] = "other_runs"
    elif change == "data":
        case.config["data"]["library"] = "data/different.pt"
    else:
        case.config["schedule"]["max_hours"] = None
    (case.proc / "cmdline").write_bytes(b"\0".join(x.encode() for x in case.argv))
    (case.run / "config.json").write_text(json.dumps(case.config))
    with pytest.raises(RuntimeError):
        runner.validate_attached_trainer(123, case.run, case.data, "library", 3.0, case.proc.parent)


def test_stale_checkpoint_never_yields_attached_completion(runner):
    initial = {"mtime_ns": 10, "size": 20, "inode": 30}
    final = {"mtime_ns": 30, "size": 25, "inode": 30}
    assert not runner.checkpoint_is_fresh(initial, initial, 20)
    assert not runner.checkpoint_is_fresh(None, initial, 20)
    assert runner.checkpoint_is_fresh(initial, final, 20)
    assert runner.completed_stage(True, True, False) == "incomplete_training_evaluated"
    assert runner.completed_stage(True, True, True) == "evaluated_budget_met_exit_unverified"
    assert runner.completed_stage(False, True, True) == "complete"


@pytest.mark.parametrize("analysis_code", [0, 7])
def test_pipeline_finishes_only_after_analysis_and_records_failures(runner, tmp_path, monkeypatch, analysis_code):
    import torch

    root = tmp_path / "repo"
    root.mkdir()
    data = root / "motions.pt"
    data.write_bytes(b"test fixture")
    run = root / "runs" / "test"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["run_and_evaluate.py", "--run-name", "test",
                        "--library", str(data), "--hours", "1", "--gpus", "0"])

    class FakeChild:
        pid = 987

        def wait(self):
            torch.save({"iteration": 100, "compute": {"training_seconds": 3601},
                        "config": {"schedule": {"total_iterations": 1000}}}, run / "ckpt_latest.pt")
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: FakeChild())

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n" if command[1] == "rev-parse" else "")
        if Path(command[1]).name == "eval_benchmark.py":
            (run / "benchmark_final.json").write_text(json.dumps({"summary": {"clips": 2, "failed": 0}}))
            return subprocess.CompletedProcess(command, 0)
        assert Path(command[1]).name == "summarize_runs.py"
        state = json.loads((run / "status.json").read_text())
        assert state["stage"] == "analyzing"
        assert "finished_at_utc" not in state
        kwargs["stdout"].write("analysis fixture output\n")
        return subprocess.CompletedProcess(command, analysis_code)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    if analysis_code:
        with pytest.raises(RuntimeError, match="analysis failed"):
            runner.main()
    else:
        runner.main()
    status = json.loads((run / "status.json").read_text())
    assert status["analysis_exit_code"] == analysis_code
    assert (run / "analysis.log").read_text() == "analysis fixture output\n"
    assert status["stage"] == ("failed" if analysis_code else "complete")
    assert status["training_exit_verified"] is True
    if analysis_code:
        assert status["failed_stage"] == "analyzing"


@pytest.mark.parametrize("hours", ["nan", "inf", "-inf"])
def test_nonfinite_budget_rejected_before_launch(runner, monkeypatch, hours):
    monkeypatch.setattr(sys, "argv", ["run_and_evaluate.py", "--run-name", "test",
                        "--library", "unused.pt", "--hours=" + hours])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
