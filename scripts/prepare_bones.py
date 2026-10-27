#!/usr/bin/env python3
"""Prepare the already-authorized BONES-SEED G1 archive without downloading data.

Uses the repository's existing CSV conversion, Appendix-E approximation filter,
and compact-library builder. Raw/intermediate data remain in a task-only tmpfs
directory; final .pt, manifests, and logs go to --destination. 120 Hz is the
BONES source convention; the downloaded metadata has no explicit FPS column.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts/data"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--input-fps", type=float, default=120.0)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "disable")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MPLCONFIGDIR"] = str(args.scratch / "matplotlib")
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.raw_dir / "g1.tar.gz"
    metadata = args.raw_dir / "metadata/seed_metadata_v004.parquet"
    download_manifest = json.loads((args.raw_dir / "download_manifest.json").read_text())
    import numpy as np
    import pandas as pd
    table = pd.read_parquet(metadata)
    expected = dict(zip(table.move_g1_path, table.move_duration_frames))
    if len(expected) != len(table) or table.filename.nunique() != len(table):
        raise ValueError("Metadata paths and clip names must be unique")
    out_meta = args.destination / "preparation_manifest.json"
    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": download_manifest, "input_fps": args.input_fps,
        "fps_note": "120 Hz confirmed by the official dataset card and NVIDIA ProtoMotions documentation; metadata has no explicit FPS field",
        "source_convention_urls": [
            "https://huggingface.co/datasets/bones-studio/seed/blob/main/README.md",
            "https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst"],
        "units": {"root_translation": "source cm -> m", "root_euler": "source xyz degrees -> wxyz quaternion", "joints": "source degrees -> radians"},
        "output_fps": 50.0, "workers": args.workers,
        "metadata_clips": len(table), "metadata_frames": int(table.move_duration_frames.sum()),
        "metadata_hours_at_input_fps": float(table.move_duration_frames.sum() / args.input_fps / 3600),
        "script_hashes": {str(f.relative_to(PROJECT)): digest(f) for f in [
            PROJECT / "scripts/prepare_bones.py", PROJECT / "scripts/data/filter_bones_seed.py",
            PROJECT / "scripts/data/build_library.py", PROJECT / "src/supermimic/data/motion_convert.py"]},
    }
    save(out_meta, summary)
    print(f"Metadata: {len(table):,} clips; {summary['metadata_hours_at_input_fps']:.6f} h at {args.input_fps:g} Hz", flush=True)

    extract_marker = args.scratch / "extraction_complete.json"
    if not extract_marker.exists():
        started = time.monotonic()
        paths, total_bytes = [], 0
        with tarfile.open(archive, mode="r|gz") as source:
            for member in source:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "g1":
                    raise ValueError(f"Unexpected archive path: {member.name}")
                if member.isdir():
                    source.extract(member, path=args.scratch, filter="data")
                    continue
                if not member.isfile() or member.name not in expected:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                source.extract(member, path=args.scratch, filter="data")
                paths.append(member.name)
                total_bytes += member.size
                if len(paths) % 10000 == 0:
                    print(f"Extracted {len(paths):,}/{len(expected):,}; {total_bytes / 1e9:.2f} GB; {time.monotonic()-started:.0f}s", flush=True)
        if set(paths) != set(expected) or len(paths) != len(expected):
            raise ValueError("Archive members differ from the pinned metadata")
        save(extract_marker, {"clips": len(paths), "uncompressed_bytes": total_bytes,
                             "wall_seconds": time.monotonic()-started})
    summary["extraction"] = json.loads(extract_marker.read_text())
    save(out_meta, summary)

    # Check source units and column order against the converter on deterministic samples.
    from supermimic.data.motion_convert import G1_JOINT_NAMES, load_csv
    from scipy.spatial.transform import Rotation
    sample_rows = []
    sorted_paths = sorted(expected)
    for index in np.linspace(0, len(sorted_paths) - 1, 20).astype(int):
        name = sorted_paths[index]
        path = args.scratch / name
        frame = pd.read_csv(path)
        raw = load_csv(path)
        if [c for c in frame if c.endswith("_joint_dof")] != [n + "_dof" for n in G1_JOINT_NAMES]:
            raise ValueError(f"Joint order differs: {name}")
        np.testing.assert_allclose(raw[:, :3], frame[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy() / 100)
        np.testing.assert_allclose(raw[:, 7:], np.deg2rad(frame[[n + "_dof" for n in G1_JOINT_NAMES]].to_numpy()))
        quaternion = Rotation.from_euler("xyz", frame[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy(), degrees=True).as_quat()
        np.testing.assert_allclose(raw[:, 3:7], quaternion)
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite source values: {name}")
        sample_rows.append({"path": name, "frames": len(raw), "metadata_frames": int(expected[name]),
                            "root_height_median_m": float(np.median(raw[:, 2]))})
    summary["source_checks"] = {"samples": sample_rows, "units_and_order_verified": True}
    save(out_meta, summary)

    npz_dir = args.scratch / "npz"
    npz_dir.mkdir(exist_ok=True)
    conversion_manifest = args.destination / "conversion_manifest.csv"
    conversion_marker = args.scratch / "conversion_complete.json"
    if not conversion_marker.exists():
        import csv_to_npz_batch as converter
        jobs = [(str(args.scratch / name), str(npz_dir / (Path(name).stem + ".npz")), args.input_fps, 50.0)
                for name in sorted_paths]
        results = []
        started = time.monotonic()
        with mp.Pool(args.workers, initializer=converter._init_worker) as pool:
            for result in pool.imap_unordered(converter._convert_one, jobs, chunksize=16):
                results.append(result)
                if len(results) % 2000 == 0:
                    print(f"Converted {len(results):,}/{len(jobs):,}; {time.monotonic()-started:.0f}s", flush=True)
        results.sort(key=lambda r: r["source"])
        with conversion_manifest.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        failed = [r for r in results if not r["ok"]]
        if failed:
            raise RuntimeError(f"Conversion failures: {len(failed)}; inspect {conversion_manifest}")
        mismatches = []
        for r in results:
            name = str(Path(r["source"]).relative_to(args.scratch))
            if r["input_frames"] != expected[name]:
                mismatches.append({"path": name, "metadata_frames": int(expected[name]), "actual_frames": r["input_frames"]})
        save(conversion_marker, {"clips": len(results), "failures": 0, "input_frames": sum(r["input_frames"] for r in results),
             "output_frames": sum(r["output_frames"] for r in results), "duration_hours": sum(r["duration_s"] for r in results)/3600,
             "metadata_frame_mismatch_count": len(mismatches), "metadata_frame_mismatch_examples": mismatches[:20],
             "wall_seconds": time.monotonic()-started})
    summary["conversion"] = json.loads(conversion_marker.read_text())
    summary["conversion"]["manifest_sha256"] = digest(conversion_manifest)
    save(out_meta, summary)
    print("Conversion complete: " + json.dumps(summary["conversion"]), flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    filtered = args.destination / "manifest_filtered.csv"
    if not filtered.exists():
        with (args.destination / "filter.log").open("w") as log:
            subprocess.run([sys.executable, str(PROJECT / "scripts/data/filter_bones_seed.py"),
                "--npz-dir", str(npz_dir), "--workers", str(args.workers), "--out", str(filtered)],
                check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = list(csv.DictReader(filtered.open()))
    if len(rows) != len(expected):
        raise ValueError("Filter manifest does not cover every source clip")
    reasons = Counter(r["reason"] or "kept" for r in rows)
    kept = [r for r in rows if r["keep"] == "1"]
    summary["filter"] = {"raw_clips": len(rows), "after_name_filter": sum(not r["reason"].startswith("name:") for r in rows),
        "kept_clips": len(kept), "kept_frames": sum(int(r["frames"]) for r in kept),
        "kept_duration_hours_frame_count": sum(int(r["frames"]) for r in kept) / 50 / 3600,
        "kept_duration_hours_endpoints": sum(int(r["frames"])-1 for r in kept) / 50 / 3600,
        "reason_counts": dict(reasons), "manifest_sha256": digest(filtered),
        "paper_targets": {"after_name_filter": 129667, "kept_clips": 129514, "hours": 264.56},
        "note": "Independent implementation's documented Appendix-E approximation; exact paper keyword list is unavailable"}
    save(out_meta, summary)
    print("Filtering complete: " + json.dumps(summary["filter"]), flush=True)

    compact = args.destination / "bones_seed_compact.pt"
    if not compact.exists():
        with (args.destination / "build_library.log").open("w") as log:
            subprocess.run([sys.executable, str(PROJECT / "scripts/data/build_library.py"), "--npz-dir", str(npz_dir),
                "--manifest", str(filtered), "--out", str(compact), "--workers", str(min(args.workers, 32))],
                check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    import torch
    library = torch.load(compact, map_location="cpu", weights_only=False, mmap=True)
    ref = library["ref"]
    if ref.shape != (summary["filter"]["kept_frames"], 71) or len(library["names"]) != len(kept):
        raise ValueError("Compact library dimensions differ from filtering manifest")
    for start in range(0, len(ref), 1000000):
        if not torch.isfinite(ref[start:start+1000000]).all():
            raise ValueError("Nonfinite compact reference states")
    summary["compact"] = {"path": str(compact), "bytes": compact.stat().st_size, "sha256": digest(compact),
        "shape": list(ref.shape), "dtype": str(ref.dtype), "all_finite": True}
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(out_meta, summary)
    print("Prepared: " + json.dumps(summary["compact"]), flush=True)


if __name__ == "__main__":
    main()
