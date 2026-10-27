#!/usr/bin/env python3
"""Download the pinned public G1 LAFAN1 retargets and optionally convert to 50 Hz.

Run with the reproduction environment's Python. Downloads need only the standard
library; conversion additionally needs the project's MuJoCo/mjlab dependencies.
The motion data retain their source's CC BY-NC-ND 4.0 terms. This script does not
grant redistribution rights to raw data or converted derivatives.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/g1?recursive=false&expand=true"
PROJECT = Path(__file__).resolve().parents[1]


def fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "supermimic-independent-reproduction/1.0"})
            with urlopen(request, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def verify(data: bytes, entry: dict) -> dict:
    if len(data) != entry["size"]:
        raise ValueError(f"size mismatch: {entry['path']}")
    sha256 = hashlib.sha256(data).hexdigest()
    if entry.get("lfs"):
        expected = entry["lfs"]["oid"]
        if sha256 != expected:
            raise ValueError(f"LFS SHA256 mismatch: {entry['path']}")
    else:
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != entry["oid"]:
            raise ValueError(f"Git blob SHA1 mismatch: {entry['path']}")
    return {"path": entry["path"], "url": f"{BASE}/{entry['path']}",
            "bytes": len(data), "sha256": sha256, "git_blob_oid": entry["oid"],
            "input_frames": len(data.splitlines()), "input_fps": 30.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data/raw/lafan1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data/npz/lafan1")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/lafan1_provenance.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Reuse cached source metadata and CSVs")
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    tree_path = args.raw_dir / "tree.pinned.json"
    readme_path = args.raw_dir / "README.source.md"
    if not args.offline:
        tree_path.write_bytes(fetch(TREE))
        readme_path.write_bytes(fetch(f"{BASE}/README.md"))
    entries = json.loads(tree_path.read_text())
    entries = sorted((e for e in entries if e["type"] == "file" and e["path"].endswith(".csv")),
                     key=lambda e: e["path"])
    if len(entries) != 40:
        raise ValueError(f"Expected exactly 40 G1 CSVs at the pinned revision, got {len(entries)}")

    def download(entry: dict) -> dict:
        path = args.raw_dir / entry["path"]
        if path.exists():
            data = path.read_bytes()
        elif args.offline:
            raise FileNotFoundError(path)
        else:
            data = fetch(f"{BASE}/{entry['path']}")
            verify(data, entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".csv.tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        row = verify(data, entry)
        print(f"verified {entry['path']}: {row['input_frames']} frames", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(download, entries))
    manifest = {
        "dataset_repo": REPO, "dataset_revision": REVISION,
        "dataset_page": f"https://huggingface.co/datasets/{REPO}",
        "metadata_url": TREE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_license": "CC BY-NC-ND 4.0 (LAFAN1, per retarget dataset README)",
        "source_license_url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
        "source_readme_sha256": hashlib.sha256(readme_path.read_bytes()).hexdigest(),
        "input_fps": 30.0, "root_quaternion_order": "xyzw",
        "clips": rows, "total_input_frames": sum(r["input_frames"] for r in rows),
        "total_input_duration_hours": sum((r["input_frames"] - 1) / 30 for r in rows) / 3600,
        "total_download_bytes": sum(r["bytes"] for r in rows),
        "conversion": None,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.download_only:
        print(f"Downloaded and verified {len(rows)} clips; manifest: {args.manifest}")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, str(PROJECT / "scripts/data/csv_to_npz_batch.py"),
               "--input-dir", str(args.raw_dir / "g1"), "--output-dir", str(args.output_dir),
               "--input-fps", "30", "--output-fps", "50", "--workers", str(args.workers)]
    subprocess.run(command, check=True, env=env)

    import numpy as np
    outputs = []
    for row in rows:
        path = args.output_dir / (Path(row["path"]).stem + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            frames = len(archive["joint_pos"])
            fps = float(archive["fps"][0])
            if fps != 50 or archive["joint_pos"].shape != (frames, 29):
                raise ValueError(f"unexpected shape/fps: {path}")
            for key in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
                if not np.isfinite(archive[key]).all():
                    raise ValueError(f"nonfinite {key}: {path}")
        outputs.append({"name": path.name, "frames": frames, "fps": fps,
                        "duration_s": (frames - 1) / fps, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest["conversion"] = {"command": command, "output_fps": 50.0,
        "all_numeric_arrays_finite": True, "total_output_frames": sum(o["frames"] for o in outputs),
        "total_output_duration_hours": sum(o["duration_s"] for o in outputs) / 3600, "clips": outputs}
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest["conversion"].items() if k != "clips"}, indent=2))


if __name__ == "__main__":
    main()
