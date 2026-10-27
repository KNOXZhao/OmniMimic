"""Download the licensed G1 archive using already-approved Hugging Face access.

The token is read from HF_TOKEN or a hidden prompt; never saved. No license is
accepted by this script. A gated-access error requires approval on Hugging Face.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token (hidden): ")
    repo = "bones-studio/seed"
    api = HfApi(token=token)
    try:
        info = api.dataset_info(repo, revision=args.revision, files_metadata=True)
        revision = info.sha
        names = [f.rfilename for f in info.siblings]
        selected = [n for n in names if n == "g1.tar.gz" or n.endswith(".parquet")]
        if "g1.tar.gz" not in selected:
            raise RuntimeError("G1 archive not found at the selected revision")
        args.destination.mkdir(parents=True, exist_ok=True)
        records = []
        print(f"Dataset: {repo}; pinned revision: {revision}", flush=True)
        for name in selected:
            print(f"Downloading {name}", flush=True)
            path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision,
                                      token=token, local_dir=args.destination))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            records.append({"file": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
            print(f"Verified {name}: {path.stat().st_size} bytes", flush=True)
        (args.destination / "download_manifest.json").write_text(json.dumps({
            "repo": repo, "revision": revision, "files": records,
            "access": "user-provided token; previously accepted dataset terms",
        }, indent=2) + "\n")
    except Exception as exc:
        # Do not print request headers, credential-bearing URLs or exception reprs.
        print(f"Download failed: {type(exc).__name__}. Check access and connectivity.", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
