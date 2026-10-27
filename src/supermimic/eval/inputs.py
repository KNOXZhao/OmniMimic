"""Benchmark source validation, independent of Torch/CUDA and simulator imports."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkSource:
    path: Path
    compact_library: bool


def benchmark_source(
    library: Path | None = None,
    npz_dir: Path | None = None,
    clips: list[str] | None = None,
    clip_list: Path | None = None,
    manifest: Path | None = None,
) -> BenchmarkSource:
    """A packed library contains its complete clip selection; filters apply to NPZ only."""
    if library is not None:
        if any(x is not None for x in (npz_dir, clips, clip_list, manifest)):
            raise ValueError("--library is exclusive with --npz-dir, --clips, --clip-list, and --manifest")
        if library.suffix != ".pt":
            raise ValueError("--library must be a packed .pt motion library")
        return BenchmarkSource(library, compact_library=True)
    if clips is not None and clip_list is not None:
        raise ValueError("choose --clips or --clip-list, not both")
    return BenchmarkSource(npz_dir or Path("data/npz/lafan1"), compact_library=False)


def benchmark_env_count(requested: int | None, input_count: int, packed: bool) -> int:
    """A packed input represents many clips and must not cap simulation at one env."""
    count = requested if requested is not None else 1024
    if count < 1 or input_count < 1:
        raise ValueError("benchmark needs positive num_envs and at least one input")
    return count if packed else min(count, input_count)
