"""Validate compact evaluation selection without loading any motion tensors."""

from pathlib import Path

import pytest

from supermimic.eval.inputs import benchmark_env_count, benchmark_source


def test_packed_library_does_not_limit_parallelism_to_one():
    assert benchmark_source(library=Path("motions.pt")).compact_library
    assert benchmark_env_count(1024, input_count=1, packed=True) == 1024
    assert benchmark_env_count(1024, input_count=40, packed=False) == 40
    assert benchmark_env_count(3, input_count=40, packed=False) == 3


@pytest.mark.parametrize("selection", [
    {"npz_dir": Path("motions")}, {"clips": ["walk"]},
    {"clip_list": Path("clips.txt")}, {"manifest": Path("manifest.csv")},
])
def test_packed_library_filters_must_be_resolved_when_building(selection):
    with pytest.raises(ValueError, match="exclusive"):
        benchmark_source(library=Path("motions.pt"), **selection)


def test_default_npz_source_and_invalid_capacity():
    source = benchmark_source()
    assert source.path == Path("data/npz/lafan1")
    assert not source.compact_library
    with pytest.raises(ValueError, match="positive"):
        benchmark_env_count(0, input_count=1, packed=True)
