"""Tests for occenv.performance: baseline and improved code must agree; cost model is sane."""
import pytest

from occenv.cli import main
from occenv.performance import (count_duplicates_baseline, count_duplicates_fast,
                                estimate_costs, measure, profile_text)


def test_fast_and_baseline_duplicate_counts_agree(raw):
    base = count_duplicates_baseline(raw["occ"])
    fast = count_duplicates_fast(raw["occ"])
    assert base == fast
    assert base[1] == 5                      # the generator inserts exactly 5 repeated events


def test_duplicate_counts_agree_across_chunk_boundaries(raw):
    # tiny chunks force duplicates to sit in different chunks
    assert count_duplicates_baseline(raw["occ"], chunksize=7) == \
           count_duplicates_fast(raw["occ"], chunksize=7) == count_duplicates_baseline(raw["occ"])


def test_nrows_limits_rows_read(raw):
    assert count_duplicates_fast(raw["occ"], nrows=100)[0] == 100
    assert count_duplicates_baseline(raw["occ"], nrows=100)[0] == 100


def test_measure_returns_result_time_and_memory():
    result, secs, peak_mb = measure(lambda n: [0] * n, 100_000)
    assert len(result) == 100_000 and secs >= 0 and peak_mb > 0.1


def test_profile_text_lists_functions():
    assert "function calls" in profile_text(sum, range(1000), top=3)


def test_cost_model_is_consistent():
    c = estimate_costs(8542, 3715).set_index("stage")
    # logistic regression predict: (2*nnz + 20) ops per row over nnz*8 bytes, nnz = 1 + 8
    assert c.loc["logreg predict", "ops_per_byte"] == pytest.approx((2 * 9 + 20) / (9 * 8))
    assert (c["ops_per_byte"] < 5).all()                      # nothing here is compute-bound
    double = estimate_costs(2 * 8542, 3715).set_index("stage")
    assert double.loc["logreg fit", "ops"] == pytest.approx(2 * c.loc["logreg fit", "ops"])
    ratio = double.loc["hgb fit", "ops"] / c.loc["hgb fit", "ops"]
    assert 1.5 < ratio < 2.0                                  # grows with data, but split search is fixed


def test_cli_benchmark_runs_and_saves(raw, tmp_path):
    out = tmp_path / "perf.txt"
    assert main(["benchmark", "--occ", raw["occ"], "--env", raw["env"], "--nrows", "2000",
                 "--out", str(out)]) == 0
    text = out.read_text()
    assert "identical duplicate counts confirmed" in text and "Analytic cost estimates" in text
