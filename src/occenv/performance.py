"""Timing, memory, profiling and cost-model tools for the performance analysis.

The slowest, most memory-hungry step in the project is the data audit of the 9-million-row
occupancy file, so that is where baseline and improved implementations are compared:

* BASELINE  :func:`count_duplicates_baseline`: builds one Python string per row and keeps every
  string in a ``set``. Simple, but memory grows with every unique row (a Python string object
  costs on the order of 100 bytes).
* IMPROVED  :func:`count_duplicates_fast`: hashes the key columns in vectorised form to 64-bit
  integers and counts repeats with ``numpy.unique`` (8 bytes per row, no Python-level loop).

Expected (and, on a 1M-row synthetic test, observed) trade-off: the improvement mainly saves
MEMORY (about 4x less at 1M rows, and the gap grows with n). Run time is similar for both
because reading and parsing the CSV dominates; the cProfile output in the benchmark shows this.

Both must return identical counts (checked by the tests and by the benchmark itself). Hash
collisions could in principle make the fast count slightly too high; for n = 9e6 rows and 64-bit
hashes the expected number of collisions is about n^2 / 2^65, i.e. about 2e-6.

:func:`estimate_costs` gives analytic operation/byte counts so that measured model timings can
be explained in terms of arithmetic intensity (operations per byte moved).
"""
import cProfile
import io
import os
import pstats
import tempfile
import time
import tracemalloc

import numpy as np
import pandas as pd

from . import cleaning
from .reporting import TextReport

KEY_COLS = ["floorspaceid", "collecteddate", "occupancystatus"]


def measure(fn, *args, **kwargs):
    """Run ``fn`` twice: once untraced for the time, once under ``tracemalloc`` for peak memory.

    ``tracemalloc`` slows code down, so timing and memory are taken from separate runs.

    Returns:
        ``(result, seconds, peak_megabytes)``. Peak memory counts Python and NumPy/pandas
        allocations made during the call.
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    seconds = time.perf_counter() - t0
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak / 1e6


def profile_text(fn, *args, top=12, **kwargs):
    """Run ``fn`` under ``cProfile`` and return the ``top`` functions by cumulative time as text."""
    pr = cProfile.Profile()
    pr.enable()
    fn(*args, **kwargs)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def count_duplicates_baseline(path, nrows=None, chunksize=200_000):
    """BASELINE: count repeated (space, time, status) rows with a Python ``set`` of strings.

    Memory grows linearly with the number of rows (one string object per unique key).

    Args:
        path: Occupancy CSV.
        nrows: Only read the first ``nrows`` rows (None = all).
        chunksize: Rows per chunk.

    Returns:
        ``(rows read, number of duplicate rows)``.
    """
    seen, dups, n = set(), 0, 0
    for ch in pd.read_csv(path, usecols=KEY_COLS, dtype=str, chunksize=chunksize, nrows=nrows):
        n += len(ch)
        for k in (ch["floorspaceid"] + "|" + ch["collecteddate"] + "|" + ch["occupancystatus"]):
            if k in seen:
                dups += 1
            else:
                seen.add(k)
    return n, dups


def count_duplicates_fast(path, nrows=None, chunksize=200_000):
    """IMPROVED: same count as the baseline using vectorised 64-bit row hashes.

    Args and Returns: as :func:`count_duplicates_baseline`.
    """
    hashes, n = [], 0
    dtypes = {"floorspaceid": "category", "occupancystatus": "category"}
    for ch in pd.read_csv(path, usecols=KEY_COLS, dtype=dtypes, chunksize=chunksize, nrows=nrows):
        n += len(ch)
        hashes.append(pd.util.hash_pandas_object(ch, index=False).to_numpy())
    h = np.concatenate(hashes)
    return n, int(len(h) - len(np.unique(h)))


def estimate_costs(n_train, n_test, d_env=8, lr_iters=100, n_trees=200, depth=3,
                   n_features_hgb=10, n_bins=255):
    """Analytic operation and memory-traffic estimates for the two model families.

    These are order-of-magnitude estimates from the algorithms' structure, not measurements.

    Logistic regression (L-BFGS) on one-hot hour-of-week plus ``d_env`` dense features: the
    one-hot block adds ONE non-zero per row, so ``nnz = 1 + d_env``. Each iteration does a
    forward pass (2 * nnz FLOPs/row) and a gradient (2 * nnz FLOPs/row) over float64 data.

    Gradient boosting (histogram trees): per tree and per level, every row adds its gradient
    and hessian into each feature's histogram (about 2 FLOPs per row per feature, upper bound
    since the subtraction trick is ignored); split search scans ``n_bins`` per feature per node.
    Features are 8-bit bin codes, so memory traffic is ~1 byte per feature plus 8 bytes of
    gradient/hessian per row per level. Prediction walks ``depth`` comparisons per tree per row
    (integer comparisons, counted as "ops").

    Returns:
        DataFrame with columns ``stage``, ``ops``, ``bytes`` and ``ops_per_byte`` (arithmetic
        intensity). Low values (below ~1) mean the stage is limited by memory, not arithmetic.
    """
    nnz = 1 + d_env
    nodes = 2 ** depth - 1
    rows = [
        ("logreg fit", lr_iters * n_train * 4 * nnz, lr_iters * n_train * 2 * nnz * 8),
        ("logreg predict", n_test * (2 * nnz + 20), n_test * nnz * 8),
        ("hgb fit", n_trees * (2 * n_train * n_features_hgb * depth
                               + nodes * n_features_hgb * n_bins * 10 + n_train * 10),
         n_trees * depth * n_train * (n_features_hgb + 8)),
        ("hgb predict", n_test * n_trees * depth, n_test * n_trees * depth * 8),
    ]
    out = pd.DataFrame(rows, columns=["stage", "ops", "bytes"])
    out["ops_per_byte"] = out["ops"] / out["bytes"]
    return out


def _sample_csv(path, nrows, directory):
    """Write the first ``nrows`` rows of ``path`` to a temporary CSV and return its path."""
    out = os.path.join(directory, "sample.csv")
    pd.read_csv(path, nrows=nrows, dtype=str).to_csv(out, index=False)
    return out


def run_benchmark(occ_path, env_path, nrows=1_000_000, report=None):
    """Run the whole performance study and log it to ``report``.

    1. Baseline vs improved duplicate counting at three input sizes (scalability), with an
       equality check on the results.
    2. ``cProfile`` of ``clean_occupancy`` (on the first ``nrows`` rows) and ``clean_env`` to
       show where cleaning time actually goes.
    3. Analytic cost estimates for the model families (see :func:`estimate_costs`).

    Raises:
        AssertionError: If the baseline and improved counts ever disagree.
    """
    report = report or TextReport()
    log = report.log
    log("=" * 60, "\n1. DUPLICATE COUNTING: baseline (set of strings) vs improved (vectorised hashes)")
    rows = []
    for n in sorted({max(nrows // 4, 1), max(nrows // 2, 1), nrows}):
        (n_b, d_b), t_b, m_b = measure(count_duplicates_baseline, occ_path, n)
        (n_f, d_f), t_f, m_f = measure(count_duplicates_fast, occ_path, n)
        assert (n_b, d_b) == (n_f, d_f), f"results differ: baseline {(n_b, d_b)} vs fast {(n_f, d_f)}"
        rows.append({"rows": n_b, "duplicates": d_b, "base_s": t_b, "fast_s": t_f,
                     "speedup": t_b / t_f, "base_MB": m_b, "fast_MB": m_f,
                     "mem_ratio": m_b / m_f})
    log(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    log("(identical duplicate counts confirmed at every size)")

    with tempfile.TemporaryDirectory() as tmp:
        log("\n" + "=" * 60 + f"\n2. cProfile of clean_occupancy on the first {nrows} rows")
        log(profile_text(cleaning.clean_occupancy, _sample_csv(occ_path, nrows, tmp)))
    log("=" * 60, "\n3. cProfile of clean_env (whole env file)")
    log(profile_text(cleaning.clean_env, env_path))

    log("=" * 60, "\n4. Analytic cost estimates (n_train=8542, n_test=3715 as in the thr=1 run)")
    log(estimate_costs(8542, 3715).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    return report
