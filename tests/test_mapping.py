"""Tests for occenv.mapping: de-seasoning, correlation guard and sensor/space diagnostics."""
import numpy as np
import pandas as pd
import pytest

from occenv.config import TZ
from occenv.mapping import (corr_if_enough, deseason, diagnose_all, diagnose_sensor,
                            screen_pairs)


def wide(occ):
    occ = occ.assign(sp=occ["floorspaceid"].str[:8])
    return occ.pivot(index="time_utc", columns="sp", values="mean_headcount")


def test_deseason_removes_a_purely_weekly_pattern():
    idx = pd.date_range("2025-09-01", periods=7 * 24 * 6 * 3, freq="10min", tz="UTC")
    local = idx.tz_convert(TZ)
    df = pd.DataFrame({"x": np.asarray(local.dayofweek * 24 + local.hour, dtype=float)}, index=idx)
    assert deseason(df)["x"].abs().max() < 1e-9


def test_corr_if_enough_guards_small_samples():
    a = pd.Series(np.arange(10.0))
    assert np.isnan(corr_if_enough(a, a, min_pairs=500))
    b = pd.Series(np.arange(600.0))
    assert corr_if_enough(b, b * 2, min_pairs=500) == pytest.approx(1.0)


def test_diagnose_identifies_the_true_space(processed):
    env, occ = processed
    res = diagnose_sensor(env.set_index("time_utc"), wide(occ), min_pairs=500)
    table = res["table"].set_index("space")
    assert res["table"].iloc[0]["space"] == "aaaaaaaa"
    assert table.loc["aaaaaaaa", "resid"] > 0.15
    assert abs(table.loc["bbbbbbbb", "resid"]) < 0.1
    # The placebo test has little power on a short (18-day) window, so compare RELATIVELY
    # (true space vs decoy, and vs its own placebo mean) rather than demanding an absolute z.
    assert table.loc["aaaaaaaa", "z"] > table.loc["bbbbbbbb", "z"]
    assert table.loc["aaaaaaaa", "resid"] > table.loc["aaaaaaaa", "placebo_mean"] + 0.1
    assert table.loc["aaaaaaaa", "half1"] > 0.1 and table.loc["aaaaaaaa", "half2"] > 0.1
    assert abs(table.loc["aaaaaaaa", "best_lag_min"]) <= 60     # physically plausible lag


def test_diagnose_sensor_returns_none_without_overlap(processed):
    env, occ = processed
    w = wide(occ)
    w.index = w.index + pd.Timedelta(days=400)                  # occupancy far in the future
    assert diagnose_sensor(env.set_index("time_utc"), w) is None


def test_screen_pairs_ranks_true_space_first(processed):
    env, occ = processed
    res = screen_pairs(env, occ)
    assert len(res) == 2
    assert res.iloc[0]["space"] == "aaaaaaaa"
    assert {"co2_ppm_raw", "co2_ppm_resid", "temp_c_resid"} <= set(res.columns)


def test_screen_pairs_rejects_missing_columns(processed):
    env, occ = processed
    with pytest.raises(ValueError, match="missing column"):
        screen_pairs(env.drop(columns="co2_ppm"), occ)


def test_diagnose_all_exclude_and_end(processed):
    env, occ = processed
    text = diagnose_all(env, occ, exclude=["bbbbbbbb"])
    assert "aaaaaaaa" in text and "bbbbbbbb" not in text
    early = diagnose_all(env, occ, end="2025-09-10")
    assert "2025-09-09" in early                                # window stops before the cutoff
    assert "No data" in diagnose_all(env, occ, end="2000-01-01")
