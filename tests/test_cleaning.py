"""Tests for occenv.cleaning: parsing, de-duplication, validity rules and time-weighting."""
import numpy as np
import pandas as pd
import pytest
from synthetic import (DEAD_SENSOR, GOOD_SENSOR, make_env_json, write_env_csv,
                       write_occ_csv)

from occenv.cleaning import clean_env, clean_occupancy, parse_env_json


def at(table, ts):
    """Row of a cleaned occupancy table at a UTC timestamp string."""
    return table.set_index("time_utc").loc[pd.Timestamp(ts, tz="UTC")]


# ---------- environmental data ----------
def test_parse_valid_row():
    r = parse_env_json(make_env_json(1700000000, 600))
    assert r["time"] == 1700000000
    assert r["values"]["co2_ppm"] == 600.0
    assert r["values"]["temp_c"] == 23.0
    assert r["invalid"] == [] and not r["time_mismatch"]


def test_parse_error_code_without_value_becomes_nan():
    r = parse_env_json(make_env_json(1700000000, 600, hcho_error=True))
    assert np.isnan(r["values"]["hcho_ugm3"])
    assert r["invalid"] == ["hcho_ugm3"]


def test_parse_unparseable_returns_none():
    assert parse_env_json("not json") is None
    assert parse_env_json(float("nan")) is None
    assert parse_env_json("[]") is None


def test_clean_env_rejects_missing_columns(tmp_path):
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="missing required"):
        clean_env(str(tmp_path / "x.csv"))


def test_clean_env_removes_duplicates_and_dead_sensor(raw):
    env, stats = clean_env(raw["env"])
    assert stats["duplicate rows removed"] > 1500          # re-sent rows incl. dead sensor
    assert set(env["sensorid"]) == {GOOD_SENSOR}
    assert DEAD_SENSOR in stats["sensors_dropped"]
    assert stats["invalid value -> NaN: hcho_ugm3"] > 0
    assert env["hcho_ugm3"].isna().any()
    # regular 10-minute grid with NaN for empty bins
    assert env["time_utc"].diff().dropna().eq(pd.Timedelta("10min")).all()
    assert env["co2_ppm"].dropna().between(300, 1000).all()


def test_isolated_reading_is_dropped(tmp_path):
    base = 1_700_000_000
    rows = [(GOOD_SENSOR, make_env_json(base + 600 * i, 500)) for i in range(20)]
    rows.append((GOOD_SENSOR, make_env_json(base + 10 * 86400, 999)))     # stale straggler
    write_env_csv(tmp_path / "e.csv", rows)
    env, stats = clean_env(str(tmp_path / "e.csv"), min_readings=5)
    assert stats["isolated/stale readings removed"] == 1
    assert env["co2_ppm"].max() == 500


def test_clean_env_all_sensors_too_short_raises(tmp_path):
    rows = [(GOOD_SENSOR, make_env_json(1_700_000_000 + 600 * i, 500)) for i in range(10)]
    write_env_csv(tmp_path / "e.csv", rows)
    with pytest.raises(ValueError, match="usable readings"):
        clean_env(str(tmp_path / "e.csv"), min_readings=100)


# ---------- occupancy data ----------
def test_occupancy_time_weighted_mean_and_timezone(tmp_path):
    # 21:00+11 = 10:00 UTC. Headcount 2 for five minutes, then 0 for five minutes.
    write_occ_csv(tmp_path / "o.csv", [("s1", 2, "2024-01-01 21:00:00+11"),
                                       ("s1", 0, "2024-01-01 21:05:00+11"),
                                       ("s1", 0, "2024-01-01 21:10:00+11")])
    row = at(clean_occupancy(str(tmp_path / "o.csv")), "2024-01-01 10:00")
    assert row["mean_headcount"] == pytest.approx(1.0)       # (2*5 + 0*5) / 10 minutes
    assert row["max_headcount"] == 2
    assert row["frac_occupied"] == pytest.approx(0.5)
    assert row["n_events"] == 2


def test_stale_nonzero_headcount_becomes_missing(tmp_path):
    write_occ_csv(tmp_path / "o.csv", [("s1", 3, "2024-01-01 10:00:00+00"),
                                       ("s1", 0, "2024-01-01 20:00:00+00"),
                                       ("s1", 0, "2024-01-01 20:10:00+00")])
    occ = clean_occupancy(str(tmp_path / "o.csv"))
    assert at(occ, "2024-01-01 10:30")["mean_headcount"] == 3          # fresh
    assert np.isnan(at(occ, "2024-01-01 15:00")["mean_headcount"])     # 3 people, no update 5 h
    assert at(occ, "2024-01-01 20:00")["mean_headcount"] == 0          # refreshed


def test_zero_headcount_persists_but_not_beyond_max_gap(tmp_path):
    write_occ_csv(tmp_path / "o.csv", [("s1", 0, "2024-01-01 10:00:00+00"),
                                       ("s1", 1, "2024-01-10 10:00:00+00"),
                                       ("s1", 1, "2024-01-10 10:10:00+00")])
    occ = clean_occupancy(str(tmp_path / "o.csv"))
    assert at(occ, "2024-01-01 22:00")["mean_headcount"] == 0          # 12 h of silence: empty
    assert np.isnan(at(occ, "2024-01-08 12:00")["mean_headcount"])     # 7 days: possible outage


def test_duplicate_timestamps_keep_last(tmp_path):
    write_occ_csv(tmp_path / "o.csv", [("s1", 5, "2024-01-01 10:00:00+00"),
                                       ("s1", 1, "2024-01-01 10:00:00+00"),
                                       ("s1", 1, "2024-01-01 10:10:00+00")])
    occ = clean_occupancy(str(tmp_path / "o.csv"))
    assert at(occ, "2024-01-01 10:00")["mean_headcount"] == 1


def test_clean_occupancy_rejects_missing_columns(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="missing required"):
        clean_occupancy(str(tmp_path / "x.csv"))
