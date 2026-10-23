"""Cleaning and regularisation of the raw environmental and occupancy extracts.

Both sources are put on the same regular 10-minute UTC grid so they can be joined.

**Environmental data** (`clean_env`)

- The timestamp inside the JSON (Unix seconds, UTC) is the true measurement time;
  `createdate` is only when the record was ingested.
- Exact repeats of (sensorid, measurement time) are removed. In the MSI extract about
  three quarters of rows are repeats of stale readings.
- A value carrying an `errorCode` or lacking a `value` becomes NaN.
- Isolated readings (no neighbour within `isolation_gap_s`) are treated as stale re-sends
  and dropped; sensors with fewer than `min_readings` readings left are dropped.
- Readings are averaged into 10-minute bins; missing bins are kept as NaN.

**Occupancy data** (`clean_occupancy`)

- The log is a step function: headcount is held from one event to the next, sampled at
  1-minute resolution (state at the end of each minute), then aggregated to 10 minutes.
- A non-zero headcount not refreshed for `max_stale_min` minutes, or any state not
  refreshed for `max_gap_h` hours, is treated as missing (possible sensor outage).
  A zero headcount may persist (an empty room produces no events).
"""
import json
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd

from .config import BIN, VAR_NAMES

logger = logging.getLogger(__name__)

MIN_READINGS = 1000      # minimum usable readings for a sensor to be kept
ISOLATION_GAP_S = 3600   # a reading with no neighbour within this many seconds is "isolated"
MAX_STALE_MIN = 120      # non-zero headcount older than this (minutes) becomes missing
MAX_GAP_H = 96           # any state older than this (hours) becomes missing
MIN_VALID_FRAC = 0.5     # a 10-minute bin needs at least this fraction of valid minutes
TIME_RE = re.compile(r'"time":\s*(\d+)')   # first measurement time in a jsondata string


def _require_columns(path, required):
    """Raise ``ValueError`` if the CSV at ``path`` lacks any of the ``required`` columns."""
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}; found {header}")


def parse_env_json(js):
    """Parse the ``jsondata`` string of one environmental reading.

    Args:
        js: JSON text: a list of ``{"values": [{"time", "value", ["errorCode"]}],
            "variable": {"name", "unit", ...}}`` objects.

    Returns:
        ``None`` if the text cannot be parsed or has no timestamp. Otherwise a dict with
        ``time`` (Unix seconds), ``values`` ({short column name: float, NaN if invalid}),
        ``invalid`` (columns set to NaN because of an error code or missing value) and
        ``time_mismatch`` (True if any variable's time differs from the row time).
    """
    try:
        payload = json.loads(js)
        t = int(payload[0]["values"][0]["time"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    values, invalid, mismatch = {}, [], False
    for var in payload:
        try:
            col = VAR_NAMES.get(var["variable"]["name"])
            v = (var.get("values") or [{}])[0]
        except (KeyError, TypeError, AttributeError):
            continue
        if col is None:
            continue
        if int(v.get("time", t)) != t:
            mismatch = True
        if "errorCode" in v or v.get("value") is None:
            values[col] = np.nan
            invalid.append(col)
        else:
            values[col] = float(v["value"])
    return {"time": t, "values": values, "invalid": invalid, "time_mismatch": mismatch}


def parse_timestamps(s):
    """Parse timestamps such as ``2024-01-01 21:00:00+11`` into tz-aware UTC values.

    Pandas cannot use its fast ISO path for a two-digit offset (``+11``) and falls back to a
    slow per-row parser (about 76 % of the cleaning time on the 9-million-row occupancy file).
    The fast path here handles the common fixed-width form: it parses the first 19 characters
    with an explicit format and applies the whole-hour offset. Anything else (missing values,
    other widths, fractional seconds, ``+05:30`` offsets, bad text) uses the slow general parser,
    so results are identical in every case.

    Args:
        s: Series of timestamp strings.

    Returns:
        Series of ``datetime64[..., UTC]`` with the same index; NaT where unparseable.
    """
    s = pd.Series(s)
    if len(s) and s.notna().all():
        txt = s.astype(str)
        if (txt.str.len() == 22).all() and txt.str[-3].isin(["+", "-"]).all():
            try:
                naive = pd.to_datetime(txt.str[:19], format="%Y-%m-%d %H:%M:%S")
                tail = txt.str[-3:]
                offset = tail.map({u: int(u) for u in tail.unique()})
                return (naive - pd.to_timedelta(offset, unit="h")).dt.tz_localize("UTC")
            except ValueError:
                pass
    return pd.to_datetime(s, utc=True, errors="coerce")


def clean_env(path, min_readings=MIN_READINGS, isolation_gap_s=ISOLATION_GAP_S, chunksize=100_000,
              dedupe_first=True):
    """Clean the environmental CSV and put it on a regular 10-minute UTC grid.

    Args:
        path: CSV with at least the columns ``sensorid`` and ``jsondata``.
        min_readings: Sensors with fewer usable readings after cleaning are dropped.
        isolation_gap_s: Readings with no neighbour within this many seconds are dropped.
        chunksize: Rows read per chunk (keeps memory low for large files).
        dedupe_first: If True (default) repeats are detected from a cheap regex on the
            timestamp BEFORE the JSON is parsed, so discarded rows are never parsed (about
            76 % of rows in the MSI extract). False parses every row first (the original,
            slower order); both give identical output.

    Returns:
        ``(table, stats)``. ``table`` has columns ``sensorid``, ``time_utc`` (tz-aware UTC,
        regular 10-minute steps per sensor) and one column per variable in
        `occenv.config.VAR_NAMES`; empty bins are NaN. ``stats`` is a dict of counts
        describing what was removed, plus ``sensors_kept`` and ``sensors_dropped``.

    Raises:
        ValueError: If required columns are missing, nothing can be parsed, or no sensor
            has enough usable readings.
    """
    _require_columns(path, ["sensorid", "jsondata"])
    stats, seen, rows = Counter(), set(), []
    for ch in pd.read_csv(path, chunksize=chunksize, dtype={"sensorid": str},
                          usecols=["sensorid", "jsondata"]):
        for sid, js in zip(ch["sensorid"], ch["jsondata"]):
            stats["raw rows"] += 1
            if dedupe_first:
                m = TIME_RE.search(js) if isinstance(js, str) else None
                if m is not None and (sid, int(m.group(1))) in seen:
                    stats["duplicate rows removed"] += 1
                    continue
            parsed = parse_env_json(js)
            if parsed is None:
                stats["unparseable rows"] += 1
                continue
            key = (sid, parsed["time"])
            if key in seen:
                stats["duplicate rows removed"] += 1
                continue
            seen.add(key)
            if parsed["time_mismatch"]:
                stats["variable time != row time"] += 1
            for col in parsed["invalid"]:
                stats[f"invalid value -> NaN: {col}"] += 1
            rows.append({"sensorid": sid, "time": parsed["time"], **parsed["values"]})
    if not rows:
        raise ValueError(f"{path}: no parseable environmental readings found")

    df = pd.DataFrame(rows).sort_values(["sensorid", "time"])
    for col in VAR_NAMES.values():
        if col not in df.columns:
            df[col] = np.nan
    stats["unique readings"] = len(df)
    sensors_before = set(df["sensorid"])
    unique_per_sensor = df.groupby("sensorid").size().to_dict()

    g = df.groupby("sensorid")["time"]
    keep = (g.diff() <= isolation_gap_s) | (-g.diff(-1) <= isolation_gap_s)
    stats["isolated/stale readings removed"] = int((~keep).sum())
    df = df[keep]
    counts = df["sensorid"].value_counts()
    df = df[df["sensorid"].isin(counts[counts >= min_readings].index)]
    if df.empty:
        raise ValueError(f"{path}: no sensor has at least {min_readings} usable readings")
    out_stats = dict(stats)
    out_stats["unique_per_sensor"] = unique_per_sensor
    out_stats["sensors_kept"] = sorted(set(df["sensorid"]))
    out_stats["sensors_dropped"] = sorted(sensors_before - set(df["sensorid"]))
    for k, v in out_stats.items():
        logger.info("  %s: %s", k, v)

    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.floor(BIN)
    cols = list(VAR_NAMES.values())
    binned = df.groupby(["sensorid", "time_utc"])[cols].mean()
    parts = []
    for sid, b in binned.groupby(level=0):
        b = b.droplevel(0)
        full = b.reindex(pd.date_range(b.index.min(), b.index.max(), freq=BIN, name="time_utc"))
        full.insert(0, "sensorid", sid)
        parts.append(full)
        logger.info("  %s: %d bins, %.1f%% empty, %s -> %s", sid, len(full),
                    100 * full["co2_ppm"].isna().mean(), full.index.min(), full.index.max())
    return pd.concat(parts).reset_index(), out_stats


def clean_occupancy(path, max_stale_min=MAX_STALE_MIN, max_gap_h=MAX_GAP_H,
                    min_valid_frac=MIN_VALID_FRAC):
    """Convert the occupancy event log to time-weighted 10-minute headcount per space.

    Args:
        path: CSV with at least ``floorspaceid``, ``headcount`` and ``collecteddate``
            (timestamps with a UTC offset, e.g. ``2024-01-01 21:00:00+11``).
        max_stale_min: Non-zero headcount older than this many minutes becomes missing.
        max_gap_h: Any state older than this many hours becomes missing.
        min_valid_frac: A 10-minute bin with a smaller fraction of valid minutes is NaN.

    Returns:
        DataFrame with columns ``floorspaceid``, ``time_utc``, ``mean_headcount``,
        ``max_headcount``, ``frac_occupied`` (share of valid minutes with headcount > 0),
        ``valid_frac`` and ``n_events``. Events sharing a timestamp keep the last one.

    Raises:
        ValueError: If required columns are missing or no valid event remains.
    """
    _require_columns(path, ["floorspaceid", "headcount", "collecteddate"])
    df = pd.read_csv(path, usecols=["floorspaceid", "headcount", "collecteddate"],
                     dtype={"floorspaceid": "category", "headcount": "float32"})
    df["t"] = parse_timestamps(df["collecteddate"])
    df = df.dropna(subset=["t", "headcount"]).drop(columns="collecteddate")
    if df.empty:
        raise ValueError(f"{path}: no valid occupancy events")
    logger.info("Occupancy events used: %d", len(df))
    parts = []
    for sid, g in df.groupby("floorspaceid", observed=True):
        s = (g.sort_values("t", kind="stable").drop_duplicates("t", keep="last")
             .set_index("t")["headcount"])
        minute = s.resample("1min").last()                    # state at the end of each minute
        held = minute.ffill()
        last_event = pd.Series(minute.index.where(minute.notna()), index=minute.index).ffill()
        now = pd.Series(minute.index, index=minute.index)
        age_min = (now - last_event).dt.total_seconds() / 60
        valid = ((held == 0) & (age_min <= max_gap_h * 60)) | \
                ((held > 0) & (age_min <= max_stale_min))
        hc = held.where(valid)
        occ = (hc > 0).astype(float).where(hc.notna())
        out = pd.DataFrame({
            "mean_headcount": hc.resample(BIN).mean(),
            "max_headcount": hc.resample(BIN).max(),
            "frac_occupied": occ.resample(BIN).mean(),
            "valid_frac": hc.notna().resample(BIN).mean(),
            "n_events": s.resample(BIN).count(),
        })
        bad = out["valid_frac"] < min_valid_frac
        out.loc[bad, ["mean_headcount", "max_headcount", "frac_occupied"]] = np.nan
        out.index.name = "time_utc"
        out.insert(0, "floorspaceid", str(sid))
        parts.append(out.reset_index())
        v = out["mean_headcount"]
        logger.info("  %s: %d bins, %.1f%% missing, occupied %.1f%% of valid time, headcount "
                    "median=%.1f p95=%.1f max=%.0f", str(sid)[:8], len(out), 100 * v.isna().mean(),
                    100 * out["frac_occupied"].mean(), v.median(), v.quantile(.95),
                    out["max_headcount"].max())
    return pd.concat(parts, ignore_index=True)
