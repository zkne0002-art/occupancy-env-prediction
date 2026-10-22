"""Inferring which occupancy space an environmental sensor belongs to.

The MSI sensor-to-space mapping is unreliable (reported as a known, unfixable cloud-sync
issue), so the mapping is treated as part of the problem and estimated from the data.

Evidence used (all on residuals after removing the mean hour-of-week profile, because any two
spaces used on weekdays correlate raw simply through the working week):

* :func:`screen_pairs`   quick correlation table for every sensor/space pair.
* :func:`diagnose_sensor` fair common window, lag scan, week-shift placebo test, stability
  across window halves, and similarity between spaces.
"""
import warnings

import numpy as np
import pandas as pd

from .config import BIN, TZ

VARS = ["co2_ppm", "temp_c", "humidity_pct", "pm2_5"]
MIN_OVERLAP = 500
MIN_PAIRS = 500
WEEK = 7 * 24 * 6                                    # 10-minute bins per week
LAGS = range(-18, 37)                                # -3 h .. +6 h
PLACEBO_WEEKS = [w for w in range(-6, 7) if w != 0]


def _require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing column(s) {missing}")


def deseason(df):
    """Subtract the mean hour-of-week profile (local time) from every column.

    Args:
        df: DataFrame with a tz-aware DatetimeIndex.

    Returns:
        DataFrame of residuals, same shape. A series that is purely periodic by hour of week
        becomes zero.
    """
    local = df.index.tz_convert(TZ)
    key = np.asarray(local.dayofweek * 24 + local.hour)
    return df - df.groupby(key).transform("mean")


def corr_if_enough(a, b, min_pairs=MIN_PAIRS):
    """Pearson correlation using only rows where both are valid; NaN if fewer than ``min_pairs``."""
    ok = a.notna() & b.notna()
    if ok.sum() < min_pairs:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return a[ok].corr(b[ok])


def screen_pairs(env, occ, variables=VARS, min_overlap=MIN_OVERLAP):
    """Correlate every environmental variable with headcount for every sensor/space pair.

    Args:
        env: Cleaned env table (``sensorid``, ``time_utc``, variables).
        occ: Cleaned occupancy table (``floorspaceid``, ``time_utc``, ``mean_headcount``).
        variables: Env columns to correlate.
        min_overlap: Pairs with fewer overlapping valid CO2/headcount bins are skipped.

    Returns:
        DataFrame, one row per pair, with raw and ``_resid`` correlations, sorted per sensor by
        absolute residual CO2 correlation. Empty if no pair overlaps enough.
    """
    _require(env, ["sensorid", "time_utc"] + list(variables), "env")
    _require(occ, ["floorspaceid", "time_utc", "mean_headcount"], "occ")
    rows = []
    for sid, e in env.groupby("sensorid"):
        e = e.set_index("time_utc")
        for fs, o in occ.groupby("floorspaceid"):
            m = e[list(variables)].join(o.set_index("time_utc")[["mean_headcount"]], how="inner")
            ok = m.dropna(subset=["co2_ppm", "mean_headcount"])
            if len(ok) < min_overlap:
                continue
            d = deseason(m)
            row = {"sensor": sid[-6:], "space": fs[:8], "bins": len(ok),
                   "from": ok.index.min().date(), "to": ok.index.max().date()}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for v in variables:
                    row[f"{v}_raw"] = m[v].corr(m["mean_headcount"])
                    row[f"{v}_resid"] = d[v].corr(d["mean_headcount"])
            rows.append(row)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["_abs"] = res["co2_ppm_resid"].abs()
    return res.sort_values(["sensor", "_abs"], ascending=[True, False]).drop(columns="_abs")


def diagnose_sensor(env_s, occ_w, min_pairs=MIN_PAIRS):
    """Run the lag, placebo, stability and similarity diagnostics for one sensor.

    All spaces are compared over the SAME window (intersection of every space's coverage and
    the sensor's CO2 coverage) so their correlations are directly comparable.

    Args:
        env_s: DataFrame indexed by ``time_utc`` with a ``co2_ppm`` column.
        occ_w: Wide DataFrame (index ``time_utc``, one column per space) of mean headcount.
        min_pairs: Minimum valid pairs for any correlation to be reported.

    Returns:
        ``None`` if there is no common window. Otherwise a dict with ``start``, ``end``,
        ``days``, ``bins``, ``table`` (per space: ``resid`` correlation at lag 0, best lag in
        minutes, placebo mean/sd, ``z``, and correlation in each ``half1``/``half2``) and
        ``space_corr`` (residual headcount correlation between spaces).
    """
    if occ_w.shape[1] == 0:
        return None
    starts = [occ_w[c].first_valid_index() for c in occ_w]
    ends = [occ_w[c].last_valid_index() for c in occ_w]
    first, last = env_s["co2_ppm"].first_valid_index(), env_s["co2_ppm"].last_valid_index()
    if first is None or any(x is None for x in starts + ends):
        return None
    s0, s1 = max(starts + [first]), min(ends + [last])
    if s0 >= s1:
        return None
    grid = pd.date_range(s0, s1, freq=BIN)
    both = pd.concat([env_s["co2_ppm"].reindex(grid).rename("co2"), occ_w.reindex(grid)], axis=1)
    res = deseason(both)
    half = grid[len(grid) // 2]
    rows = []
    for sp in occ_w.columns:
        c, h = res["co2"], res[sp]
        lag_corr = {k: corr_if_enough(c, h.shift(k), min_pairs) for k in LAGS}
        best_k = max(lag_corr, key=lambda k: -np.inf if np.isnan(lag_corr[k]) else abs(lag_corr[k]))
        plc = np.array([corr_if_enough(c, h.shift(w * WEEK), min_pairs) for w in PLACEBO_WEEKS])
        plc = plc[~np.isnan(plc)]
        r0 = lag_corr[0]
        rows.append({
            "space": sp, "bins": int((both["co2"].notna() & both[sp].notna()).sum()),
            "raw": corr_if_enough(both["co2"], both[sp], min_pairs), "resid": r0,
            "best_lag_min": best_k * 10, "resid_at_best": lag_corr[best_k],
            "placebo_mean": plc.mean() if len(plc) else np.nan,
            "placebo_sd": plc.std() if len(plc) else np.nan,
            "z": (r0 - plc.mean()) / plc.std() if len(plc) > 2 and plc.std() > 0 else np.nan,
            "half1": corr_if_enough(c[:half], h[:half], min_pairs),
            "half2": corr_if_enough(c[half:], h[half:], min_pairs),
        })
    table = pd.DataFrame(rows).sort_values("resid", key=abs, ascending=False, ignore_index=True)
    return {"start": s0, "end": s1, "days": (s1 - s0).days, "bins": len(grid), "table": table,
            "space_corr": res[list(occ_w.columns)].corr()}


def format_diagnosis(sid, res):
    """Format the result of :func:`diagnose_sensor` as text (``res`` may be ``None``)."""
    if res is None:
        return f"SENSOR {sid[-6:]}: no common window across all spaces.\n"
    fmt = lambda x: f"{x:.2f}"
    return (f"SENSOR {sid[-6:]}  common window {res['start'].date()} -> {res['end'].date()} "
            f"({res['days']} days, {res['bins']} bins)\n"
            f"{res['table'].to_string(index=False, float_format=fmt)}\n\n"
            f"Correlation between spaces (residual headcount):\n"
            f"{res['space_corr'].to_string(float_format=fmt)}\n")


def diagnose_all(env, occ, end=None, exclude=(), min_pairs=MIN_PAIRS):
    """Run :func:`diagnose_sensor` for every sensor and return the combined text.

    Args:
        env: Cleaned env table.
        occ: Cleaned occupancy table.
        end: Optional UTC date/time; only data BEFORE it is used (so the mapping choice
            cannot see a later test period).
        exclude: Space id prefixes (8 characters) to leave out.
        min_pairs: Minimum valid pairs for a correlation.

    Returns:
        Multi-line text report.
    """
    _require(env, ["sensorid", "time_utc", "co2_ppm"], "env")
    _require(occ, ["floorspaceid", "time_utc", "mean_headcount"], "occ")
    occ = occ.assign(floorspaceid=occ["floorspaceid"].str[:8])
    occ = occ[~occ["floorspaceid"].isin(list(exclude))]
    if end is not None:
        cut = pd.Timestamp(end)
        cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
        env, occ = env[env["time_utc"] < cut], occ[occ["time_utc"] < cut]
    if env.empty or occ.empty:
        return "No data left after applying the filters."
    wide = occ.pivot(index="time_utc", columns="floorspaceid", values="mean_headcount")
    parts = []
    for sid, e in env.groupby("sensorid"):
        e = e.set_index("time_utc")
        lo, hi = e["co2_ppm"].first_valid_index(), e["co2_ppm"].last_valid_index()
        if lo is None:
            continue
        w = wide.loc[lo:hi].dropna(axis=1, thresh=min_pairs)
        parts.append(format_diagnosis(sid, diagnose_sensor(e, w, min_pairs)))
    return "\n".join(parts) if parts else "No sensor has usable data."
