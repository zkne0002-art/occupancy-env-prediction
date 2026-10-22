"""Synthetic raw data in the same format as the MSI extracts (used only by the tests).

Two spaces ('a' and 'b') with weekday working-hours headcount; env sensor ...326 follows
space 'a' (CO2 = 450 + 25 * headcount + noise), has 10 % missing readings, repeated rows and
formaldehyde errors; sensor ...777 is a dead sensor re-sending one stale reading 1500 times.
"""
import json

import numpy as np
import pandas as pd

SPACE_A = "aaaaaaaa-0000-0000-0000-000000000001"
SPACE_B = "bbbbbbbb-0000-0000-0000-000000000002"
GOOD_SENSOR = "6012002000326"
DEAD_SENSOR = "6012002000777"


def make_env_json(t, co2, hcho_error=False):
    """One env ``jsondata`` string in the MSI format (11 variables, one timestamp)."""
    def var(i, name, unit, val, err=None):
        v = {"time": int(t)}
        if val is not None:
            v["value"] = val
        if err:
            v["errorCode"] = err
        return {"values": [v], "variable": {"id": i, "name": name, "unit": unit}}
    return json.dumps([
        var(66, "Battery", "V", 4.1), var(67, "Carbon dioxide", "ppm", co2),
        var(72, "Humidity", "HR%", 55),
        var(70, "Formaldehyde", "µg/m3", None if hcho_error else 5, 8 if hcho_error else None),
        var(80, "Pressure", "mb", 1010), var(84, "Temperature", "°C", 23),
        var(99, "Light Volatile Organic Compounds", "ppb", 7),
        var(47, "Particulate matter 1", "µg/m3", 3), var(48, "Particulate matter 4", "µg/m3", 3),
        var(49, "Particulate matter 2.5", "µg/m3", 3), var(50, "Particulate matter 10", "µg/m3", 3),
    ])


def write_occ_csv(path, events):
    """Write an occupancy CSV from ``(space, headcount, 'YYYY-mm-dd HH:MM:SS+hh')`` tuples."""
    df = pd.DataFrame(events, columns=["floorspaceid", "headcount", "collecteddate"])
    df.insert(0, "deviceid", "dev")
    df.insert(0, "id", range(len(df)))
    df["occupancystatus"] = "CurrentlyOccupied"
    df["occupancystatuschangedate"] = df["collecteddate"]
    df["previousoccupancystatus"] = "NotOccupied"
    df.to_csv(path, index=False)


def write_env_csv(path, rows):
    """Write an env CSV from ``(sensorid, jsondata)`` tuples."""
    pd.DataFrame({"id": range(len(rows)), "sensorid": [r[0] for r in rows], "devicetype": "env",
                  "jsondata": [r[1] for r in rows], "status": "Processed",
                  "processing_errors": None, "createdate": "2026-01-01 00:00:00+11",
                  "processdate": "2026-01-01 00:00:00+11"}).to_csv(path, index=False)


def _headcount(idx, rng, scale, shift):
    hour = np.asarray(idx.hour) + shift
    working = (np.asarray(idx.dayofweek) < 5) & (hour >= 9) & (hour < 17)
    return np.clip(np.where(working, scale, 0) + rng.integers(-1, 2, len(idx)) * working, 0, None)


def make_raw_files(directory, start="2025-09-01", end="2025-09-22", seed=0):
    """Write ``occ.csv`` and ``env.csv`` into ``directory``; return their paths as strings."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1min", tz="Australia/Melbourne")
    hc_a, hc_b = _headcount(idx, rng, 10, 0), _headcount(idx, rng, 6, 1)
    events = []
    for sid, hc in ((SPACE_A, hc_a), (SPACE_B, hc_b)):
        s = pd.Series(hc, index=idx)
        for t, v in s[s.diff().fillna(1) != 0].items():
            events.append((sid, int(v), t.strftime("%Y-%m-%d %H:%M:%S%z")[:-2]))
    events += events[:5]                                  # exact duplicate events
    write_occ_csv(f"{directory}/occ.csv", events)

    tt = pd.date_range(start, end, freq="10min", tz="UTC")
    h_a = pd.Series(hc_a, index=idx).reindex(tt, method="nearest").values
    rows = []
    for i, (t, h) in enumerate(zip(tt, h_a)):
        if rng.random() < 0.1:
            continue                                      # missing reading
        js = make_env_json(t.timestamp(), round(450 + 25 * h + rng.normal(0, 10)),
                           hcho_error=(i % 3 == 0))
        rows += [(GOOD_SENSOR, js)] * (3 if i % 4 == 0 else 1)   # re-sent duplicates
    rows += [(DEAD_SENSOR, make_env_json(1683326565, 425))] * 1500
    write_env_csv(f"{directory}/env.csv", rows)
    return {"occ": f"{directory}/occ.csv", "env": f"{directory}/env.csv"}
