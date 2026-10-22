"""Chunked audit of the raw (large) CSV files.

These functions only READ the raw data and summarise it (row counts, time ranges, duplicate
counts, error codes, per-variable statistics, mapping coverage). They are the *baseline*
implementation: the duplicate checks keep Python sets of every key, which is simple but slow
and memory hungry on the 9-million-row occupancy file.
"""
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from .reporting import TextReport

CHUNK = 200_000
TIME_RE = re.compile(r'"time":\s*(\d+)')    # first measurement time in a jsondata string
JSON_SAMPLE_EVERY = 50                       # fully parse 1 in N env rows (JSON parsing is slow)


def profile_occupancy(path, report=None):
    """Summarise the occupancy event log.

    Args:
        path: Occupancy CSV (``floorspaceid``, ``headcount``, ``collecteddate``, ...).
        report: Optional :class:`~occenv.reporting.TextReport` (a console one is made if omitted).

    Returns:
        The set of floorspace ids found in the file.
    """
    report = report or TextReport()
    log = report.log
    log("=" * 60, "\nOCCUPANCY FILE:", path)
    n = 0
    status, headcount = Counter(), Counter()
    per_space = {}                           # floorspaceid -> [n, tmin, tmax, max headcount]
    dup_keys, dups = set(), 0
    for ch in pd.read_csv(path, chunksize=CHUNK, dtype=str):
        n += len(ch)
        status.update(ch["occupancystatus"].fillna("NULL"))
        headcount.update(ch["headcount"].fillna("NULL"))
        t = pd.to_datetime(ch["collecteddate"], utc=True, errors="coerce")
        hc = pd.to_numeric(ch["headcount"], errors="coerce")
        g = (pd.DataFrame({"sid": ch["floorspaceid"], "t": t, "hc": hc}).groupby("sid")
             .agg(n=("t", "size"), tmin=("t", "min"), tmax=("t", "max"), hmax=("hc", "max")))
        for sid, r in g.iterrows():
            cur = per_space.get(sid)
            if cur is None:
                per_space[sid] = [r.n, r.tmin, r.tmax, r.hmax]
            else:
                cur[0] += r.n
                cur[1], cur[2] = min(cur[1], r.tmin), max(cur[2], r.tmax)
                cur[3] = max(cur[3], r.hmax)
        for k in (ch["floorspaceid"] + "|" + ch["collecteddate"] + "|" + ch["occupancystatus"]):
            if k in dup_keys:
                dups += 1
            else:
                dup_keys.add(k)
    log("rows:", n)
    log("occupancystatus:", dict(status))
    log("headcount values (top 15):", dict(headcount.most_common(15)))
    log("duplicate (space, time, status) rows:", dups)
    log("\nPer floorspace (id | events | first | last | max headcount):")
    for sid, (cnt, t0, t1, hm) in sorted(per_space.items()):
        log(f"  {sid} | {cnt} | {t0} | {t1} | {hm}")
    return set(per_space)


def profile_env(path, report=None):
    """Summarise the environmental sensor file.

    Args:
        path: Env CSV (``sensorid``, ``jsondata``, ``createdate``, ...).
        report: Optional :class:`~occenv.reporting.TextReport`.

    Returns:
        The set of sensor ids found in the file.
    """
    report = report or TextReport()
    log = report.log
    log("=" * 60, "\nENV FILE:", path)
    n = 0
    sensors = Counter()
    seen, dups = set(), 0
    sensor_times = defaultdict(set)          # sensorid -> unique measurement times
    problems = Counter()
    lag_days = []
    var_vals = defaultdict(list)
    var_units, err_codes, no_value = {}, Counter(), Counter()
    for ch in pd.read_csv(path, chunksize=CHUNK, dtype={"sensorid": str}):
        n += len(ch)
        sensors.update(ch["sensorid"])
        cdate = pd.to_datetime(ch["createdate"], utc=True, errors="coerce")
        for i, (sid, js, cd) in enumerate(zip(ch["sensorid"], ch["jsondata"], cdate)):
            m = TIME_RE.search(js) if isinstance(js, str) else None
            if not m:
                problems["no time found"] += 1
                continue
            t = int(m.group(1))
            if (sid, t) in seen:
                dups += 1
            else:
                seen.add((sid, t))
                sensor_times[sid].add(t)
            if i % JSON_SAMPLE_EVERY == 0:
                if pd.notna(cd):
                    lag_days.append((cd.timestamp() - t) / 86400)
                try:
                    for var in json.loads(js):
                        name = var["variable"]["name"]
                        var_units[name] = var["variable"].get("unit")
                        for v in var["values"]:
                            if "errorCode" in v:
                                err_codes[(name, v["errorCode"])] += 1
                            if v.get("value") is None:
                                no_value[name] += 1
                            else:
                                var_vals[name].append(v["value"])
                except (ValueError, KeyError, TypeError):
                    problems["json parse failure"] += 1
    log("rows:", n, "| unique sensors:", len(sensors), "|", dict(problems))
    log("duplicate (sensorid, time) rows:", dups, f"({100 * dups / max(n, 1):.1f}%)")
    if lag_days:
        s = pd.Series(lag_days)
        log(f"ingest-minus-measurement lag, days (sampled): median={s.median():.2f}, "
            f"p90={s.quantile(.9):.2f}, max={s.max():.1f}")
    log("\nPer sensor (id | rows | unique readings | first | last | median gap min):")
    month_counts = Counter()
    for sid in sorted(sensors):
        ts = sorted(sensor_times[sid])
        gap = (pd.Series(ts).diff().dropna() / 60).median() if len(ts) > 1 else float("nan")
        log(f"  {sid} | {sensors[sid]} | {len(ts)} | {pd.to_datetime(ts[0], unit='s')} | "
            f"{pd.to_datetime(ts[-1], unit='s')} | {gap:.1f}")
        month_counts.update(pd.to_datetime(pd.Series(ts), unit="s").dt.strftime("%Y-%m"))
    log("\nUnique readings per measurement month:", dict(sorted(month_counts.items())))
    log("\nVariable statistics (sampled 1 in %d rows; non-missing values only):" % JSON_SAMPLE_EVERY)
    for name, vals in var_vals.items():
        s = pd.Series(vals)
        log(f"  {name} [{var_units.get(name)}]: n={len(s)} min={s.min():.2f} p1={s.quantile(.01):.2f} "
            f"median={s.median():.2f} p99={s.quantile(.99):.2f} max={s.max():.2f}")
    log("errorCode occurrences (sampled):", dict(err_codes))
    log("values with NO 'value' field (sampled):", dict(no_value))
    return set(sensors)


def check_mapping(xlsx, env_sensors, occ_spaces, report=None):
    """Report how many sensors/spaces in the data are covered by the Excel mapping.

    Args:
        xlsx: Path to ``Sensor_ID_and_Locations.xlsx`` (sheets ``ENV`` and ``OCC...``).
        env_sensors: Set of env sensor ids in the data.
        occ_spaces: Set of floorspace ids in the data.
        report: Optional :class:`~occenv.reporting.TextReport`.

    If the file does not exist a message is logged and the function returns without raising.
    """
    report = report or TextReport()
    log = report.log
    log("=" * 60, "\nMAPPING COVERAGE (xlsx)")
    if not os.path.exists(xlsx):
        log(f"!! Could not find '{xlsx}' in {os.getcwd()} - check the path and re-run.")
        return
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype=str)
    env_map = sheets["ENV"].copy()
    occ_map = sheets[[k for k in sheets if k.startswith("OCC")][0]].copy()
    env_map["ID"] = env_map["ID"].map(lambda x: str(int(float(x))) if pd.notna(x) else None)
    mapped_env, mapped_occ = set(env_map["ID"].dropna()), set(occ_map["SpaceId"].dropna())
    log(f"env sensors in data: {len(env_sensors)}; matched in xlsx: {len(env_sensors & mapped_env)}")
    for _, r in env_map[env_map["ID"].isin(env_sensors)].iterrows():
        log(f"  env {r['ID']} -> building {r['Building']}, room {r['Floor/Room']}, {r['Area']}")
    log("  env sensors NOT in xlsx:", sorted(env_sensors - mapped_env))
    log(f"occupancy floorspaces in data: {len(occ_spaces)}; matched in xlsx: "
        f"{len(occ_spaces & mapped_occ)}")
    for _, r in occ_map[occ_map["SpaceId"].isin(occ_spaces)].iterrows():
        log(f"  space {r['SpaceId']} -> building {r['Building']}, room {r['Floor/Room']}, "
            f"sensor {r['ID']}, note {r['Note']}")
    log("  spaces NOT in xlsx:", sorted(occ_spaces - mapped_occ))
