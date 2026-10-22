"""Command line interface: `python -m occenv <command> --help`.

Commands (run in this order):

- `profile`: audit the raw files (read-only)
- `prepare`: clean the raw files and write 10-minute tables to `data/processed`
- `screen`: quick sensor-vs-space correlation table
- `diagnose`: fair-window, lag, placebo and stability tests for the sensor-to-space mapping
- `train`: build features, validate chronologically, compare models
- `benchmark`: time and profile the baseline vs improved code (performance section)
"""
import argparse
import logging
import os
import sys

import pandas as pd

from . import cleaning, mapping, modelling, performance, profiling
from .reporting import TextReport


def _load_processed(directory):
    """Load the cleaned tables written by ``prepare``; exit with a clear message if absent."""
    try:
        env = pd.read_csv(os.path.join(directory, "env_10min.csv.gz"), dtype={"sensorid": str})
        occ = pd.read_csv(os.path.join(directory, "occupancy_10min.csv.gz"))
    except FileNotFoundError as err:
        raise SystemExit(f"{err}\nRun 'python -m occenv prepare' first (or pass --dir).")
    env["time_utc"] = pd.to_datetime(env["time_utc"], utc=True)
    occ["time_utc"] = pd.to_datetime(occ["time_utc"], utc=True)
    return env, occ


def cmd_profile(a):
    """Audit the raw files and save ``--out``."""
    rep = TextReport()
    spaces = profiling.profile_occupancy(a.occ, rep)
    sensors = profiling.profile_env(a.env, rep)
    if a.xlsx:
        profiling.check_mapping(a.xlsx, sensors, spaces, rep)
    rep.save(a.out)


def cmd_prepare(a):
    """Clean both raw files and write the processed tables."""
    os.makedirs(a.out, exist_ok=True)
    env, _ = cleaning.clean_env(a.env)
    env.to_csv(os.path.join(a.out, "env_10min.csv.gz"), index=False)
    cleaning.clean_occupancy(a.occ).to_csv(
        os.path.join(a.out, "occupancy_10min.csv.gz"), index=False)
    print(f"Wrote cleaned files to {a.out}")


def cmd_screen(a):
    """Print and save the quick correlation screen."""
    env, occ = _load_processed(a.dir)
    res = mapping.screen_pairs(env, occ)
    rep = TextReport()
    rep.log(res.to_string(index=False, float_format=lambda x: f"{x:.2f}") if len(res)
            else "No sensor/space pair has enough overlapping data.")
    rep.save(a.out)


def cmd_diagnose(a):
    """Print and save the mapping diagnostics."""
    env, occ = _load_processed(a.dir)
    rep = TextReport()
    rep.log(mapping.diagnose_all(env, occ, end=a.end, exclude=a.exclude))
    rep.save(a.out)


def cmd_train(a):
    """Run the modelling experiment and save the report."""
    env, occ = _load_processed(a.dir)
    rep = TextReport()
    try:
        modelling.run_experiment(env, occ, a.sensor, a.spaces, a.thr, rep, n_boot=a.n_boot)
    except ValueError as err:
        raise SystemExit(f"Error: {err}")
    rep.save(a.out)


def cmd_benchmark(a):
    """Run the performance study and save ``--out``."""
    rep = TextReport()
    performance.run_benchmark(a.occ, a.env, a.nrows, rep)
    rep.save(a.out)


def build_parser():
    """Create the argument parser with one sub-command per pipeline stage."""
    ap = argparse.ArgumentParser(prog="occenv", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="audit the raw CSV files")
    p.add_argument("--occ", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--xlsx", help="Sensor_ID_and_Locations.xlsx (optional)")
    p.add_argument("--out", default="results/profile_summary.txt")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("prepare", help="clean raw files into 10-minute tables")
    p.add_argument("--occ", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--out", default="data/processed")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("screen", help="quick sensor-vs-space correlation table")
    p.add_argument("--dir", default="data/processed")
    p.add_argument("--out", default="results/mapping_screen.txt")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("diagnose", help="mapping diagnostics (window, lag, placebo, stability)")
    p.add_argument("--dir", default="data/processed")
    p.add_argument("--end", help="use only data BEFORE this UTC date, e.g. 2024-05-14")
    p.add_argument("--exclude", nargs="*", default=[], help="space id prefixes to leave out")
    p.add_argument("--out", default="results/mapping_diagnostics.txt")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("train", help="chronological validation and model comparison")
    p.add_argument("--dir", default="data/processed")
    p.add_argument("--sensor", required=True, help="suffix of the env sensor id, e.g. 869")
    p.add_argument("--spaces", nargs="+", required=True, help="floorspace id prefixes (8 chars)")
    p.add_argument("--thr", type=float, default=1.0, help="zone headcount threshold for occupied")
    p.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    p.add_argument("--out", default="results/model_results.txt")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("benchmark", help="time and profile baseline vs improved code")
    p.add_argument("--occ", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--nrows", type=int, default=1_000_000,
                   help="largest occupancy sample size (smaller sizes are also run)")
    p.add_argument("--out", default="results/performance.txt")
    p.set_defaults(func=cmd_benchmark)
    return ap


def main(argv=None):
    """Entry point. Returns 0 on success."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0
