"""Figures for the written report, regenerated from the data and the model results.

Every figure is produced by code (nothing is drawn by hand), so the report's figures can be
reproduced with ``python -m occenv figures``. The figures are:

1. `fig_data_audit`: how much of the raw data is usable (repeats, dead sensors, event vs time share)
2. `fig_availability`: when each sensor and space has data, and the test period
3. `fig_weekly_profile`: mean zone headcount by hour, weekday vs weekend
4. `fig_co2_vs_headcount`: a typical week of CO2 and headcount, and residual scatter plots
5. `fig_mapping`: sensor-vs-space evidence for the inferred mapping
6. `fig_forest`: paired AUC differences with 95 % intervals
7. `fig_test_auc`: held-out test AUC of the key models
"""
import logging
import os
import warnings

import matplotlib

matplotlib.use("Agg")                      # no display needed
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.patches import Patch       # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

from . import cleaning                                             # noqa: E402
from .config import TZ                                             # noqa: E402
from .mapping import deseason, diagnose_sensor                     # noqa: E402
from .modelling import (PAIRS, build_dataset, run_experiment,      # noqa: E402
                        split_chronological)
from .config import ENV_ALL                                        # noqa: E402

warnings.filterwarnings("ignore", message="Using categorical units")
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)   # harmless axis notice
DPI = 200
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"
PAIR_LABELS = {
    ("hgb_schedule", "schedule_lookup"): "HGB schedule vs lookup",
    ("logreg_sched+env", "logreg_schedule"): "LogReg: + all env vs schedule",
    ("logreg_sched+co2", "logreg_schedule"): "LogReg: + CO2 vs schedule",
    ("hgb_sched+env", "hgb_schedule"): "HGB: + all env vs schedule",
    ("hgb_sched+co2", "hgb_schedule"): "HGB: + CO2 vs schedule",
    ("hgb_sched+co2", "logreg_sched+co2"): "HGB vs LogReg (CO2)",
    ("hgb_sched+env", "logreg_sched+env"): "HGB vs LogReg (all env)",
}
AUC_MODELS = ["schedule_lookup", "logreg_schedule", "logreg_sched+co2",
              "hgb_schedule", "hgb_sched+co2", "hgb_env_only"]


def _save(fig, path):
    """Tidy the layout, write the PNG to ``path`` (creating folders) and close the figure."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def compute_audit(env_path, occ_path, occ_processed):
    """Collect the numbers behind the data-audit figure from the RAW files.

    Args:
        env_path: Raw env CSV.
        occ_path: Raw occupancy CSV (only the status column is read).
        occ_processed: Cleaned occupancy table (for the time-weighted occupied share).

    Returns:
        Dict with ``env_stats`` (from `occenv.cleaning.clean_env`), ``event_share`` (share of raw
        events labelled CurrentlyOccupied) and ``time_share`` (share of valid time with headcount > 0).
    """
    _, env_stats = cleaning.clean_env(env_path)
    status = pd.read_csv(occ_path, usecols=["occupancystatus"],
                         dtype="category")["occupancystatus"].value_counts()
    return {"env_stats": env_stats,
            "event_share": float(status.get("CurrentlyOccupied", 0) / status.sum()),
            "time_share": float(occ_processed["frac_occupied"].mean())}


def fig_data_audit(audit, path):
    """Three panels: env repeats, unique readings per sensor, occupancy event vs time share."""
    st = audit["env_stats"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
    raw, dup, uniq = st["raw rows"], st.get("duplicate rows removed", 0), st["unique readings"]
    ax[0].barh(["env rows"], [uniq], color=BLUE, label=f"unique readings: {uniq:,}")
    ax[0].barh(["env rows"], [raw - uniq], left=[uniq], color=GREY,
               label=f"repeats/unusable: {raw - uniq:,} ({100 * (raw - uniq) / raw:.0f}%)")
    ax[0].set(title="A. Env rows in the raw file", xlabel="rows (thousands)")
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
    ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}"))
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    per = st["unique_per_sensor"]
    sensors = sorted(per)
    kept = set(st["sensors_kept"])
    bars = ax[1].bar([f"...{s[-3:]}" for s in sensors], [per[s] for s in sensors],
                     color=[BLUE if s in kept else RED for s in sensors])
    ax[1].set(title="B. Unique readings per sensor", ylabel="unique readings (log scale)",
              yscale="log")
    for b, s in zip(bars, sensors):
        ax[1].text(b.get_x() + b.get_width() / 2, per[s] * 1.15, f"{per[s]:,}", ha="center",
                   fontsize=8)
    ax[1].set_ylim(0.7, max(per.values()) * 4)
    ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)],
                 labels=["kept", "dropped (dead)"], frameon=False, loc="upper center",
                 bbox_to_anchor=(0.5, -0.18), ncol=2)
    shares = [100 * audit["event_share"], 100 * audit["time_share"]]
    ax[2].bar(["events labelled\nCurrentlyOccupied", "time with\nheadcount > 0"], shares,
              color=[GREY, BLUE])
    for i, v in enumerate(shares):
        ax[2].text(i, v + 1.5, f"{v:.0f}%", ha="center")
    ax[2].set(title="C. Occupancy: events vs time", ylabel="percent", ylim=(0, 118),
              yticks=range(0, 101, 20))
    return _save(fig, path)


def _span(frame, col):
    valid = frame.dropna(subset=[col])
    return valid["time_utc"].min(), valid["time_utc"].max()


def _num(ts):
    return mdates.date2num(ts.tz_convert(None).to_pydatetime())


def fig_availability(env, occ, path, test_window=None):
    """Timeline of the first-to-last valid bin of every sensor (blue) and space (orange)."""
    rows = []
    for sid, e in env.groupby("sensorid"):
        rows.append((f"env ...{sid[-3:]}", *_span(e, "co2_ppm"), BLUE))
    for fs, o in occ.groupby("floorspaceid"):
        rows.append((f"space {fs[:4]}", *_span(o, "mean_headcount"), ORANGE))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.3))
    for i, (label, t0, t1, colour) in enumerate(rows):
        ax.barh(i, _num(t1) - _num(t0), left=_num(t0), color=colour, height=0.6)
    handles = [Patch(color=BLUE, label="environmental sensor"),
               Patch(color=ORANGE, label="occupancy space")]
    if test_window is not None:
        ax.axvspan(_num(test_window[0]), _num(test_window[1]), color=RED, alpha=0.2)
        handles.append(Patch(color=RED, alpha=0.2, label="main experiment: test period"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set(xlabel="first to last valid 10-minute bin")
    return _save(fig, path)


def fig_weekly_profile(datasets, path):
    """Mean zone headcount by local hour, weekday vs weekend, one panel per dataset.

    Args:
        datasets: Dict ``{panel title: DataFrame from build_dataset}``.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), sharey=True,
                             squeeze=False)
    for ax, (title, df) in zip(axes[0], datasets.items()):
        d = df.dropna(subset=["zone_hc"])
        for name, mask, colour in (("weekday", d["dow"] < 5, BLUE), ("weekend", d["dow"] >= 5, ORANGE)):
            prof = d[mask].groupby("hour")["zone_hc"].mean()
            ax.plot(prof.index, prof.values, label=name, color=colour, marker="o", ms=3)
        ax.set(title=title, xlabel="hour of day (local)", xticks=range(0, 24, 3))
        ax.legend(frameon=False)
    axes[0][0].set_ylabel("mean zone headcount (people)")
    return _save(fig, path)


def _typical_week(d):
    """Pick the (nearly complete) week whose mean headcount is closest to the median week."""
    local = d.index.tz_convert(TZ)
    week = (local.normalize() - pd.to_timedelta(local.dayofweek, unit="D")).tz_localize(None)
    g = d.groupby(np.asarray(week))
    full = g.size()[g.size() >= 0.8 * 1008].index
    if len(full) == 0:
        full = g.size().index
    means = g["zone_hc"].mean().loc[full]
    return means.index[int(np.argmin(np.abs(means - means.median())))]


def fig_co2_vs_headcount(df_main, path, label_main, df_control=None, label_control=""):
    """A typical week of CO2 and zone headcount, plus residual scatter for the main and control sensor.

    The week is chosen automatically (mean headcount closest to the median week), not by eye.
    Residuals have the mean hour-of-week profile removed.
    """
    d = df_main.dropna(subset=["co2_ppm", "zone_hc"])
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2 if df_control is not None else 1, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, :])
    start = _typical_week(d)
    local = d.index.tz_convert(TZ).tz_localize(None)
    wk = d[(local >= start) & (local < start + pd.Timedelta(days=7))]
    x = wk.index.tz_convert(TZ).tz_localize(None)
    ax.plot(x, wk["co2_ppm"], color=BLUE, lw=1)
    ax.set(ylabel="CO2 (ppm)", title=f"A. {label_main}: typical week from {start.date()}")
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.plot(x, wk["zone_hc"], color=ORANGE, lw=1)
    ax2.set_ylabel("zone headcount (people)")
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    panels = [(df_main, f"B. {label_main}: residuals")]
    if df_control is not None:
        panels.append((df_control, f"C. {label_control}: residuals"))
    resids = [deseason(df.dropna(subset=["co2_ppm", "zone_hc"])[["co2_ppm", "zone_hc"]])
              for df, _ in panels]
    lo = min(r["co2_ppm"].quantile(0.005) for r in resids)      # shared range: 0.5th-99.5th percentile
    hi = max(r["co2_ppm"].quantile(0.995) for r in resids)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    for j, ((df, title), res) in enumerate(zip(panels, resids)):
        a = fig.add_subplot(gs[1, j])
        keep = res["co2_ppm"].between(lo, hi)
        a.hexbin(res.loc[keep, "zone_hc"], res.loc[keep, "co2_ppm"], gridsize=30, mincnt=1,
                 bins="log", cmap="Blues")
        r = res["co2_ppm"].corr(res["zone_hc"])                 # r uses ALL points, drawn or not
        a.set(title=f"{title} (r = {r:.2f})", xlabel="zone headcount minus weekly profile",
              ylabel="CO2 minus weekly profile (ppm)", ylim=(lo, hi))
        a.axhline(0, color=GREY, lw=0.5)
        a.axvline(0, color=GREY, lw=0.5)
        if (~keep).sum():
            a.text(0.98, 0.02, f"{int((~keep).sum())} extreme points not drawn (included in r)",
                   transform=a.transAxes, ha="right", va="bottom", fontsize=7)
    return _save(fig, path)


def mapping_result(env, occ, sensor, end=None, exclude=()):
    """Run the fair-window mapping diagnostics for one sensor (see `occenv.mapping.diagnose_all`)."""
    occ = occ.assign(floorspaceid=occ["floorspaceid"].str[:8])
    occ = occ[~occ["floorspaceid"].isin(list(exclude))]
    env = env[env["sensorid"].str.endswith(sensor)]
    if end is not None:
        cut = pd.Timestamp(end, tz="UTC")
        env, occ = env[env["time_utc"] < cut], occ[occ["time_utc"] < cut]
    if env.empty or occ.empty:
        return None
    wide = occ.pivot(index="time_utc", columns="floorspaceid", values="mean_headcount")
    e = env.set_index("time_utc")
    lo, hi = e["co2_ppm"].first_valid_index(), e["co2_ppm"].last_valid_index()
    return diagnose_sensor(e, wide.loc[lo:hi].dropna(axis=1, thresh=500))


def fig_mapping(results, zone, path):
    """Residual CO2-headcount correlation per space, one panel per sensor.

    Args:
        results: Dict ``{panel title: result of mapping_result}`` (None entries are skipped).
        zone: Space id prefixes of the chosen zone (drawn in orange).
        path: Output PNG.
    """
    results = {k: v for k, v in results.items() if v is not None}
    allr = pd.concat([r["table"]["resid"] for r in results.values()])
    xlim = (min(0.0, allr.min()) - 0.03, max(0.05, allr.max()) * 1.35 + 0.04)
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 3.6), sharex=True,
                             squeeze=False)
    for ax, (title, res) in zip(axes[0], results.items()):
        t = res["table"]
        colours = [ORANGE if s in zone else GREY for s in t["space"]]
        ax.barh(t["space"], t["resid"], color=colours)
        for i, (r, z) in enumerate(zip(t["resid"], t["z"])):
            ztxt = f"z = {z:.1f}" if pd.notna(z) else ""
            ax.text(max(r, 0) + 0.01, i, ztxt, va="center", ha="left", fontsize=8)
        ax.axvline(0, color="black", lw=0.6)
        ax.invert_yaxis()
        ax.set(title=f"{title}\n({res['start'].date()} to {res['end'].date()}, {res['days']} days)",
               xlabel="residual correlation: CO2 vs headcount", xlim=xlim)
    return _save(fig, path)


def fig_forest(results, path):
    """Forest plot of paired test-AUC differences (A - B) with 95 % day-block bootstrap intervals.

    Args:
        results: Dict ``{experiment label: paired DataFrame from run_experiment}``.
    """
    pairs = [p for p in PAIRS if any(((df["A"] == p[0]) & (df["B"] == p[1])).any()
                                     for df in results.values())]
    n = len(results)
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(pairs) + 1.8))
    colours = [BLUE, GREEN, ORANGE, RED, GREY]
    for j, (label, df) in enumerate(results.items()):
        for i, p in enumerate(pairs):
            row = df[(df["A"] == p[0]) & (df["B"] == p[1])]
            if row.empty:
                continue
            r = row.iloc[0]
            y = i + (j - (n - 1) / 2) * 0.2
            ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="o",
                        color=colours[j % len(colours)], capsize=2, ms=4,
                        label=label if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(pairs)), [PAIR_LABELS[p] for p in pairs])
    ax.invert_yaxis()
    ax.set(xlabel="difference in test AUC, A minus B (95% interval)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _save(fig, path)


def fig_test_auc(experiments, path):
    """Grouped bars of held-out test AUC for the key models, one group per experiment.

    Args:
        experiments: Dict ``{label: test-results DataFrame from run_experiment}``.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    k = len(AUC_MODELS)
    width = 0.8 / k
    colours = [GREY, "#aec7e8", BLUE, "#ffbb78", ORANGE, GREEN]
    for m, (name, colour) in enumerate(zip(AUC_MODELS, colours)):
        vals = [df.set_index("model").loc[name, "auc"] for df in experiments.values()]
        pos = np.arange(len(vals)) + (m - (k - 1) / 2) * width
        ax.bar(pos, vals, width, label=name, color=colour)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.004, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(experiments)), list(experiments))
    ax.set(ylabel="held-out test AUC", ylim=(0.5, 1.09), yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, path)


def make_all(processed_dir, out_dir, env_raw=None, occ_raw=None, sensor="869",
             spaces=("0f3f37f8", "941f0352"), control="326", mapping_end="2024-05-14",
             mapping_exclude=("3e5a14df",), thresholds=(1.0, 2.0), n_boot=1000):
    """Re-run the analyses and write every report figure into ``out_dir``.

    Args:
        processed_dir: Folder with the cleaned tables written by ``prepare``.
        out_dir: Output folder for the PNG files.
        env_raw: Raw env CSV (needed for figure 1; figure 1 is skipped if omitted).
        occ_raw: Raw occupancy CSV (needed for figure 1).
        sensor: Suffix of the main sensor.
        spaces: Zone (floorspace id prefixes) for the main experiment.
        control: Suffix of the negative-control sensor, or None.
        mapping_end: Mapping evidence for the main sensor uses data before this date only.
        mapping_exclude: Spaces left out of the main sensor's mapping evidence.
        thresholds: Headcount thresholds to run for the main sensor.
        n_boot: Bootstrap resamples per paired comparison.

    Returns:
        List of written file paths.
    """
    env = pd.read_csv(os.path.join(processed_dir, "env_10min.csv.gz"), dtype={"sensorid": str})
    occ = pd.read_csv(os.path.join(processed_dir, "occupancy_10min.csv.gz"))
    env["time_utc"] = pd.to_datetime(env["time_utc"], utc=True)
    occ["time_utc"] = pd.to_datetime(occ["time_utc"], utc=True)
    out = lambda name: os.path.join(out_dir, name)
    written = []

    if env_raw and occ_raw:
        written.append(fig_data_audit(compute_audit(env_raw, occ_raw, occ), out("fig1_data_audit.png")))

    thr0 = thresholds[0]
    df_main, _ = build_dataset(env, occ, sensor, list(spaces), thr0)
    train, test = split_chronological(df_main.dropna(subset=ENV_ALL + ["zone_hc"]))
    written.append(fig_availability(env, occ, out("fig2_availability.png"),
                                    (test.index.min(), test.index.max())))

    df_ctrl = None
    datasets = {f"sensor ...{sensor} zone": df_main}
    if control:
        df_ctrl, _ = build_dataset(env, occ, control, list(spaces), thr0)
        datasets[f"control ...{control} zone"] = df_ctrl
    written.append(fig_weekly_profile(datasets, out("fig3_weekly_profile.png")))
    written.append(fig_co2_vs_headcount(df_main, out("fig4_co2_vs_headcount.png"),
                                        f"sensor ...{sensor}", df_ctrl, f"control ...{control}"))

    maps = {f"sensor ...{sensor} (before {mapping_end})":
            mapping_result(env, occ, sensor, mapping_end, mapping_exclude)}
    if control:
        maps[f"control ...{control} (all data)"] = mapping_result(env, occ, control)
    written.append(fig_mapping(maps, spaces, out("fig5_mapping.png")))

    paired, tests = {}, {}
    for thr in thresholds:
        r = run_experiment(env, occ, sensor, list(spaces), thr, n_boot=n_boot)
        paired[f"...{sensor}, threshold {thr:g}"] = r["paired"]
        tests[f"...{sensor}\nthreshold {thr:g}"] = r["test"]
    if control:
        r = run_experiment(env, occ, control, list(spaces), thr0, n_boot=n_boot)
        paired[f"control ...{control}, threshold {thr0:g}"] = r["paired"]
        tests[f"control ...{control}\nthreshold {thr0:g}"] = r["test"]
    written.append(fig_forest(paired, out("fig6_forest.png")))
    written.append(fig_test_auc(tests, out("fig7_test_auc.png")))
    return written
