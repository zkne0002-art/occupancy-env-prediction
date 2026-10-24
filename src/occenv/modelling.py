"""Predict zone occupancy from environmental signals and test whether they beat the schedule.

**Design**

- One env sensor and a ZONE (sum of headcount over the chosen spaces) on their common window.
- Target `occupied` = zone headcount >= `thr` (10-minute time-weighted mean headcount).
- Features: schedule (hour, day of week, hour of week) and environment (CO2, temperature,
  humidity, PM2.5, plus rate of change, 1 h mean, height above the 24 h minimum). Every
  engineered feature looks only at the PAST, so nothing leaks from the future.
- Validation: chronological split (first 70 % train, 1-day gap, last 30 % test); models are
  compared with forward-chaining CV inside the training period; the test set is scored once.
- Models: schedule lookup (baseline), logistic regression and gradient boosting, each with
  schedule only and schedule + env. The paired difference between the two is the evidence
  for what the environmental data adds beyond the clock.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BIN, ENV_ALL, ENV_BASE, ENV_CO2, TZ
from .mapping import deseason
from .reporting import TextReport

GAP_BINS = 144   # 1 day between train and test / CV folds
PAIRS = [("hgb_schedule", "schedule_lookup"), ("logreg_sched+env", "logreg_schedule"),
         ("logreg_sched+co2", "logreg_schedule"), ("hgb_sched+env", "hgb_schedule"),
         ("hgb_sched+co2", "hgb_schedule"),
         # is the simpler model good enough? (model selection evidence)
         ("hgb_sched+co2", "logreg_sched+co2"), ("hgb_sched+env", "logreg_sched+env")]


class ScheduleLookup(BaseEstimator, ClassifierMixin):
    """Baseline classifier: P(occupied) is the training occupancy rate for that hour of week.

    Expects a DataFrame with an integer ``how`` (hour of week, 0..167) column. Hours of the
    week never seen in training fall back to the overall training rate.
    """

    def fit(self, X, y):
        """Learn the occupancy rate per hour of week from ``X['how']`` and labels ``y``."""
        y = pd.Series(np.asarray(y), index=X.index)
        self.rate_ = y.groupby(X["how"]).mean()
        self.global_ = float(y.mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Return an (n, 2) array of class probabilities."""
        p = X["how"].map(self.rate_).fillna(self.global_).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Predict occupied (1) where the probability is at least 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_dataset(env, occ, sensor, spaces, thr):
    """Build the modelling table for one sensor and one zone.

    Args:
        env: Cleaned env table (``sensorid``, ``time_utc``, variables).
        occ: Cleaned occupancy table (``floorspaceid``, ``time_utc``, ``mean_headcount``).
        sensor: Suffix of the env sensor id (e.g. ``"869"``).
        spaces: List of floorspace id prefixes (8 characters) that form the zone.
        thr: Zone headcount at or above which a bin counts as ``occupied``.

    Returns:
        ``(df, per_space)``: ``df`` on a regular 10-minute UTC grid over the common window,
        with env columns, engineered features (``co2_d30``, ``co2_roll1h``, ``co2_above_base``,
        ``temp_d30``), schedule columns (``hour``, ``dow``, ``how``), ``zone_hc`` (NaN if ANY
        space is missing in that bin) and ``occupied``; ``per_space`` is the headcount per space.

    Raises:
        ValueError: If a column is missing, the sensor is not found, or a space has no data.
    """
    for cols, frame, name in [(["sensorid", "time_utc"] + ENV_BASE, env, "env"),
                              (["floorspaceid", "time_utc", "mean_headcount"], occ, "occ")]:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} table is missing column(s) {missing}")
    e = env[env["sensorid"].str.endswith(sensor)].set_index("time_utc").sort_index()
    occ = occ.assign(sp=occ["floorspaceid"].str[:8])
    wide = occ[occ["sp"].isin(spaces)].pivot(index="time_utc", columns="sp", values="mean_headcount")
    missing = set(spaces) - set(wide.columns)
    if e.empty:
        raise ValueError(f"Sensor ending '{sensor}' not found in the env table")
    if missing:
        raise ValueError(f"Space(s) {sorted(missing)} not found in the occupancy table")
    start = max([e["co2_ppm"].first_valid_index()] + [wide[c].first_valid_index() for c in wide])
    end = min([e["co2_ppm"].last_valid_index()] + [wide[c].last_valid_index() for c in wide])
    if start >= end:
        raise ValueError("The sensor and the spaces have no overlapping time window")
    grid = pd.date_range(start, end, freq=BIN)
    df = e.reindex(grid)[ENV_BASE].copy()
    per_space = wide.reindex(grid)
    df["zone_hc"] = per_space.sum(axis=1, min_count=len(spaces))
    df["co2_d30"] = df["co2_ppm"] - df["co2_ppm"].shift(3)
    df["co2_roll1h"] = df["co2_ppm"].rolling(6, min_periods=4).mean()
    df["co2_above_base"] = df["co2_ppm"] - df["co2_ppm"].rolling(144, min_periods=72).min()
    df["temp_d30"] = df["temp_c"] - df["temp_c"].shift(3)
    local = grid.tz_convert(TZ)
    df["hour"] = np.asarray(local.hour)
    df["dow"] = np.asarray(local.dayofweek)
    df["how"] = df["dow"] * 24 + df["hour"]
    df["occupied"] = (df["zone_hc"] >= thr).astype(int)
    return df, per_space


def describe(df, per_space, thr, report):
    """Log the window, missingness, class balance and CO2/headcount correlations to ``report``."""
    log = report.log
    log(f"Window: {df.index.min()} -> {df.index.max()}  ({len(df)} bins, "
        f"{(df.index.max() - df.index.min()).days} days)")
    log(f"Missing: co2 {100 * df['co2_ppm'].isna().mean():.1f}%, "
        f"zone headcount {100 * df['zone_hc'].isna().mean():.1f}%")
    z = df["zone_hc"].dropna()
    log("Zone headcount quantiles (p10/p50/p90/p99/max): "
        + "/".join(f"{z.quantile(q):.1f}" for q in (.1, .5, .9, .99, 1)))
    log("Fraction of bins with zone headcount >= t:  "
        + ", ".join(f"t={t}: {100 * (z >= t).mean():.0f}%" for t in (0.5, 1, 2, 5, 10, 20)))
    log(f"Chosen threshold {thr}: occupied = {100 * (z >= thr).mean():.1f}% of valid bins")
    wk = df[(df["dow"] < 5) & df["zone_hc"].notna()]
    log("Weekday mean zone headcount by hour: "
        + " ".join(f"{h}:{v:.1f}" for h, v in wk.groupby("hour")["zone_hc"].mean().items()))
    res = deseason(pd.concat([df[["co2_ppm"]], per_space, df[["zone_hc"]]], axis=1))
    log("CO2 vs headcount, residual corr (hour-of-week profile removed): "
        + ", ".join(f"{c}={res['co2_ppm'].corr(res[c]):.2f}" for c in per_space.columns)
        + f", zone_sum={res['co2_ppm'].corr(res['zone_hc']):.2f}")


def make_models():
    """Return ``{name: (estimator, feature columns)}`` for every model compared."""
    def hgb():
        # early_stopping=False: scikit-learn's default ('auto') switches early stopping on for
        # training sets above 10,000 rows, using a RANDOM validation split. That would train the
        # larger control dataset differently from the main experiment (and leak within the
        # training period), so every model uses exactly 200 boosting iterations.
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              early_stopping=False, random_state=0)
    how = ("how", OneHotEncoder(handle_unknown="ignore"), ["how"])
    lr_sched = ColumnTransformer([how])
    lr_all = ColumnTransformer([how, ("env", StandardScaler(), ENV_ALL)])
    lr_co2 = ColumnTransformer([how, ("env", StandardScaler(), ENV_CO2)])
    return {
        "schedule_lookup": (ScheduleLookup(), ["how"]),
        "logreg_schedule": (make_pipeline(lr_sched, LogisticRegression(max_iter=2000)), ["how"]),
        "logreg_sched+env": (make_pipeline(lr_all, LogisticRegression(max_iter=2000)),
                             ["how"] + ENV_ALL),
        "hgb_schedule": (hgb(), ["hour", "dow"]),
        "hgb_env_only": (hgb(), ENV_ALL),
        "hgb_sched+env": (hgb(), ["hour", "dow"] + ENV_ALL),
        "logreg_sched+co2": (make_pipeline(lr_co2, LogisticRegression(max_iter=2000)),
                             ["how"] + ENV_CO2),
        "hgb_sched+co2": (hgb(), ["hour", "dow"] + ENV_CO2),
    }


def metrics(y, p):
    """Score probabilities ``p`` against binary labels ``y``.

    Returns:
        Dict with ``auc`` (NaN if ``y`` has one class), ``bal_acc`` and ``f1`` (at 0.5), ``brier``.
    """
    y = pd.Series(np.asarray(y))
    p = np.asarray(p)
    return {"auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
            "bal_acc": balanced_accuracy_score(y, p >= 0.5),
            "f1": f1_score(y, p >= 0.5, zero_division=0),
            "brier": brier_score_loss(y, p)}


def paired_auc_diff(y, p_a, p_b, days, n_boot=1000, seed=0):
    """AUC(a) - AUC(b) with a 95 % day-block bootstrap interval.

    Whole days are resampled together because 10-minute bins within a day are strongly
    autocorrelated; resampling single bins would give falsely narrow intervals.

    Args:
        y: Binary labels.
        p_a: Predicted probabilities of model A.
        p_b: Predicted probabilities of model B.
        days: Day label per row (rows with the same label are resampled together).
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        ``(observed difference, lower 2.5 %, upper 97.5 %)``.
    """
    rng = np.random.default_rng(seed)
    y, p_a, p_b, days = map(np.asarray, (y, p_a, p_b, days))
    groups = [np.flatnonzero(days == d) for d in np.unique(days)]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return roc_auc_score(y, p_a) - roc_auc_score(y, p_b), lo, hi


def split_chronological(data, train_frac=0.7, gap_days=1):
    """Split a time-indexed table into an earlier train part and a later test part.

    Args:
        data: DataFrame with a sorted DatetimeIndex.
        train_frac: Fraction of rows (by time order) before the test period starts.
        gap_days: Days left unused before the test period, so neighbouring bins do not leak.

    Returns:
        ``(train, test)`` with no time overlap.

    Raises:
        ValueError: If ``train_frac`` is not strictly between 0 and 1.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be strictly between 0 and 1")
    cut = data.index[int(len(data) * train_frac)]
    return data[data.index < cut - pd.Timedelta(days=gap_days)], data[data.index >= cut]


def cross_validate(models, train, y, n_splits=4, gap=GAP_BINS):
    """Forward-chaining CV (ROC-AUC) inside the training period.

    Returns:
        DataFrame with ``model``, ``mean_auc``, ``sd_auc`` and ``folds`` (folds usable).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []
    for name, (est, cols) in models.items():
        aucs = []
        for tr, va in tscv.split(train):
            if y.iloc[tr].nunique() < 2 or y.iloc[va].nunique() < 2:
                continue
            m = clone(est).fit(train.iloc[tr][cols], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(train.iloc[va][cols])[:, 1]))
        rows.append({"model": name, "mean_auc": np.mean(aucs) if aucs else np.nan,
                     "sd_auc": np.std(aucs) if aucs else np.nan, "folds": len(aucs)})
    return pd.DataFrame(rows)


def evaluate_on_test(models, train, test):
    """Fit each model on ``train`` and score it once on ``test``.

    Returns:
        ``(table, preds, fitted)``: ``table`` has the metrics plus ``fit_s``/``predict_s``
        timings; ``preds`` maps model name to test probabilities; ``fitted`` maps name to
        ``(fitted model, feature columns)``.
    """
    rows, preds, fitted = [], {}, {}
    for name, (est, cols) in models.items():
        t0 = time.perf_counter()
        m = clone(est).fit(train[cols], train["occupied"])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = m.predict_proba(test[cols])[:, 1]
        t_pred = time.perf_counter() - t0
        rows.append({"model": name, **metrics(test["occupied"], p),
                     "fit_s": t_fit, "predict_s": t_pred})
        preds[name], fitted[name] = p, (m, cols)
    return pd.DataFrame(rows), preds, fitted


def run_experiment(env, occ, sensor, spaces, thr=1.0, report=None, n_boot=1000, n_splits=4):
    """Run the whole experiment for one sensor and zone.

    Args:
        env: Cleaned env table.
        occ: Cleaned occupancy table.
        sensor: Suffix of the env sensor id.
        spaces: Floorspace id prefixes forming the zone.
        thr: Zone headcount threshold for ``occupied``.
        report: Optional `occenv.reporting.TextReport` that receives all output.
        n_boot: Bootstrap resamples for the paired AUC differences.
        n_splits: Forward-chaining CV folds.

    Returns:
        Dict with ``cv``, ``test``, ``paired`` (A, B, diff, lo, hi), ``importance`` (permutation
        importance of ``hgb_sched+env`` on test) and ``report``.

    Raises:
        ValueError: If the train or test period contains only one class (adjust ``thr``).
    """
    report = report or TextReport(echo=False)
    log = report.log
    df, per_space = build_dataset(env, occ, sensor, spaces, thr)
    log(f"Sensor ...{sensor}, zone = {list(spaces)}")
    describe(df, per_space, thr, report)

    data = df.dropna(subset=ENV_ALL + ["zone_hc"])
    log(f"Rows with complete features and target: {len(data)} of {len(df)}")
    train, test = split_chronological(data)
    ytr, yte = train["occupied"], test["occupied"]
    log(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} bins, "
        f"{100 * ytr.mean():.0f}% occupied) | Test: {test.index.min().date()} -> "
        f"{test.index.max().date()} ({len(test)} bins, {100 * yte.mean():.0f}% occupied)")
    if yte.nunique() < 2 or ytr.nunique() < 2:
        raise ValueError("The train or test period contains one class only; adjust the threshold")

    models = make_models()
    cv = cross_validate(models, train, ytr, n_splits=n_splits)
    log("\nForward-chaining CV on TRAIN only (ROC-AUC mean +- sd over folds):")
    for r in cv.itertuples():
        log(f"  {r.model:18s} {r.mean_auc:.3f} +- {r.sd_auc:.3f}  (folds used: {r.folds})")

    table, preds, fitted = evaluate_on_test(models, train, test)
    log("\nHELD-OUT TEST (fit on full train, scored once):")
    log(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    days = test.index.tz_convert(TZ).strftime("%Y-%m-%d")
    log(f"\nPaired test-AUC differences, A - B (95% bootstrap CI over {len(set(days))} test days; "
        "a CI that includes 0 means no reliable difference):")
    prow = []
    for a, b in PAIRS:
        d, lo, hi = paired_auc_diff(yte, preds[a], preds[b], days, n_boot=n_boot)
        prow.append({"A": a, "B": b, "diff": d, "lo": lo, "hi": hi})
        log(f"  {a:18s} - {b:16s}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    m, cols = fitted["hgb_sched+env"]
    pi = permutation_importance(m, test[cols], yte, scoring="roc_auc", n_repeats=10, random_state=0)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    log("\nPermutation importance on test (drop in AUC), hgb_sched+env:")
    log(imp.round(3).to_string())
    return {"cv": cv, "test": table, "paired": pd.DataFrame(prow), "importance": imp,
            "report": report}
