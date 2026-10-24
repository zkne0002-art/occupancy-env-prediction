"""Tests for occenv.modelling: features, split, metrics, bootstrap and the full experiment."""
import warnings

import numpy as np
import pandas as pd
import pytest

from occenv.modelling import (PAIRS, ScheduleLookup, build_dataset, make_models, metrics,
                              paired_auc_diff, run_experiment, split_chronological)

SPACES = ["aaaaaaaa", "bbbbbbbb"]


@pytest.fixture(scope="session")
def experiment(processed):
    env, occ = processed
    return run_experiment(env, occ, "326", SPACES, thr=1.0, n_boot=200)


def test_schedule_lookup_rates_and_fallback():
    m = ScheduleLookup().fit(pd.DataFrame({"how": [1, 1, 2, 2, 2]}), [1, 0, 1, 1, 1])
    p = m.predict_proba(pd.DataFrame({"how": [1, 2, 99]}))[:, 1]
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(1.0)
    assert p[2] == pytest.approx(0.8)             # unseen hour -> overall rate 4/5


def test_metrics_perfect_predictions():
    m = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == 1.0 and m["bal_acc"] == 1.0 and m["f1"] == 1.0
    assert m["brier"] == pytest.approx(0.025)


def test_metrics_single_class_gives_nan_auc():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)       # sklearn warns about the single class
        assert np.isnan(metrics([1, 1, 1], [0.6, 0.7, 0.8])["auc"])


def test_gradient_boosting_never_stops_early():
    """All HGB models must use the same fixed 200 iterations, whatever the training-set size."""
    for name, (est, _) in make_models().items():
        if name.startswith("hgb"):
            assert est.early_stopping is False and est.max_iter == 200


def test_paired_auc_diff_identical_models_is_zero():
    y = np.tile([0, 1], 100)
    p = np.random.default_rng(1).random(200)
    d, lo, hi = paired_auc_diff(y, p, p, np.repeat(np.arange(20), 10), n_boot=100)
    assert d == 0 and lo == 0 and hi == 0


def test_paired_auc_diff_detects_a_better_model():
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 100)
    good, bad = y + 0.3 * rng.normal(size=200), rng.random(200)
    d, lo, hi = paired_auc_diff(y, good, bad, np.repeat(np.arange(20), 10), n_boot=200)
    assert d > 0.2 and lo > 0


def test_build_dataset_unknown_sensor_or_space_raises(processed):
    env, occ = processed
    with pytest.raises(ValueError, match="not found"):
        build_dataset(env, occ, "999", SPACES, 1.0)
    with pytest.raises(ValueError, match="not found"):
        build_dataset(env, occ, "326", ["zzzzzzzz"], 1.0)


def test_zone_is_nan_when_any_space_is_missing(processed):
    env, occ = processed
    occ2 = occ.copy()
    day = (occ2["time_utc"] >= pd.Timestamp("2025-09-05", tz="UTC")) & \
          (occ2["time_utc"] < pd.Timestamp("2025-09-06", tz="UTC"))
    occ2.loc[day & occ2["floorspaceid"].str.startswith("bbbbbbbb"), "mean_headcount"] = np.nan
    df, _ = build_dataset(env, occ2, "326", SPACES, 1.0)
    assert df.loc["2025-09-05", "zone_hc"].isna().all()
    assert df.loc["2025-09-08", "zone_hc"].notna().mean() > 0.9


def test_features_use_only_past_values(processed):
    """Changing env readings from a cut-off onward must not alter any earlier feature."""
    env, occ = processed
    base, _ = build_dataset(env, occ, "326", SPACES, 1.0)
    cut = pd.Timestamp("2025-09-15", tz="UTC")
    env2 = env.copy()
    env2.loc[env2["time_utc"] >= cut, ["co2_ppm", "temp_c"]] = 99999.0
    alt, _ = build_dataset(env2, occ, "326", SPACES, 1.0)
    before = base.index < cut
    for col in ["co2_ppm", "co2_roll1h", "co2_d30", "co2_above_base", "temp_d30"]:
        pd.testing.assert_series_equal(base.loc[before, col], alt.loc[before, col])
    assert not base.loc[~before, "co2_roll1h"].equals(alt.loc[~before, "co2_roll1h"])


def test_split_is_chronological_with_gap():
    idx = pd.date_range("2025-01-01", periods=1000, freq="10min", tz="UTC")
    train, test = split_chronological(pd.DataFrame({"x": range(1000)}, index=idx))
    assert train.index.max() < test.index.min()
    assert test.index.min() - train.index.max() >= pd.Timedelta(days=1)
    assert len(train) + len(test) < 1000                        # gap rows are unused
    with pytest.raises(ValueError):
        split_chronological(pd.DataFrame({"x": range(1000)}, index=idx), train_frac=1.5)


def test_run_experiment_outputs(experiment):
    names = set(make_models())
    assert set(experiment["test"]["model"]) == names
    assert set(experiment["cv"]["model"]) == names
    assert experiment["test"]["auc"].between(0.5, 1.0).all()
    assert len(experiment["paired"]) == len(PAIRS) == 7
    assert (experiment["paired"]["lo"] <= experiment["paired"]["hi"]).all()
    assert "co2_ppm" in experiment["importance"].index
    assert "HELD-OUT TEST" in experiment["report"].text()


def test_run_experiment_rejects_single_class_target(processed):
    env, occ = processed
    with pytest.raises(ValueError, match="one class"):
        run_experiment(env, occ, "326", SPACES, thr=1000.0, n_boot=10)
