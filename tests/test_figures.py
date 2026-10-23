"""Tests for occenv.figures: every figure is written, is a valid PNG and the audit numbers are right."""
import os

import pytest

from occenv.cli import main
from occenv.figures import compute_audit, make_all

EXPECTED = ["fig1_data_audit.png", "fig2_availability.png", "fig3_weekly_profile.png",
            "fig4_co2_vs_headcount.png", "fig5_mapping.png", "fig6_forest.png", "fig7_test_auc.png"]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="session")
def proc_dir(raw, tmp_path_factory):
    proc = tmp_path_factory.mktemp("proc")
    main(["prepare", "--occ", raw["occ"], "--env", raw["env"], "--out", str(proc)])
    return str(proc)


def test_make_all_writes_every_figure(raw, proc_dir, tmp_path):
    written = make_all(proc_dir, str(tmp_path), env_raw=raw["env"], occ_raw=raw["occ"], sensor="326",
                       spaces=("aaaaaaaa", "bbbbbbbb"), control="326", mapping_end="2025-09-20",
                       mapping_exclude=("bbbbbbbb",), thresholds=(1.0, 2.0), n_boot=20)
    assert sorted(os.path.basename(p) for p in written) == sorted(EXPECTED)
    for p in written:
        with open(p, "rb") as f:
            assert f.read(8) == PNG_MAGIC and os.path.getsize(p) > 5_000


def test_figures_work_without_control_or_raw_files(proc_dir, tmp_path):
    written = make_all(proc_dir, str(tmp_path), sensor="326", spaces=("aaaaaaaa", "bbbbbbbb"),
                       control=None, mapping_end="2025-09-20", mapping_exclude=("bbbbbbbb",),
                       thresholds=(1.0,), n_boot=20)
    names = {os.path.basename(p) for p in written}
    assert "fig1_data_audit.png" not in names and "fig6_forest.png" in names


def test_audit_numbers_match_the_synthetic_raw_data(raw, proc_dir):
    import pandas as pd
    occ = pd.read_csv(os.path.join(proc_dir, "occupancy_10min.csv.gz"))
    audit = compute_audit(raw["env"], raw["occ"], occ)
    st = audit["env_stats"]
    assert st["unique_per_sensor"]["6012002000777"] == 1          # the dead sensor
    assert st["unique_per_sensor"]["6012002000326"] > 2000
    assert st["raw rows"] == st["unique readings"] + st["duplicate rows removed"]
    assert audit["event_share"] == 1.0                             # synthetic events are all 'CurrentlyOccupied'
    assert 0.1 < audit["time_share"] < 0.6                         # time-weighted share is much lower


def test_cli_figures_command(raw, proc_dir, tmp_path):
    assert main(["figures", "--dir", proc_dir, "--occ", raw["occ"], "--env", raw["env"], "--out",
                 str(tmp_path / "f"), "--sensor", "326", "--spaces", "aaaaaaaa", "bbbbbbbb",
                 "--no-control", "--mapping-end", "2025-09-20", "--mapping-exclude", "bbbbbbbb",
                 "--thr", "1", "--n-boot", "20"]) == 0
    assert sorted(os.listdir(tmp_path / "f")) == sorted(EXPECTED)   # raw files given -> all 7 figures
