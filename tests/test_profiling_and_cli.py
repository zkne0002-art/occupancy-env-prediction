"""Smoke and integration tests for occenv.profiling and the command line interface."""
import pandas as pd
import pytest
from synthetic import DEAD_SENSOR, GOOD_SENSOR

from occenv.cli import main
from occenv.profiling import check_mapping, profile_env, profile_occupancy
from occenv.reporting import TextReport


def test_profile_occupancy_finds_spaces_and_duplicates(raw):
    rep = TextReport(echo=False)
    spaces = profile_occupancy(raw["occ"], rep)
    assert len(spaces) == 2
    assert "duplicate (space, time, status) rows: 5" in rep.text()


def test_profile_env_finds_sensors_and_stale_repeats(raw):
    rep = TextReport(echo=False)
    assert profile_env(raw["env"], rep) == {GOOD_SENSOR, DEAD_SENSOR}
    assert "| 1 |" in rep.text()                    # the dead sensor has ONE unique reading


def test_check_mapping_reports_coverage(tmp_path):
    pytest.importorskip("openpyxl")
    path = str(tmp_path / "map.xlsx")
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"Building": ["60"], "Floor/Room": ["G.38"], "Area": ["Compactus"],
                      "ID": [GOOD_SENSOR]}).to_excel(xw, sheet_name="ENV", index=False)
        pd.DataFrame({"Building": ["60"], "Floor/Room": ["G.20"], "ID": ["x"], "Note": [None],
                      "SpaceId": ["space-1"]}).to_excel(xw, sheet_name="OCC", index=False)
    rep = TextReport(echo=False)
    check_mapping(path, {GOOD_SENSOR, DEAD_SENSOR}, {"space-1", "space-2"}, rep)
    text = rep.text()
    assert "matched in xlsx: 1" in text and "G.38" in text and "G.20" in text


def test_check_mapping_missing_file_does_not_crash(tmp_path):
    rep = TextReport(echo=False)
    check_mapping(str(tmp_path / "nope.xlsx"), set(), set(), rep)
    assert "Could not find" in rep.text()


def test_report_saves_to_file(tmp_path):
    rep = TextReport(echo=False)
    rep.log("a", 1)
    rep.save(str(tmp_path / "sub" / "r.txt"))
    assert (tmp_path / "sub" / "r.txt").read_text() == "a 1"


def test_cli_full_pipeline(raw, tmp_path):
    proc = tmp_path / "proc"
    assert main(["prepare", "--occ", raw["occ"], "--env", raw["env"], "--out", str(proc)]) == 0
    assert (proc / "env_10min.csv.gz").exists() and (proc / "occupancy_10min.csv.gz").exists()
    assert main(["screen", "--dir", str(proc), "--out", str(tmp_path / "s.txt")]) == 0
    assert main(["diagnose", "--dir", str(proc), "--end", "2025-09-20", "--exclude", "bbbbbbbb",
                 "--out", str(tmp_path / "d.txt")]) == 0
    assert "aaaaaaaa" in (tmp_path / "d.txt").read_text()
    assert main(["train", "--dir", str(proc), "--sensor", "326", "--spaces", "aaaaaaaa",
                 "bbbbbbbb", "--n-boot", "50", "--out", str(tmp_path / "m.txt")]) == 0
    assert "HELD-OUT TEST" in (tmp_path / "m.txt").read_text()


def test_cli_gives_clear_error_when_data_not_prepared(tmp_path):
    with pytest.raises(SystemExit, match="prepare"):
        main(["screen", "--dir", str(tmp_path), "--out", str(tmp_path / "s.txt")])


def test_cli_train_reports_bad_sensor(raw, tmp_path):
    proc = tmp_path / "proc"
    main(["prepare", "--occ", raw["occ"], "--env", raw["env"], "--out", str(proc)])
    with pytest.raises(SystemExit, match="not found"):
        main(["train", "--dir", str(proc), "--sensor", "999", "--spaces", "aaaaaaaa",
              "--out", str(tmp_path / "m.txt")])
