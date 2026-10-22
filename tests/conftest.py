"""Shared fixtures: synthetic raw files and their cleaned versions (built once per session)."""
import pytest
from synthetic import make_raw_files

from occenv.cleaning import clean_env, clean_occupancy


@pytest.fixture(scope="session")
def raw(tmp_path_factory):
    """Paths of the synthetic raw occupancy and env CSVs."""
    return make_raw_files(str(tmp_path_factory.mktemp("raw")))


@pytest.fixture(scope="session")
def processed(raw):
    """``(env, occ)`` cleaned tables for the synthetic data."""
    env, _ = clean_env(raw["env"])
    return env, clean_occupancy(raw["occ"])
