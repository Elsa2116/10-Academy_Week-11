"""
Unit tests for data loading utilities.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    load_unified_data,
    load_reference_codes,
    get_observations,
    get_events,
    get_impact_links,
    get_targets,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "ethiopia_fi_unified_data.csv"
REF_PATH = Path(__file__).parent.parent / "data" / "raw" / "reference_codes.csv"


class TestDataLoader:
    def test_load_unified_data(self):
        df = load_unified_data(DATA_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "record_type" in df.columns

    def test_load_reference_codes(self):
        df = load_reference_codes(REF_PATH)
        assert isinstance(df, pd.DataFrame)
        assert "field" in df.columns
        assert "code" in df.columns

    def test_record_types_present(self):
        df = load_unified_data(DATA_PATH)
        types = df["record_type"].unique()
        assert "observation" in types
        assert "event" in types
        assert "impact_link" in types
        assert "target" in types

    def test_observation_count(self):
        df = load_unified_data(DATA_PATH)
        obs = get_observations(df)
        assert len(obs) >= 30

    def test_event_count(self):
        df = load_unified_data(DATA_PATH)
        evts = get_events(df)
        assert len(evts) >= 10

    def test_impact_link_count(self):
        df = load_unified_data(DATA_PATH)
        links = get_impact_links(df)
        assert len(links) >= 14

    def test_target_count(self):
        df = load_unified_data(DATA_PATH)
        targets = get_targets(df)
        assert len(targets) >= 3

    def test_access_trajectory_increasing(self):
        df = load_unified_data(DATA_PATH)
        from data_loader import get_access_trajectory
        traj = get_access_trajectory(df)
        values = traj["value_numeric"].values
        assert values[-1] > values[0]  # Overall upward trend
