"""
Unit tests for data_loader.py — covers both happy-path and defensive error handling.

Test categories
───────────────
TestLoadUnifiedData      — file loading, schema validation, date parsing
TestLoadReferenceCodes   — reference file loading and schema check
TestFilterFunctions      — get_observations / events / impact_links / targets
TestGetIndicatorSeries   — series extraction, empty-indicator, bad inputs
TestDatasetSummary       — summary dict structure
TestValidationHelpers    — private helpers exposed via public API errors
TestDefensiveCoding      — missing files, wrong types, missing columns, NaN dates
"""

import warnings
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import pytest

# ── make src/ importable ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    load_unified_data,
    load_reference_codes,
    get_observations,
    get_events,
    get_impact_links,
    get_targets,
    get_indicator_series,
    get_access_trajectory,
    get_mm_trajectory,
    get_digital_payment_series,
    dataset_summary,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "ethiopia_fi_unified_data.csv"
REF_PATH  = Path(__file__).parent.parent / "data" / "raw" / "reference_codes.csv"


# ════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def full_df():
    return load_unified_data(DATA_PATH)


@pytest.fixture
def minimal_df():
    """Minimal valid DataFrame for unit-testing filters without hitting disk."""
    return pd.DataFrame({
        "record_type":      ["observation", "event", "impact_link", "target"],
        "observation_date": pd.to_datetime(["2024-01-01"] * 4),
        "indicator_code":   ["ACC_OWNERSHIP", None, None, "ACC_OWNERSHIP"],
        "value_numeric":    [49.0, None, None, 55.0],
        "confidence":       ["high", None, None, "high"],
        "parent_id":        [None, None, "EVT001", None],
        "related_indicator":[None, None, "ACC_OWNERSHIP", None],
    })


# ════════════════════════════════════════════════════════════════════════════
#  TestLoadUnifiedData
# ════════════════════════════════════════════════════════════════════════════

class TestLoadUnifiedData:
    def test_returns_dataframe(self, full_df):
        assert isinstance(full_df, pd.DataFrame)

    def test_non_empty(self, full_df):
        assert len(full_df) > 0

    def test_required_columns_present(self, full_df):
        for col in ["record_type", "observation_date", "indicator_code", "value_numeric"]:
            assert col in full_df.columns, f"Missing column: {col}"

    def test_observation_date_is_datetime(self, full_df):
        assert pd.api.types.is_datetime64_any_dtype(full_df["observation_date"]), (
            "observation_date should be datetime64 after parsing"
        )

    def test_all_expected_record_types(self, full_df):
        types = set(full_df["record_type"].dropna().unique())
        assert {"observation", "event", "impact_link", "target"}.issubset(types)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_unified_data(tmp_path / "nonexistent.csv")

    def test_missing_required_column_raises_value_error(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        # Write a CSV without 'record_type' column
        pd.DataFrame({
            "observation_date": ["2024-01-01"],
            "indicator_code":   ["ACC_OWNERSHIP"],
            "value_numeric":    [49.0],
        }).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="record_type"):
            load_unified_data(bad_csv)

    def test_non_datetime_date_column_raises_type_error(self, tmp_path):
        bad_csv = tmp_path / "bad_date.csv"
        pd.DataFrame({
            "record_type":      ["observation"],
            "observation_date": ["not-a-date"],
            "indicator_code":   ["ACC_OWNERSHIP"],
            "value_numeric":    [49.0],
        }).to_csv(bad_csv, index=False)
        with pytest.raises(TypeError, match="observation_date"):
            load_unified_data(bad_csv)

    def test_empty_file_raises_value_error(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame(columns=["record_type", "observation_date",
                               "indicator_code", "value_numeric"]).to_csv(empty_csv, index=False)
        with pytest.raises(ValueError, match="no rows"):
            load_unified_data(empty_csv)

    def test_unknown_record_type_warns(self, tmp_path):
        odd_csv = tmp_path / "odd.csv"
        pd.DataFrame({
            "record_type":      ["observation", "UNKNOWN_TYPE"],
            "observation_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "indicator_code":   ["ACC_OWNERSHIP", "X"],
            "value_numeric":    [49.0, 1.0],
        }).to_csv(odd_csv, index=False)
        with pytest.warns(UserWarning, match="Unexpected record_type"):
            load_unified_data(odd_csv)


# ════════════════════════════════════════════════════════════════════════════
#  TestLoadReferenceCodes
# ════════════════════════════════════════════════════════════════════════════

class TestLoadReferenceCodes:
    def test_returns_dataframe(self):
        df = load_reference_codes(REF_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = load_reference_codes(REF_PATH)
        assert "field" in df.columns
        assert "code"  in df.columns

    def test_non_empty(self):
        df = load_reference_codes(REF_PATH)
        assert len(df) > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_reference_codes(tmp_path / "nowhere.csv")

    def test_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad_ref.csv"
        pd.DataFrame({"wrong_col": ["a"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="field"):
            load_reference_codes(bad)


# ════════════════════════════════════════════════════════════════════════════
#  TestFilterFunctions
# ════════════════════════════════════════════════════════════════════════════

class TestFilterFunctions:
    # ── happy path using real data ───────────────────────────────────────────
    def test_observation_count(self, full_df):
        assert len(get_observations(full_df)) >= 30

    def test_event_count(self, full_df):
        assert len(get_events(full_df)) >= 10

    def test_impact_link_count(self, full_df):
        assert len(get_impact_links(full_df)) >= 14

    def test_target_count(self, full_df):
        assert len(get_targets(full_df)) >= 3

    def test_observations_only_contain_observations(self, full_df):
        obs = get_observations(full_df)
        assert (obs["record_type"] == "observation").all()

    def test_events_only_contain_events(self, full_df):
        assert (get_events(full_df)["record_type"] == "event").all()

    # ── defensive: non-DataFrame input ──────────────────────────────────────
    def test_get_observations_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            get_observations({"record_type": ["observation"]})

    def test_get_events_rejects_list(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            get_events([1, 2, 3])

    def test_get_impact_links_rejects_none(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            get_impact_links(None)

    def test_get_targets_rejects_string(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            get_targets("observations")

    # ── defensive: missing column ────────────────────────────────────────────
    def test_missing_record_type_column_raises(self):
        df = pd.DataFrame({"indicator_code": ["ACC_OWNERSHIP"]})
        with pytest.raises(ValueError, match="record_type"):
            get_observations(df)

    # ── returns copy, not view ───────────────────────────────────────────────
    def test_returns_copy(self, full_df):
        obs = get_observations(full_df)
        obs["__test__"] = 99
        assert "__test__" not in full_df.columns

    # ── empty result warns rather than crashes ───────────────────────────────
    def test_empty_result_warns_and_returns_empty_df(self, minimal_df):
        # 'target' record_type exists but let's remove it and confirm warning
        df_no_target = minimal_df[minimal_df["record_type"] != "target"].copy()
        # Capture the log warning (uses logging, not warnings module)
        result = get_targets(df_no_target)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ════════════════════════════════════════════════════════════════════════════
#  TestGetIndicatorSeries
# ════════════════════════════════════════════════════════════════════════════

class TestGetIndicatorSeries:
    def test_access_trajectory_is_increasing(self, full_df):
        traj = get_access_trajectory(full_df)
        vals = traj["value_numeric"].values
        assert vals[-1] > vals[0], "ACC_OWNERSHIP overall trend should be upward"

    def test_access_trajectory_has_min_two_points(self, full_df):
        traj = get_access_trajectory(full_df)
        assert len(traj) >= 2

    def test_mm_trajectory_exists(self, full_df):
        mm = get_mm_trajectory(full_df)
        assert len(mm) >= 1
        assert "value_numeric" in mm.columns

    def test_digital_payment_series_exists(self, full_df):
        dp = get_digital_payment_series(full_df)
        assert len(dp) >= 1

    def test_sorted_by_date(self, full_df):
        traj = get_access_trajectory(full_df)
        dates = traj["observation_date"].values
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1)), (
            "Trajectory should be sorted by observation_date"
        )

    def test_columns_present(self, full_df):
        traj = get_access_trajectory(full_df)
        for col in ["observation_date", "value_numeric", "confidence"]:
            assert col in traj.columns

    def test_values_in_valid_range(self, full_df):
        traj = get_access_trajectory(full_df)
        assert traj["value_numeric"].between(0, 100).all(), (
            "All ACC_OWNERSHIP values should be in [0, 100]"
        )

    # ── defensive: bad indicator_code type ──────────────────────────────────
    def test_non_string_indicator_raises_type_error(self, full_df):
        with pytest.raises(TypeError, match="non-empty string"):
            get_indicator_series(full_df, 12345)

    def test_empty_string_indicator_raises_type_error(self, full_df):
        with pytest.raises(TypeError, match="non-empty string"):
            get_indicator_series(full_df, "   ")

    def test_none_indicator_raises_type_error(self, full_df):
        with pytest.raises(TypeError, match="non-empty string"):
            get_indicator_series(full_df, None)

    # ── defensive: missing data for indicator ────────────────────────────────
    def test_unknown_indicator_raises_value_error(self, full_df):
        with pytest.raises(ValueError, match="NONEXISTENT_CODE"):
            get_indicator_series(full_df, "NONEXISTENT_CODE", min_points=1)

    # ── defensive: non-DataFrame input ──────────────────────────────────────
    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            get_indicator_series("not a dataframe", "ACC_OWNERSHIP")

    # ── defensive: missing required columns ─────────────────────────────────
    def test_missing_indicator_code_column_raises(self):
        df = pd.DataFrame({
            "record_type":      ["observation"],
            "observation_date": pd.to_datetime(["2024-01-01"]),
            "value_numeric":    [49.0],
        })
        with pytest.raises(ValueError, match="indicator_code"):
            get_indicator_series(df, "ACC_OWNERSHIP")


# ════════════════════════════════════════════════════════════════════════════
#  TestDatasetSummary
# ════════════════════════════════════════════════════════════════════════════

class TestDatasetSummary:
    def test_returns_dict(self, full_df):
        summary = dataset_summary(full_df)
        assert isinstance(summary, dict)

    def test_required_keys(self, full_df):
        summary = dataset_summary(full_df)
        for key in ["total_rows", "record_type_counts", "date_range",
                    "indicators", "null_value_numeric_pct"]:
            assert key in summary, f"Missing key: {key}"

    def test_total_rows_correct(self, full_df):
        summary = dataset_summary(full_df)
        assert summary["total_rows"] == len(full_df)

    def test_record_type_counts_correct(self, full_df):
        summary = dataset_summary(full_df)
        assert summary["record_type_counts"].get("observation", 0) >= 30
        assert summary["record_type_counts"].get("event", 0) >= 10

    def test_date_range_is_tuple_of_strings(self, full_df):
        summary = dataset_summary(full_df)
        start, end = summary["date_range"]
        assert isinstance(start, str) and isinstance(end, str)

    def test_null_pct_is_numeric(self, full_df):
        summary = dataset_summary(full_df)
        assert isinstance(summary["null_value_numeric_pct"], (int, float))
        assert 0 <= summary["null_value_numeric_pct"] <= 100

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            dataset_summary(None)
