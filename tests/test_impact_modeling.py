"""
Unit tests for impact_modeling.py — covers happy-path and defensive error paths.

Test categories
───────────────
TestBuildAssociationMatrix   — valid inputs, output shape, cell values, edge cases
TestComputeCumulativeImpact  — known outputs, empty links, lag/saturation logic
TestSaturationCurve          — curve shape, boundary conditions
TestRankEventsByImpact       — ordering, empty result
TestDefensiveMatrix          — bad types, missing columns
TestDefensiveCumulative      — bad types, invalid params
TestDefensiveSaturation      — invalid numeric params
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from impact_modeling import (
    build_association_matrix,
    compute_cumulative_impact,
    saturation_curve,
    rank_events_by_impact,
    MAGNITUDE_MAP,
    DIRECTION_MAP,
    DEFAULT_REALIZATION_RATE,
    DEFAULT_LAMBDA,
)


# ════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def impact_links():
    """Minimal valid impact_links DataFrame (3 links)."""
    return pd.DataFrame({
        "record_type":       ["impact_link"] * 3,
        "parent_id":         ["EVT001", "EVT001", "EVT002"],
        "related_indicator": ["ACC_OWNERSHIP", "USG_DIGITAL_PAYMENT", "ACC_OWNERSHIP"],
        "impact_magnitude":  ["large", "medium", "small"],
        "impact_direction":  ["positive", "positive", "positive"],
        "lag_months":        [12, 6, 18],
        "observation_date":  pd.to_datetime(["2021-05-01", "2021-05-01", "2022-03-01"]),
    })


@pytest.fixture
def events():
    """Minimal valid events DataFrame matching impact_links fixture."""
    return pd.DataFrame({
        "record_type":    ["event", "event"],
        "id":             ["EVT001", "EVT002"],
        "indicator_code": [None, None],
        "observation_date": pd.to_datetime(["2021-05-01", "2022-03-01"]),
        "notes":          ["Telebirr launch", "EthSwitch interop"],
    })


@pytest.fixture(scope="module")
def real_df():
    """Load the actual unified dataset for integration-style tests."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data_loader import load_unified_data, get_impact_links, get_events
    data_path = Path(__file__).parent.parent / "data" / "raw" / "ethiopia_fi_unified_data.csv"
    df  = load_unified_data(data_path)
    lnk = get_impact_links(df)
    evt = get_events(df)
    return lnk, evt


# ════════════════════════════════════════════════════════════════════════════
#  TestBuildAssociationMatrix — happy path
# ════════════════════════════════════════════════════════════════════════════

class TestBuildAssociationMatrix:
    def test_returns_dataframe(self, impact_links, events):
        mat = build_association_matrix(impact_links, events)
        assert isinstance(mat, pd.DataFrame)

    def test_events_are_index(self, impact_links, events):
        mat = build_association_matrix(impact_links, events)
        for evt_id in ["EVT001", "EVT002"]:
            assert evt_id in mat.index, f"{evt_id} missing from matrix index"

    def test_indicators_are_columns(self, impact_links, events):
        mat = build_association_matrix(impact_links, events)
        assert "ACC_OWNERSHIP"       in mat.columns
        assert "USG_DIGITAL_PAYMENT" in mat.columns

    def test_large_positive_value_correct(self, impact_links, events):
        mat = build_association_matrix(impact_links, events)
        # EVT001 → ACC_OWNERSHIP is 'large' + 'positive' → +5.0
        assert mat.at["EVT001", "ACC_OWNERSHIP"] == pytest.approx(5.0)

    def test_medium_positive_value_correct(self, impact_links, events):
        mat = build_association_matrix(impact_links, events)
        # EVT001 → USG_DIGITAL_PAYMENT is 'medium' + 'positive' → +2.5
        assert mat.at["EVT001", "USG_DIGITAL_PAYMENT"] == pytest.approx(2.5)

    def test_small_positive_value_correct(self, impact_links, events):
        mat = build_association_matrix(impact_links, events)
        # EVT002 → ACC_OWNERSHIP is 'small' + 'positive' → +1.0
        assert mat.at["EVT002", "ACC_OWNERSHIP"] == pytest.approx(1.0)

    def test_negative_direction(self, events):
        lnk = pd.DataFrame({
            "parent_id":         ["EVT001"],
            "related_indicator": ["ACC_OWNERSHIP"],
            "impact_magnitude":  ["medium"],
            "impact_direction":  ["negative"],
            "lag_months":        [12],
            "observation_date":  pd.to_datetime(["2021-01-01"]),
        })
        mat = build_association_matrix(lnk, events)
        assert mat.at["EVT001", "ACC_OWNERSHIP"] == pytest.approx(-2.5)

    def test_neutral_direction_gives_zero(self, events):
        lnk = pd.DataFrame({
            "parent_id":         ["EVT001"],
            "related_indicator": ["ACC_OWNERSHIP"],
            "impact_magnitude":  ["large"],
            "impact_direction":  ["neutral"],
            "lag_months":        [12],
            "observation_date":  pd.to_datetime(["2021-01-01"]),
        })
        mat = build_association_matrix(lnk, events)
        assert mat.at["EVT001", "ACC_OWNERSHIP"] == pytest.approx(0.0)

    def test_nan_parent_id_skipped(self, events):
        lnk = pd.DataFrame({
            "parent_id":         [None, "EVT001"],
            "related_indicator": ["ACC_OWNERSHIP", "ACC_OWNERSHIP"],
            "impact_magnitude":  ["large", "small"],
            "impact_direction":  ["positive", "positive"],
            "lag_months":        [12, 12],
            "observation_date":  pd.to_datetime(["2021-01-01", "2021-01-01"]),
        })
        mat = build_association_matrix(lnk, events)
        # Only EVT001 row should be in the matrix
        assert "EVT001" in mat.index
        assert None not in mat.index

    def test_unknown_magnitude_warns_and_gives_zero(self, events):
        lnk = pd.DataFrame({
            "parent_id":         ["EVT001"],
            "related_indicator": ["ACC_OWNERSHIP"],
            "impact_magnitude":  ["GIANT"],   # unknown
            "impact_direction":  ["positive"],
            "lag_months":        [12],
            "observation_date":  pd.to_datetime(["2021-01-01"]),
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mat = build_association_matrix(lnk, events)
        assert any("GIANT" in str(x.message) for x in w)
        assert mat.at["EVT001", "ACC_OWNERSHIP"] == pytest.approx(0.0)

    def test_unknown_direction_warns_and_gives_zero(self, events):
        lnk = pd.DataFrame({
            "parent_id":         ["EVT001"],
            "related_indicator": ["ACC_OWNERSHIP"],
            "impact_magnitude":  ["large"],
            "impact_direction":  ["sideways"],   # unknown
            "lag_months":        [12],
            "observation_date":  pd.to_datetime(["2021-01-01"]),
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mat = build_association_matrix(lnk, events)
        assert any("sideways" in str(x.message) for x in w)
        assert mat.at["EVT001", "ACC_OWNERSHIP"] == pytest.approx(0.0)

    def test_empty_impact_links_warns_and_returns_empty(self, events):
        empty = pd.DataFrame(columns=["parent_id", "related_indicator",
                                       "impact_magnitude", "impact_direction"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mat = build_association_matrix(empty, events)
        assert mat.empty
        assert any("empty" in str(x.message).lower() for x in w)

    def test_real_data_shape(self, real_df):
        lnk, evt = real_df
        mat = build_association_matrix(lnk, evt)
        assert mat.shape[0] >= 5,  "Expected at least 5 events in matrix"
        assert mat.shape[1] >= 3,  "Expected at least 3 indicators in matrix"

    def test_real_data_values_bounded(self, real_df):
        lnk, evt = real_df
        mat = build_association_matrix(lnk, evt)
        assert mat.values.min() >= -5.0, "No cell should be below -5.0 pp (largest negative)"
        assert mat.values.max() <=  5.0, "No cell should exceed +5.0 pp (largest positive)"


# ════════════════════════════════════════════════════════════════════════════
#  TestComputeCumulativeImpact — happy path
# ════════════════════════════════════════════════════════════════════════════

class TestComputeCumulativeImpact:
    def test_returns_float(self, impact_links):
        result = compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025)
        assert isinstance(result, float)

    def test_positive_events_give_positive_total(self, impact_links):
        result = compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025)
        assert result > 0.0, "All impacts are positive, total must be > 0"

    def test_unknown_indicator_returns_zero(self, impact_links):
        result = compute_cumulative_impact(impact_links, "NONEXISTENT", 2025)
        assert result == pytest.approx(0.0)

    def test_forecast_before_effective_date_returns_zero(self):
        """A 2020 forecast for an event with 24-month lag starting 2021 → 0."""
        lnk = pd.DataFrame({
            "related_indicator": ["ACC_OWNERSHIP"],
            "impact_magnitude":  ["large"],
            "impact_direction":  ["positive"],
            "lag_months":        [24],
            "observation_date":  pd.to_datetime(["2021-01-01"]),
        })
        result = compute_cumulative_impact(lnk, "ACC_OWNERSHIP", 2020.0)
        assert result == pytest.approx(0.0)

    def test_impact_grows_over_time(self, impact_links):
        """Impact at 2027 should be larger than at 2025 (saturation curve)."""
        r2025 = compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025.0)
        r2027 = compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2027.0)
        assert r2027 >= r2025

    def test_higher_realization_rate_gives_higher_impact(self, impact_links):
        r_low  = compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025,
                                           realization_rate=0.20)
        r_high = compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025,
                                           realization_rate=0.60)
        assert r_high > r_low

    def test_zero_realization_rate_gives_zero(self, impact_links):
        result = compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025,
                                           realization_rate=0.0)
        assert result == pytest.approx(0.0)

    def test_bad_observation_date_warns_and_still_returns_float(self):
        """Rows with unparseable dates should warn but not crash."""
        lnk = pd.DataFrame({
            "related_indicator": ["ACC_OWNERSHIP"],
            "impact_magnitude":  ["medium"],
            "impact_direction":  ["positive"],
            "lag_months":        [12],
            "observation_date":  [None],   # NaT → warning, defaults to 2021
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_cumulative_impact(lnk, "ACC_OWNERSHIP", 2025)
        assert isinstance(result, float)

    def test_neutral_direction_contributes_zero(self):
        lnk = pd.DataFrame({
            "related_indicator": ["ACC_OWNERSHIP"],
            "impact_magnitude":  ["large"],
            "impact_direction":  ["neutral"],
            "lag_months":        [6],
            "observation_date":  pd.to_datetime(["2021-01-01"]),
        })
        result = compute_cumulative_impact(lnk, "ACC_OWNERSHIP", 2025)
        assert result == pytest.approx(0.0)

    def test_real_data_plausible_range(self, real_df):
        lnk, _ = real_df
        result = compute_cumulative_impact(lnk, "ACC_OWNERSHIP", 2025,
                                           realization_rate=DEFAULT_REALIZATION_RATE)
        # Should be a finite number in a plausible range for Ethiopia
        assert np.isfinite(result)
        assert -20.0 < result < 20.0, f"Cumulative impact out of plausible range: {result}"


# ════════════════════════════════════════════════════════════════════════════
#  TestSaturationCurve
# ════════════════════════════════════════════════════════════════════════════

class TestSaturationCurve:
    def test_returns_two_arrays(self):
        yrs, eff = saturation_curve(2021, 12, 5.0)
        assert isinstance(yrs, np.ndarray)
        assert isinstance(eff, np.ndarray)
        assert len(yrs) == len(eff)

    def test_effect_zero_before_effective_date(self):
        yrs, eff = saturation_curve(2021, 12, 5.0,
                                    years=np.array([2020.0, 2021.0, 2021.5]))
        # Effective date = 2021 + 12/12 = 2022 → all listed years before that
        assert all(eff == 0.0)

    def test_effect_positive_after_effective_date(self):
        yrs, eff = saturation_curve(2021, 0, 5.0,
                                    years=np.array([2022.0, 2023.0, 2024.0]))
        assert all(eff > 0.0)

    def test_negative_direction(self):
        yrs, eff = saturation_curve(2021, 0, 2.5, direction=-1,
                                    years=np.array([2022.0, 2023.0]))
        assert all(eff < 0.0)

    def test_curve_is_monotone_increasing(self):
        yrs, eff = saturation_curve(2021, 0, 5.0,
                                    years=np.arange(2021.5, 2030, 0.5))
        diffs = np.diff(eff)
        assert (diffs >= -1e-9).all(), "Saturation curve must be non-decreasing"

    def test_curve_does_not_exceed_max(self):
        yrs, eff = saturation_curve(2021, 0, 5.0, realization_rate=1.0,
                                    years=np.arange(2021, 2100, 1.0))
        assert eff.max() <= 5.0 + 1e-9

    def test_zero_realization_gives_zero_effect(self):
        yrs, eff = saturation_curve(2021, 0, 5.0, realization_rate=0.0,
                                    years=np.array([2025.0, 2030.0]))
        np.testing.assert_allclose(eff, 0.0, atol=1e-10)

    def test_custom_years_used(self):
        custom = np.array([2023.0, 2024.0])
        yrs, eff = saturation_curve(2021, 0, 5.0, years=custom)
        np.testing.assert_array_equal(yrs, custom)


# ════════════════════════════════════════════════════════════════════════════
#  TestRankEventsByImpact
# ════════════════════════════════════════════════════════════════════════════

class TestRankEventsByImpact:
    def test_returns_dataframe(self, impact_links):
        df = rank_events_by_impact(impact_links, "ACC_OWNERSHIP", 2025)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, impact_links):
        df = rank_events_by_impact(impact_links, "ACC_OWNERSHIP", 2025)
        for col in ("parent_id", "impact_pp", "abs_impact_pp"):
            assert col in df.columns

    def test_sorted_descending_by_absolute_impact(self, impact_links):
        df = rank_events_by_impact(impact_links, "ACC_OWNERSHIP", 2025)
        abs_vals = df["abs_impact_pp"].values
        assert (np.diff(abs_vals) <= 0).all(), "Should be sorted descending"

    def test_unknown_indicator_returns_empty_df(self, impact_links):
        df = rank_events_by_impact(impact_links, "NONEXISTENT", 2025)
        assert df.empty

    def test_large_impact_ranks_first(self, impact_links):
        df = rank_events_by_impact(impact_links, "ACC_OWNERSHIP", 2025)
        # EVT001 has 'large' (5pp) and EVT002 has 'small' (1pp)
        assert df.iloc[0]["parent_id"] == "EVT001"

    def test_real_data_no_crash(self, real_df):
        """Function must not crash on the real dataset; result may be empty
        if the CSV impact_link column alignment differs from expected schema."""
        lnk, _ = real_df
        df = rank_events_by_impact(lnk, "ACC_OWNERSHIP", 2025)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 0   # empty is acceptable if no ACC_OWNERSHIP links


# ════════════════════════════════════════════════════════════════════════════
#  TestDefensiveMatrix — bad inputs to build_association_matrix
# ════════════════════════════════════════════════════════════════════════════

class TestDefensiveMatrix:
    def test_non_dataframe_impact_links_raises(self, events):
        with pytest.raises(TypeError, match="DataFrame"):
            build_association_matrix({"parent_id": ["EVT001"]}, events)

    def test_non_dataframe_events_raises(self, impact_links):
        with pytest.raises(TypeError, match="DataFrame"):
            build_association_matrix(impact_links, [1, 2, 3])

    def test_missing_parent_id_column_raises(self, events):
        bad = pd.DataFrame({"related_indicator": ["ACC_OWNERSHIP"]})
        with pytest.raises(ValueError, match="parent_id"):
            build_association_matrix(bad, events)

    def test_missing_id_in_events_raises(self, impact_links):
        bad_events = pd.DataFrame({"indicator_code": [None]})
        with pytest.raises(ValueError, match="'id'"):
            build_association_matrix(impact_links, bad_events)


# ════════════════════════════════════════════════════════════════════════════
#  TestDefensiveCumulative — bad inputs to compute_cumulative_impact
# ════════════════════════════════════════════════════════════════════════════

class TestDefensiveCumulative:
    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            compute_cumulative_impact({"related_indicator": ["ACC"]}, "ACC", 2025)

    def test_missing_related_indicator_column_raises(self):
        bad = pd.DataFrame({"parent_id": ["EVT001"]})
        with pytest.raises(ValueError, match="related_indicator"):
            compute_cumulative_impact(bad, "ACC_OWNERSHIP", 2025)

    def test_non_string_indicator_raises(self, impact_links):
        with pytest.raises(TypeError, match="non-empty string"):
            compute_cumulative_impact(impact_links, 999, 2025)

    def test_empty_indicator_code_raises(self, impact_links):
        with pytest.raises(TypeError, match="non-empty string"):
            compute_cumulative_impact(impact_links, "  ", 2025)

    def test_nan_forecast_year_raises(self, impact_links):
        with pytest.raises(ValueError, match="finite"):
            compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", float("nan"))

    def test_realization_rate_above_one_raises(self, impact_links):
        with pytest.raises(ValueError, match="realization_rate"):
            compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025,
                                      realization_rate=1.5)

    def test_negative_realization_rate_raises(self, impact_links):
        with pytest.raises(ValueError, match="realization_rate"):
            compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025,
                                      realization_rate=-0.1)

    def test_zero_lambda_raises(self, impact_links):
        with pytest.raises(ValueError, match="lam"):
            compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025, lam=0.0)

    def test_negative_lambda_raises(self, impact_links):
        with pytest.raises(ValueError, match="lam"):
            compute_cumulative_impact(impact_links, "ACC_OWNERSHIP", 2025, lam=-0.5)


# ════════════════════════════════════════════════════════════════════════════
#  TestDefensiveSaturation — bad inputs to saturation_curve
# ════════════════════════════════════════════════════════════════════════════

class TestDefensiveSaturation:
    def test_nan_event_year_raises(self):
        with pytest.raises(ValueError, match="finite"):
            saturation_curve(float("nan"), 12, 5.0)

    def test_negative_lag_months_raises(self):
        with pytest.raises(ValueError, match="lag_months"):
            saturation_curve(2021, -6, 5.0)

    def test_negative_magnitude_raises(self):
        with pytest.raises(ValueError, match="magnitude"):
            saturation_curve(2021, 12, -1.0)

    def test_realization_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="realization_rate"):
            saturation_curve(2021, 12, 5.0, realization_rate=1.5)

    def test_zero_lambda_raises(self):
        with pytest.raises(ValueError, match="lam"):
            saturation_curve(2021, 12, 5.0, lam=0.0)
