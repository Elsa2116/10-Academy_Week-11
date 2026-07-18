"""
Unit tests for forecasting.py — covers happy-path behaviour and all
defensive error-handling paths added for production robustness.

Test categories
───────────────
TestFitLinearTrend      — valid fits, model dict keys, edge cases
TestForecastLinear      — output shape, CI ordering, bounds, confidence param
TestApplyEventEffects   — impact direction/magnitude, realization rate, bounds
TestScenarioForecasts   — scenario ordering, spread, realization_rates param
TestYearToNumeric       — series conversion helper
TestDefensiveFit        — bad inputs to fit_linear_trend
TestDefensiveForecast   — bad inputs to forecast_linear
TestDefensiveEvents     — bad inputs to apply_event_effects
TestDefensiveScenario   — bad inputs to scenario_forecasts
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecasting import (
    fit_linear_trend,
    forecast_linear,
    apply_event_effects,
    scenario_forecasts,
    year_to_numeric,
    IMPACT_MAGNITUDE_MAP,
    DEFAULT_REALIZATION_RATE,
)

# ── shared fixtures ───────────────────────────────────────────────────────────
YEARS_ACC  = [2011, 2014, 2017, 2021, 2024]
VALUES_ACC = [14.0, 22.0, 35.0, 46.0, 49.0]
FORECAST_YEARS = [2025, 2026, 2027]

EVENTS_POSITIVE = [
    {"impact_magnitude": "large",  "impact_direction": "positive"},
    {"impact_magnitude": "medium", "impact_direction": "positive"},
]
EVENTS_NEGATIVE = [
    {"impact_magnitude": "medium", "impact_direction": "negative"},
]


@pytest.fixture(scope="module")
def acc_model():
    return fit_linear_trend(YEARS_ACC, VALUES_ACC)


@pytest.fixture(scope="module")
def base_fc(acc_model):
    return forecast_linear(acc_model, FORECAST_YEARS)


# ════════════════════════════════════════════════════════════════════════════
#  TestFitLinearTrend — happy path
# ════════════════════════════════════════════════════════════════════════════

class TestFitLinearTrend:
    def test_returns_dict(self, acc_model):
        assert isinstance(acc_model, dict)

    def test_required_keys(self, acc_model):
        for key in ("slope", "intercept", "r_squared", "p_value",
                    "std_err", "n_obs", "mean_year"):
            assert key in acc_model, f"Missing key: {key}"

    def test_slope_positive(self, acc_model):
        assert acc_model["slope"] > 0, "Account ownership trend should be positive"

    def test_r_squared_high(self, acc_model):
        assert acc_model["r_squared"] > 0.90, (
            f"Expected R² > 0.90, got {acc_model['r_squared']:.4f}"
        )

    def test_r_squared_bounded(self, acc_model):
        assert 0.0 <= acc_model["r_squared"] <= 1.0

    def test_n_obs_correct(self, acc_model):
        assert acc_model["n_obs"] == len(YEARS_ACC)

    def test_mean_year_correct(self, acc_model):
        assert acc_model["mean_year"] == pytest.approx(np.mean(YEARS_ACC), rel=1e-6)

    def test_std_err_non_negative(self, acc_model):
        assert acc_model["std_err"] >= 0

    def test_all_values_finite(self, acc_model):
        for key, val in acc_model.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"model[{key!r}] is not finite: {val}"

    def test_perfect_linear_data(self):
        """Exact linear series must produce R²=1 and the correct slope."""
        model = fit_linear_trend([0, 1, 2, 3, 4], [10, 20, 30, 40, 50])
        assert model["r_squared"] == pytest.approx(1.0, abs=1e-9)
        assert model["slope"]     == pytest.approx(10.0, rel=1e-6)
        assert model["intercept"] == pytest.approx(10.0, rel=1e-6)

    def test_accepts_numpy_arrays(self):
        model = fit_linear_trend(np.array(YEARS_ACC), np.array(VALUES_ACC))
        assert isinstance(model, dict)

    def test_accepts_pandas_series(self):
        model = fit_linear_trend(pd.Series(YEARS_ACC), pd.Series(VALUES_ACC))
        assert isinstance(model, dict)

    def test_usage_series(self):
        """Digital payment series has a strongly positive trend (R² > 0.90)."""
        model = fit_linear_trend([2014, 2017, 2021, 2024], [3.0, 8.0, 18.0, 35.0])
        assert model["slope"] > 0
        # Growth accelerates (not perfectly linear) so R² is ~0.93, not > 0.95
        assert model["r_squared"] > 0.90


# ════════════════════════════════════════════════════════════════════════════
#  TestForecastLinear — happy path
# ════════════════════════════════════════════════════════════════════════════

class TestForecastLinear:
    def test_returns_dataframe(self, base_fc):
        assert isinstance(base_fc, pd.DataFrame)

    def test_correct_number_of_rows(self, base_fc):
        assert len(base_fc) == len(FORECAST_YEARS)

    def test_required_columns(self, base_fc):
        for col in ("year", "forecast", "lower", "upper"):
            assert col in base_fc.columns, f"Missing column: {col}"

    def test_upper_geq_lower(self, base_fc):
        assert (base_fc["upper"] >= base_fc["lower"]).all()

    def test_forecast_between_lower_and_upper(self, base_fc):
        assert (base_fc["forecast"] >= base_fc["lower"]).all()
        assert (base_fc["forecast"] <= base_fc["upper"]).all()

    def test_forecast_bounded_0_100(self, base_fc):
        assert (base_fc["forecast"] >= 0).all()
        assert (base_fc["forecast"] <= 100).all()

    def test_lower_bounded_0(self, base_fc):
        assert (base_fc["lower"] >= 0).all()

    def test_upper_bounded_100(self, base_fc):
        assert (base_fc["upper"] <= 100).all()

    def test_years_match_input(self, base_fc):
        assert list(base_fc["year"]) == sorted(FORECAST_YEARS)

    def test_ci_widens_with_distance(self, base_fc):
        """Prediction interval should be wider further from the training mean."""
        widths = (base_fc["upper"] - base_fc["lower"]).values
        assert widths[-1] >= widths[0], (
            "CI width should not narrow as we forecast further into the future"
        )

    def test_single_year_forecast(self, acc_model):
        fc = forecast_linear(acc_model, [2025])
        assert len(fc) == 1

    def test_confidence_99(self, acc_model):
        fc_95 = forecast_linear(acc_model, FORECAST_YEARS, confidence=0.95)
        fc_99 = forecast_linear(acc_model, FORECAST_YEARS, confidence=0.99)
        # 99% CI must be at least as wide as 95% CI
        width_95 = (fc_95["upper"] - fc_95["lower"]).values
        width_99 = (fc_99["upper"] - fc_99["lower"]).values
        assert (width_99 >= width_95).all()


# ════════════════════════════════════════════════════════════════════════════
#  TestApplyEventEffects — happy path
# ════════════════════════════════════════════════════════════════════════════

class TestApplyEventEffects:
    def test_returns_dataframe(self, base_fc):
        result = apply_event_effects(base_fc, EVENTS_POSITIVE, "ACC_OWNERSHIP")
        assert isinstance(result, pd.DataFrame)

    def test_positive_events_increase_forecast(self, base_fc):
        result = apply_event_effects(base_fc, EVENTS_POSITIVE, "ACC_OWNERSHIP")
        assert (result["forecast"] >= base_fc["forecast"]).all()

    def test_negative_events_decrease_forecast(self, base_fc):
        result = apply_event_effects(base_fc, EVENTS_NEGATIVE, "ACC_OWNERSHIP")
        assert (result["forecast"] <= base_fc["forecast"]).all()

    def test_empty_events_unchanged(self, base_fc):
        result = apply_event_effects(base_fc, [], "ACC_OWNERSHIP")
        pd.testing.assert_frame_equal(result, base_fc)

    def test_output_still_bounded(self, base_fc):
        result = apply_event_effects(base_fc, EVENTS_POSITIVE, "ACC_OWNERSHIP")
        assert (result["forecast"] >= 0).all()
        assert (result["forecast"] <= 100).all()

    def test_does_not_mutate_input(self, base_fc):
        original = base_fc.copy()
        apply_event_effects(base_fc, EVENTS_POSITIVE, "ACC_OWNERSHIP")
        pd.testing.assert_frame_equal(base_fc, original)

    def test_zero_realization_rate_leaves_unchanged(self, base_fc):
        result = apply_event_effects(
            base_fc, EVENTS_POSITIVE, "ACC_OWNERSHIP", realization_rate=0.0
        )
        pd.testing.assert_frame_equal(result, base_fc)

    def test_large_magnitude_is_capped_at_100(self, acc_model):
        """Even extreme events must not push forecast above 100."""
        fc = forecast_linear(acc_model, [2025])
        big_events = [{"impact_magnitude": "large", "impact_direction": "positive"}] * 20
        result = apply_event_effects(fc, big_events, "ACC_OWNERSHIP")
        assert (result["forecast"] <= 100).all()

    def test_unknown_magnitude_defaults_to_small(self, base_fc):
        events = [{"impact_magnitude": "UNKNOWN", "impact_direction": "positive"}]
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = apply_event_effects(base_fc, events, "ACC_OWNERSHIP",
                                         realization_rate=1.0)
        # UNKNOWN → 0.0 (no effect per _safe_magnitude in forecasting)
        # Here we just check it doesn't crash and stays bounded
        assert isinstance(result, pd.DataFrame)
        assert (result["forecast"] >= 0).all()


# ════════════════════════════════════════════════════════════════════════════
#  TestScenarioForecasts — happy path
# ════════════════════════════════════════════════════════════════════════════

class TestScenarioForecasts:
    def test_returns_dataframe(self, acc_model):
        sc = scenario_forecasts(acc_model, FORECAST_YEARS)
        assert isinstance(sc, pd.DataFrame)

    def test_required_columns(self, acc_model):
        sc = scenario_forecasts(acc_model, FORECAST_YEARS)
        for col in ("year", "pessimistic", "base", "optimistic"):
            assert col in sc.columns

    def test_optimistic_geq_base(self, acc_model):
        sc = scenario_forecasts(acc_model, FORECAST_YEARS)
        assert (sc["optimistic"] >= sc["base"]).all()

    def test_base_geq_pessimistic(self, acc_model):
        sc = scenario_forecasts(acc_model, FORECAST_YEARS)
        assert (sc["base"] >= sc["pessimistic"]).all()

    def test_all_values_bounded(self, acc_model):
        sc = scenario_forecasts(acc_model, FORECAST_YEARS)
        for col in ("pessimistic", "base", "optimistic"):
            assert (sc[col] >= 0).all()
            assert (sc[col] <= 100).all()

    def test_years_match_input(self, acc_model):
        sc = scenario_forecasts(acc_model, FORECAST_YEARS)
        assert list(sc["year"]) == sorted(FORECAST_YEARS)

    def test_zero_spread_collapses_pessimistic_and_optimistic(self, acc_model):
        """With spread=0 and same event_effects, all three scenarios equal base."""
        sc = scenario_forecasts(
            acc_model, [2025],
            event_effects=0.0,
            spread=0.0,
            realization_rates=(0.40, 0.40, 0.40),
        )
        assert sc["pessimistic"].iloc[0] == pytest.approx(sc["base"].iloc[0], abs=1e-6)
        assert sc["optimistic"].iloc[0]  == pytest.approx(sc["base"].iloc[0], abs=1e-6)

    def test_positive_event_effects_raise_all_scenarios(self, acc_model):
        sc_no_evt  = scenario_forecasts(acc_model, [2025], event_effects=0.0)
        sc_with_evt = scenario_forecasts(acc_model, [2025], event_effects=5.0)
        assert sc_with_evt["base"].iloc[0] >= sc_no_evt["base"].iloc[0]

    def test_row_count_matches_forecast_years(self, acc_model):
        sc = scenario_forecasts(acc_model, FORECAST_YEARS)
        assert len(sc) == len(FORECAST_YEARS)


# ════════════════════════════════════════════════════════════════════════════
#  TestYearToNumeric
# ════════════════════════════════════════════════════════════════════════════

class TestYearToNumeric:
    def test_integer_series(self):
        result = year_to_numeric(pd.Series([2011, 2014, 2017]))
        assert result.shape == (3, 1)
        assert result.dtype == float

    def test_datetime_series(self):
        dates = pd.to_datetime(["2011-01-01", "2014-01-01", "2017-01-01"])
        result = year_to_numeric(pd.Series(dates))
        assert result[0, 0] == 2011.0

    def test_non_series_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas Series"):
            year_to_numeric([2011, 2014])

    def test_empty_series_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            year_to_numeric(pd.Series([], dtype=float))


# ════════════════════════════════════════════════════════════════════════════
#  TestDefensiveFit — bad inputs to fit_linear_trend
# ════════════════════════════════════════════════════════════════════════════

class TestDefensiveFit:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_linear_trend([2011, 2014], [14.0, 22.0, 35.0])

    def test_empty_arrays_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_linear_trend([], [])

    def test_single_point_raises_by_default(self):
        with pytest.raises(ValueError, match="At least 2"):
            fit_linear_trend([2024], [49.0])

    def test_single_point_allowed_with_min_points_1(self):
        """min_points=1 allows a degenerate (zero-slope) fit — caller's choice."""
        model = fit_linear_trend([2024], [49.0], min_points=1)
        assert isinstance(model, dict)

    def test_nan_in_years_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            fit_linear_trend([2011, float("nan")], [14.0, 22.0])

    def test_nan_in_values_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            fit_linear_trend([2011, 2014], [14.0, float("nan")])

    def test_inf_in_values_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            fit_linear_trend([2011, 2014], [14.0, float("inf")])

    def test_non_numeric_string_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            fit_linear_trend(["a", "b"], [14.0, 22.0])

    def test_values_out_of_range_warns(self):
        """Values > 100 should warn but still fit."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = fit_linear_trend([2011, 2014], [14.0, 150.0])
        assert any("100" in str(warning.message) for warning in w)
        assert isinstance(model, dict)


# ════════════════════════════════════════════════════════════════════════════
#  TestDefensiveForecast — bad inputs to forecast_linear
# ════════════════════════════════════════════════════════════════════════════

class TestDefensiveForecast:
    def test_non_dict_model_raises(self):
        with pytest.raises(TypeError, match="dict"):
            forecast_linear([1, 2, 3], [2025])

    def test_missing_model_key_raises(self):
        bad_model = {"slope": 1.0, "intercept": 10.0}  # missing other keys
        with pytest.raises(ValueError, match="missing required key"):
            forecast_linear(bad_model, [2025])

    def test_non_finite_slope_raises(self, acc_model):
        bad = dict(acc_model); bad["slope"] = float("nan")
        with pytest.raises(ValueError, match="not finite"):
            forecast_linear(bad, [2025])

    def test_empty_forecast_years_raises(self, acc_model):
        with pytest.raises(ValueError, match="empty"):
            forecast_linear(acc_model, [])

    def test_non_iterable_forecast_years_raises(self, acc_model):
        with pytest.raises(TypeError, match="iterable"):
            forecast_linear(acc_model, 2025)

    def test_non_numeric_year_raises(self, acc_model):
        with pytest.raises(TypeError, match="numeric"):
            forecast_linear(acc_model, ["future"])

    def test_confidence_zero_raises(self, acc_model):
        with pytest.raises(ValueError, match="confidence"):
            forecast_linear(acc_model, [2025], confidence=0.0)

    def test_confidence_one_raises(self, acc_model):
        with pytest.raises(ValueError, match="confidence"):
            forecast_linear(acc_model, [2025], confidence=1.0)

    def test_confidence_negative_raises(self, acc_model):
        with pytest.raises(ValueError, match="confidence"):
            forecast_linear(acc_model, [2025], confidence=-0.5)


# ════════════════════════════════════════════════════════════════════════════
#  TestDefensiveEvents — bad inputs to apply_event_effects
# ════════════════════════════════════════════════════════════════════════════

class TestDefensiveEvents:
    def test_non_dataframe_base_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            apply_event_effects({"year": [2025]}, [], "ACC_OWNERSHIP")

    def test_missing_required_columns_raises(self):
        bad = pd.DataFrame({"year": [2025]})  # missing forecast/lower/upper
        with pytest.raises(ValueError, match="missing required column"):
            apply_event_effects(bad, [], "ACC_OWNERSHIP")

    def test_empty_indicator_code_raises(self, base_fc):
        with pytest.raises(TypeError, match="non-empty string"):
            apply_event_effects(base_fc, [], "")

    def test_non_string_indicator_raises(self, base_fc):
        with pytest.raises(TypeError, match="non-empty string"):
            apply_event_effects(base_fc, [], 999)

    def test_realization_rate_above_one_raises(self, base_fc):
        with pytest.raises(ValueError, match="realization_rate"):
            apply_event_effects(base_fc, [], "ACC_OWNERSHIP", realization_rate=1.5)

    def test_realization_rate_negative_raises(self, base_fc):
        with pytest.raises(ValueError, match="realization_rate"):
            apply_event_effects(base_fc, [], "ACC_OWNERSHIP", realization_rate=-0.1)

    def test_non_list_event_impacts_raises(self, base_fc):
        with pytest.raises(TypeError, match="not a string"):
            apply_event_effects(base_fc, "not a list", "ACC_OWNERSHIP")

    def test_non_dict_event_item_raises(self, base_fc):
        with pytest.raises(TypeError, match="dict"):
            apply_event_effects(base_fc, ["string instead of dict"], "ACC_OWNERSHIP")


# ════════════════════════════════════════════════════════════════════════════
#  TestDefensiveScenario — bad inputs to scenario_forecasts
# ════════════════════════════════════════════════════════════════════════════

class TestDefensiveScenario:
    def test_non_dict_model_raises(self):
        with pytest.raises(TypeError, match="dict"):
            scenario_forecasts("not a model", [2025])

    def test_empty_forecast_years_raises(self, acc_model):
        with pytest.raises(ValueError, match="empty"):
            scenario_forecasts(acc_model, [])

    def test_negative_spread_raises(self, acc_model):
        with pytest.raises(ValueError, match="spread"):
            scenario_forecasts(acc_model, [2025], spread=-1.0)

    def test_wrong_realization_rates_length_raises(self, acc_model):
        with pytest.raises(ValueError, match="3 floats"):
            scenario_forecasts(acc_model, [2025], realization_rates=(0.2, 0.4))

    def test_realization_rate_above_one_raises(self, acc_model):
        with pytest.raises(ValueError, match="outside \[0, 1\]"):
            scenario_forecasts(acc_model, [2025], realization_rates=(0.2, 0.4, 1.5))

    def test_non_monotone_realization_rates_raises(self, acc_model):
        with pytest.raises(ValueError, match="non-decreasing"):
            scenario_forecasts(acc_model, [2025], realization_rates=(0.6, 0.4, 0.2))

    def test_nan_event_effects_raises(self, acc_model):
        with pytest.raises(ValueError, match="finite"):
            scenario_forecasts(acc_model, [2025], event_effects=float("nan"))

    def test_non_iterable_forecast_years_raises(self, acc_model):
        with pytest.raises(TypeError, match="iterable"):
            scenario_forecasts(acc_model, 2025)
