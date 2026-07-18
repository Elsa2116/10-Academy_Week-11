"""
Unit tests for forecasting utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecasting import fit_linear_trend, forecast_linear, scenario_forecasts


class TestLinearTrend:
    def test_fit_returns_expected_keys(self):
        years = [2011, 2014, 2017, 2021, 2024]
        values = [14, 22, 35, 46, 49]
        model = fit_linear_trend(years, values)
        assert "slope" in model
        assert "intercept" in model
        assert "r_squared" in model

    def test_slope_is_positive(self):
        years = [2011, 2014, 2017, 2021, 2024]
        values = [14, 22, 35, 46, 49]
        model = fit_linear_trend(years, values)
        assert model["slope"] > 0

    def test_r_squared_high(self):
        years = [2011, 2014, 2017, 2021, 2024]
        values = [14, 22, 35, 46, 49]
        model = fit_linear_trend(years, values)
        assert model["r_squared"] > 0.9


class TestForecastLinear:
    def test_forecast_returns_dataframe(self):
        years = [2011, 2014, 2017, 2021, 2024]
        values = [14, 22, 35, 46, 49]
        model = fit_linear_trend(years, values)
        fc = forecast_linear(model, [2025, 2026, 2027])
        assert isinstance(fc, pd.DataFrame)
        assert len(fc) == 3

    def test_upper_geq_lower(self):
        years = [2011, 2014, 2017, 2021, 2024]
        values = [14, 22, 35, 46, 49]
        model = fit_linear_trend(years, values)
        fc = forecast_linear(model, [2025, 2026, 2027])
        assert all(fc["upper"] >= fc["lower"])

    def test_forecast_bounded(self):
        years = [2011, 2014, 2017, 2021, 2024]
        values = [14, 22, 35, 46, 49]
        model = fit_linear_trend(years, values)
        fc = forecast_linear(model, [2025, 2026, 2027])
        assert all(fc["forecast"] >= 0)
        assert all(fc["forecast"] <= 100)


class TestScenarioForecasts:
    def test_optimistic_geq_base_geq_pessimistic(self):
        years = [2011, 2014, 2017, 2021, 2024]
        values = [14, 22, 35, 46, 49]
        model = fit_linear_trend(years, values)
        sc = scenario_forecasts(model, [2025, 2026, 2027])
        assert all(sc["optimistic"] >= sc["base"])
        assert all(sc["base"] >= sc["pessimistic"])
