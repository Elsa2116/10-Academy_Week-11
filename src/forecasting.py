"""
Forecasting models for Ethiopia Financial Inclusion Indicators.
Selam Analytics
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def year_to_numeric(year_series):
    """Convert year or datetime to numeric for regression."""
    if hasattr(year_series.iloc[0], "year"):
        return year_series.apply(lambda x: x.year).values.reshape(-1, 1)
    return year_series.values.reshape(-1, 1)


def fit_linear_trend(years, values):
    """Fit a simple OLS linear trend and return model."""
    X = np.array(years).reshape(-1, 1)
    y = np.array(values)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        X.flatten(), y
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
    }


def forecast_linear(model, forecast_years, confidence=0.95):
    """Generate linear forecast with confidence intervals."""
    forecasts = []
    n = len(forecast_years)
    z = stats.norm.ppf((1 + confidence) / 2)
    for yr in forecast_years:
        point = model["intercept"] + model["slope"] * yr
        margin = z * model["std_err"] * (1 + 0.5 * (yr - 2024))
        forecasts.append(
            {
                "year": yr,
                "forecast": max(0.0, min(100.0, point)),
                "lower": max(0.0, min(100.0, point - margin)),
                "upper": max(0.0, min(100.0, point + margin)),
            }
        )
    return pd.DataFrame(forecasts)


def apply_event_effects(base_forecast, event_impacts, indicator_code):
    """
    Apply event impacts to baseline forecast.
    event_impacts: list of dicts with keys: year_effective, magnitude
    """
    df = base_forecast.copy()
    impact_lookup = {"large": 5.0, "medium": 2.5, "small": 1.0}
    total_impact = 0.0
    for evt in event_impacts:
        mag = impact_lookup.get(evt.get("impact_magnitude", "small"), 0.0)
        direction = 1 if evt.get("impact_direction", "positive") == "positive" else -1
        # Events are applied with diminishing contribution
        total_impact += direction * mag * 0.5  # 50% realization rate
    df["forecast"] = (df["forecast"] + total_impact).clip(0, 100)
    df["lower"] = (df["lower"] + total_impact * 0.5).clip(0, 100)
    df["upper"] = (df["upper"] + total_impact * 1.5).clip(0, 100)
    return df


def scenario_forecasts(model, forecast_years, event_effects=0.0):
    """Generate optimistic, base, pessimistic scenarios."""
    results = []
    for yr in forecast_years:
        base = model["intercept"] + model["slope"] * yr + event_effects
        results.append(
            {
                "year": yr,
                "pessimistic": max(0.0, min(100.0, base - 4)),
                "base": max(0.0, min(100.0, base)),
                "optimistic": max(0.0, min(100.0, base + 4)),
            }
        )
    return pd.DataFrame(results)
