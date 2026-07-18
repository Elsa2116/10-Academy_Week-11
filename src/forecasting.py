"""
Forecasting models for Ethiopia Financial Inclusion Indicators.
Selam Analytics

Defensive coding conventions
─────────────────────────────
• _validate_years_values()  — checks length, dtype, no NaN, min data points
• _validate_model_dict()    — checks all required keys are present and numeric
• _validate_forecast_years()— checks non-empty list of numeric years
• _clamp()                  — clips a value or Series to [0, 100]
All public functions call these helpers before touching any data.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Keys that every model dict returned by fit_linear_trend() must contain.
_MODEL_REQUIRED_KEYS: tuple[str, ...] = (
    "slope", "intercept", "r_squared", "p_value", "std_err",
    "n_obs", "mean_year",
)

# Impact magnitude look-up (pp at full realisation).
IMPACT_MAGNITUDE_MAP: dict[str, float] = {
    "large":  5.0,
    "medium": 2.5,
    "small":  1.0,
}

DEFAULT_REALIZATION_RATE: float = 0.40


# ════════════════════════════════════════════════════════════════════════════
#  PRIVATE VALIDATION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _clamp(value: float | np.ndarray | pd.Series, lo: float = 0.0, hi: float = 100.0):
    """Clip a scalar, ndarray, or Series to [lo, hi]."""
    if isinstance(value, pd.Series):
        return value.clip(lo, hi)
    if isinstance(value, np.ndarray):
        return np.clip(value, lo, hi)
    return max(lo, min(hi, float(value)))


def _validate_years_values(
    years: Sequence,
    values: Sequence,
    *,
    min_points: int = 2,
    label: str = "fit_linear_trend",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce *years* and *values* for regression.

    Returns
    -------
    (years_arr, values_arr) as float64 ndarrays.

    Raises
    ------
    TypeError  : if inputs cannot be converted to numeric arrays.
    ValueError : if lengths differ, arrays are empty, contain NaN/Inf,
                 or have fewer than *min_points* valid observations.
    """
    # --- coerce to numpy ---
    try:
        y_arr = np.asarray(years,  dtype=float)
        v_arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"[{label}] 'years' and 'values' must be numeric sequences. "
            f"Conversion failed: {exc}"
        ) from exc

    # --- length match ---
    if y_arr.shape != v_arr.shape:
        raise ValueError(
            f"[{label}] 'years' and 'values' must have the same length; "
            f"got {len(y_arr)} and {len(v_arr)}."
        )

    # --- not empty ---
    if y_arr.size == 0:
        raise ValueError(f"[{label}] 'years' and 'values' must not be empty.")

    # --- minimum points for regression ---
    if y_arr.size < min_points:
        raise ValueError(
            f"[{label}] At least {min_points} data point(s) are required for "
            f"regression; only {y_arr.size} provided."
        )

    # --- NaN / Inf check ---
    bad_y = ~np.isfinite(y_arr)
    bad_v = ~np.isfinite(v_arr)
    if bad_y.any():
        raise ValueError(
            f"[{label}] 'years' contains NaN or Inf at index/indices: "
            f"{np.where(bad_y)[0].tolist()}"
        )
    if bad_v.any():
        raise ValueError(
            f"[{label}] 'values' contains NaN or Inf at index/indices: "
            f"{np.where(bad_v)[0].tolist()}"
        )

    # --- reasonable range warnings (non-fatal) ---
    if v_arr.min() < 0 or v_arr.max() > 100:
        warnings.warn(
            f"[{label}] Some 'values' are outside the expected [0, 100] "
            f"percentage range (min={v_arr.min():.2f}, max={v_arr.max():.2f}). "
            f"Results will be clamped to [0, 100] in forecasts.",
            stacklevel=3,
        )

    return y_arr, v_arr


def _validate_model_dict(model: Any, label: str = "model") -> None:
    """Raise TypeError / ValueError if *model* is not a valid model dict."""
    if not isinstance(model, dict):
        raise TypeError(
            f"'{label}' must be a dict returned by fit_linear_trend(); "
            f"got {type(model).__name__!r}."
        )
    missing = [k for k in _MODEL_REQUIRED_KEYS if k not in model]
    if missing:
        raise ValueError(
            f"'{label}' dict is missing required key(s): {missing}.\n"
            f"  Use the dict returned by fit_linear_trend()."
        )
    for key in ("slope", "intercept", "std_err"):
        if not np.isfinite(model[key]):
            raise ValueError(
                f"'{label}[{key!r}]' is not finite ({model[key]!r}). "
                f"Re-fit the model — the training data may be degenerate."
            )


def _validate_forecast_years(forecast_years: Sequence, label: str = "forecast_years") -> list[float]:
    """Validate and return forecast years as a sorted list of floats.

    Raises
    ------
    TypeError  : if *forecast_years* is not iterable or contains non-numeric values.
    ValueError : if the sequence is empty.
    """
    if not hasattr(forecast_years, "__iter__"):
        raise TypeError(
            f"'{label}' must be an iterable of numeric years; "
            f"got {type(forecast_years).__name__!r}."
        )
    try:
        yr_list = [float(y) for y in forecast_years]
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"All elements of '{label}' must be numeric; conversion failed: {exc}"
        ) from exc

    if not yr_list:
        raise ValueError(f"'{label}' must not be empty.")

    return sorted(yr_list)


def _validate_event_impacts(event_impacts: Sequence[dict], label: str = "event_impacts") -> None:
    """Warn (non-fatal) about individual impact dicts that look malformed."""
    # Strings are iterable but are never a valid list of impact dicts.
    if isinstance(event_impacts, (str, bytes)):
        raise TypeError(
            f"'{label}' must be an iterable of dicts, not a string. "
            f"Pass a list of impact-dict objects."
        )
    if not hasattr(event_impacts, "__iter__"):
        raise TypeError(
            f"'{label}' must be an iterable of dicts; "
            f"got {type(event_impacts).__name__!r}."
        )
    for i, evt in enumerate(event_impacts):
        if not isinstance(evt, dict):
            raise TypeError(
                f"'{label}[{i}]' must be a dict; got {type(evt).__name__!r}."
            )
        if "impact_magnitude" not in evt:
            warnings.warn(
                f"'{label}[{i}]' is missing 'impact_magnitude'; "
                f"defaulting to 'small' (1.0 pp).",
                stacklevel=3,
            )
        if "impact_direction" not in evt:
            warnings.warn(
                f"'{label}[{i}]' is missing 'impact_direction'; "
                f"defaulting to 'positive'.",
                stacklevel=3,
            )


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def fit_linear_trend(
    years: Sequence,
    values: Sequence,
    *,
    min_points: int = 2,
) -> dict:
    """Fit an OLS linear trend to (years, values) data.

    Parameters
    ----------
    years : array-like of numeric
        Observation years (e.g. ``[2011, 2014, 2017, 2021, 2024]``).
    values : array-like of numeric
        Corresponding indicator values in the range [0, 100].
    min_points : int, optional
        Minimum number of data points required. Default 2.

    Returns
    -------
    dict with keys:
        slope, intercept, r_squared, p_value, std_err, n_obs, mean_year

    Raises
    ------
    TypeError  : if inputs are not numeric sequences.
    ValueError : if inputs are invalid (see _validate_years_values).
    RuntimeError : if scipy.stats.linregress fails unexpectedly.
    """
    y_arr, v_arr = _validate_years_values(
        years, values, min_points=min_points, label="fit_linear_trend"
    )

    # Degenerate case: only 1 point — scipy returns all-NaN.
    # Return a horizontal-line model pinned to the single observed value.
    if y_arr.size == 1:
        import warnings as _warnings
        _warnings.warn(
            "fit_linear_trend: only 1 data point supplied. "
            "Returning a zero-slope model (horizontal line). "
            "Confidence intervals will be zero-width.",
            stacklevel=2,
        )
        return {
            "slope":     0.0,
            "intercept": float(v_arr[0]),
            "r_squared": float("nan"),   # undefined for 1 point
            "p_value":   float("nan"),
            "std_err":   0.0,
            "n_obs":     1,
            "mean_year": float(y_arr[0]),
        }

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_arr, v_arr)
    except Exception as exc:
        raise RuntimeError(
            f"OLS regression failed: {exc}\n"
            f"  years  = {y_arr.tolist()}\n"
            f"  values = {v_arr.tolist()}"
        ) from exc

    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise RuntimeError(
            "OLS regression produced non-finite slope or intercept. "
            "Check for duplicate year values or zero-variance in years."
        )

    model = {
        "slope":     float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value":   float(p_value),
        "std_err":   float(std_err),
        "n_obs":     int(y_arr.size),
        "mean_year": float(y_arr.mean()),
    }

    logger.info(
        "fit_linear_trend: slope=%.4f, intercept=%.2f, R²=%.4f (n=%d)",
        model["slope"], model["intercept"], model["r_squared"], model["n_obs"],
    )
    return model


def forecast_linear(
    model: dict,
    forecast_years: Sequence,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Generate a linear point forecast with symmetric confidence intervals.

    Parameters
    ----------
    model : dict
        Output of :func:`fit_linear_trend`.
    forecast_years : iterable of numeric
        Years to forecast.
    confidence : float, optional
        Confidence level for the interval. Must be in (0, 1). Default 0.95.

    Returns
    -------
    pd.DataFrame with columns: year, forecast, lower, upper.
        All values are clamped to [0, 100].

    Raises
    ------
    TypeError  : if *model* is not a dict or *forecast_years* is not iterable.
    ValueError : if *model* is missing required keys, *forecast_years* is empty,
                 or *confidence* is outside (0, 1).
    """
    _validate_model_dict(model, "model")
    yr_list = _validate_forecast_years(forecast_years, "forecast_years")

    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"'confidence' must be in the open interval (0, 1); got {confidence!r}."
        )

    z = float(stats.norm.ppf((1.0 + confidence) / 2.0))
    n = model["n_obs"]
    mean_yr = model["mean_year"]

    # Approximate prediction-interval half-width using OLS formula:
    # se_pred(x) = std_err * sqrt(1 + 1/n + (x - x̄)² / Σ(xᵢ - x̄)²)
    # We approximate the variance term as (yr - mean_yr)² / n for simplicity
    # (conservative when n is small, which is our case).
    rows = []
    for yr in yr_list:
        point = model["intercept"] + model["slope"] * yr
        # Width grows with distance from the training mean — appropriate for
        # small-sample extrapolation (n=5 Findex surveys).
        extra_variance = (yr - mean_yr) ** 2 / max(n, 1)
        margin = z * model["std_err"] * float(np.sqrt(1.0 + 1.0 / n + extra_variance))
        rows.append({
            "year":     int(yr),
            "forecast": float(_clamp(point)),
            "lower":    float(_clamp(point - margin)),
            "upper":    float(_clamp(point + margin)),
        })

    return pd.DataFrame(rows)


def apply_event_effects(
    base_forecast: pd.DataFrame,
    event_impacts: Sequence[dict],
    indicator_code: str,
    *,
    realization_rate: float = DEFAULT_REALIZATION_RATE,
) -> pd.DataFrame:
    """Overlay cumulative event effects on a baseline forecast DataFrame.

    Parameters
    ----------
    base_forecast : pd.DataFrame
        Output of :func:`forecast_linear`. Must contain columns
        ``year``, ``forecast``, ``lower``, ``upper``.
    event_impacts : list of dict
        Each dict must have keys ``impact_magnitude`` (str) and
        ``impact_direction`` (str, 'positive' | 'negative' | 'neutral').
    indicator_code : str
        The indicator being adjusted — used only for log messages.
    realization_rate : float, optional
        Fraction of the theoretical maximum impact that materialises.
        Must be in [0, 1]. Default 0.40 (40%).

    Returns
    -------
    pd.DataFrame with the same columns as *base_forecast*, adjusted values
        clamped to [0, 100].

    Raises
    ------
    TypeError  : if *base_forecast* is not a DataFrame, or *event_impacts*
                 is not an iterable of dicts.
    ValueError : if required columns are missing, *realization_rate* is
                 outside [0, 1], or *indicator_code* is empty.
    """
    if not isinstance(base_forecast, pd.DataFrame):
        raise TypeError(
            f"'base_forecast' must be a pandas DataFrame; "
            f"got {type(base_forecast).__name__!r}."
        )
    required_cols = {"year", "forecast", "lower", "upper"}
    missing = required_cols - set(base_forecast.columns)
    if missing:
        raise ValueError(
            f"'base_forecast' is missing required column(s): {sorted(missing)}.\n"
            f"  Use the DataFrame returned by forecast_linear()."
        )
    if not isinstance(indicator_code, str) or not indicator_code.strip():
        raise TypeError(
            f"'indicator_code' must be a non-empty string; got {indicator_code!r}."
        )
    if not (0.0 <= realization_rate <= 1.0):
        raise ValueError(
            f"'realization_rate' must be in [0, 1]; got {realization_rate!r}."
        )

    _validate_event_impacts(event_impacts, "event_impacts")

    total_impact = 0.0
    for evt in event_impacts:
        raw_mag = IMPACT_MAGNITUDE_MAP.get(
            str(evt.get("impact_magnitude", "small")).lower(), 1.0
        )
        direction_str = str(evt.get("impact_direction", "positive")).lower()
        direction = 1 if direction_str == "positive" else (-1 if direction_str == "negative" else 0)
        total_impact += direction * raw_mag * realization_rate

    df = base_forecast.copy()
    df["forecast"] = _clamp(df["forecast"] + total_impact)
    df["lower"]    = _clamp(df["lower"]    + total_impact * 0.5)
    df["upper"]    = _clamp(df["upper"]    + total_impact * 1.5)

    logger.info(
        "apply_event_effects: indicator=%s, n_events=%d, total_impact=%.3f pp",
        indicator_code, len(list(event_impacts)), total_impact,
    )
    return df


def scenario_forecasts(
    model: dict,
    forecast_years: Sequence,
    *,
    event_effects: float = 0.0,
    spread: float = 4.0,
    realization_rates: tuple[float, float, float] = (0.20, 0.40, 0.60),
) -> pd.DataFrame:
    """Generate pessimistic, base, and optimistic scenario forecasts.

    Parameters
    ----------
    model : dict
        Output of :func:`fit_linear_trend`.
    forecast_years : iterable of numeric
        Years to forecast.
    event_effects : float, optional
        Base cumulative event contribution (pp) added on top of the trend.
        Scenarios scale this by *realization_rates*. Default 0.0.
    spread : float, optional
        Extra fixed spread (pp) applied symmetrically beyond the event
        scaling for the pessimistic/optimistic scenarios. Default 4.0.
    realization_rates : tuple of 3 floats, optional
        Realization rates for (pessimistic, base, optimistic).
        Default (0.20, 0.40, 0.60).

    Returns
    -------
    pd.DataFrame with columns: year, pessimistic, base, optimistic.
        All values clamped to [0, 100].

    Raises
    ------
    TypeError  : if *model* is not a dict or *forecast_years* is not iterable.
    ValueError : if *model* is missing keys, *forecast_years* is empty,
                 *spread* is negative, or *realization_rates* are invalid.
    """
    _validate_model_dict(model, "model")
    yr_list = _validate_forecast_years(forecast_years, "forecast_years")

    if spread < 0:
        raise ValueError(f"'spread' must be non-negative; got {spread!r}.")
    if len(realization_rates) != 3:
        raise ValueError(
            f"'realization_rates' must be a tuple of exactly 3 floats "
            f"(pessimistic, base, optimistic); got {len(realization_rates)} values."
        )
    for i, rr in enumerate(realization_rates):
        if not (0.0 <= rr <= 1.0):
            raise ValueError(
                f"realization_rates[{i}]={rr!r} is outside [0, 1]."
            )
    pess_rr, base_rr, opt_rr = realization_rates
    if not (pess_rr <= base_rr <= opt_rr):
        raise ValueError(
            f"realization_rates must be non-decreasing "
            f"(pessimistic ≤ base ≤ optimistic); "
            f"got {realization_rates}."
        )

    if not np.isfinite(event_effects):
        raise ValueError(
            f"'event_effects' must be a finite number; got {event_effects!r}."
        )

    rows = []
    for yr in yr_list:
        trend = model["intercept"] + model["slope"] * yr
        rows.append({
            "year":        int(yr),
            "pessimistic": float(_clamp(trend + event_effects * pess_rr - spread)),
            "base":        float(_clamp(trend + event_effects * base_rr)),
            "optimistic":  float(_clamp(trend + event_effects * opt_rr + spread)),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: year-to-numeric converter (unchanged, kept for compatibility)
# ════════════════════════════════════════════════════════════════════════════

def year_to_numeric(year_series: pd.Series) -> np.ndarray:
    """Convert a year or datetime Series to a numeric (float) column vector."""
    if not isinstance(year_series, pd.Series):
        raise TypeError(
            f"'year_series' must be a pandas Series; "
            f"got {type(year_series).__name__!r}."
        )
    if year_series.empty:
        raise ValueError("'year_series' must not be empty.")
    if hasattr(year_series.iloc[0], "year"):
        return year_series.apply(lambda x: x.year).values.reshape(-1, 1)
    try:
        return year_series.astype(float).values.reshape(-1, 1)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"Could not convert 'year_series' to numeric: {exc}"
        ) from exc
