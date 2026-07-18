"""
Event impact modeling for Ethiopia Financial Inclusion Forecasting.
Selam Analytics

Defensive coding conventions
─────────────────────────────
• _require_dataframe()   — raises TypeError when caller passes wrong type
• _require_columns()     — raises ValueError listing every missing column
• _require_non_empty()   — raises ValueError on empty DataFrames
• _safe_magnitude()      — returns 0.0 and warns on unknown magnitude strings
• _safe_direction()      — returns 0 and warns on unknown direction strings
All public functions call these helpers before touching any data.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── look-up tables ───────────────────────────────────────────────────────────
MAGNITUDE_MAP: dict[str, float] = {
    "large":  5.0,
    "medium": 2.5,
    "small":  1.0,
}
DIRECTION_MAP: dict[str, int] = {
    "positive":  1,
    "negative": -1,
    "neutral":   0,
}

# Default realization rate: fraction of theoretical max that materialises.
DEFAULT_REALIZATION_RATE: float = 0.40

# Saturation rate constant λ — controls how quickly an event's effect grows
# toward its maximum (exponential saturation model).
DEFAULT_LAMBDA: float = 0.18


# ════════════════════════════════════════════════════════════════════════════
#  PRIVATE VALIDATION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _require_dataframe(obj: object, label: str = "argument") -> None:
    """Raise TypeError if *obj* is not a pandas DataFrame."""
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame for '{label}', "
            f"got {type(obj).__name__!r} instead."
        )


def _require_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    label: str = "DataFrame",
) -> None:
    """Raise ValueError listing every missing column at once."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{label}' is missing required column(s): {missing}\n"
            f"  Columns present: {sorted(df.columns.tolist())}"
        )


def _require_non_empty(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Raise ValueError if the DataFrame has no rows."""
    if df.empty:
        raise ValueError(f"'{label}' contains no rows. Nothing to model.")


def _safe_magnitude(raw: object, row_index: int) -> float:
    """Return the numeric magnitude for *raw*, warning on unknown values."""
    key = str(raw).strip().lower() if not (isinstance(raw, float) and np.isnan(raw)) else "small"
    val = MAGNITUDE_MAP.get(key)
    if val is None:
        warnings.warn(
            f"Unknown impact_magnitude {raw!r} at row index {row_index}. "
            f"Valid values: {list(MAGNITUDE_MAP)}. Defaulting to 0.0 (no impact).",
            stacklevel=3,
        )
        return 0.0
    return val


def _safe_direction(raw: object, row_index: int) -> int:
    """Return the numeric direction sign for *raw*, warning on unknown values."""
    key = str(raw).strip().lower() if not (isinstance(raw, float) and np.isnan(raw)) else "neutral"
    val = DIRECTION_MAP.get(key)
    if val is None:
        warnings.warn(
            f"Unknown impact_direction {raw!r} at row index {row_index}. "
            f"Valid values: {list(DIRECTION_MAP)}. Defaulting to 0 (neutral).",
            stacklevel=3,
        )
        return 0
    return val


def _validate_realization_rate(rate: float, label: str = "realization_rate") -> None:
    """Raise ValueError if *rate* is not in [0, 1]."""
    if not isinstance(rate, (int, float)) or not np.isfinite(rate):
        raise TypeError(
            f"'{label}' must be a finite number; got {rate!r}."
        )
    if not (0.0 <= float(rate) <= 1.0):
        raise ValueError(
            f"'{label}' must be in [0, 1]; got {rate!r}."
        )


def _validate_lambda(lam: float, label: str = "lam") -> None:
    """Raise ValueError if *lam* is not a positive finite number."""
    if not isinstance(lam, (int, float)) or not np.isfinite(lam) or lam <= 0:
        raise ValueError(
            f"'{label}' must be a positive finite number; got {lam!r}."
        )


def _validate_indicator_code(code: object, label: str = "indicator_code") -> str:
    """Raise TypeError if *code* is not a non-empty string."""
    if not isinstance(code, str) or not code.strip():
        raise TypeError(
            f"'{label}' must be a non-empty string; "
            f"got {type(code).__name__!r}: {code!r}."
        )
    return code.strip()


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def build_association_matrix(
    impact_links_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build an Event × Indicator association matrix of signed impact magnitudes.

    Each cell contains ``direction × magnitude`` (pp) for the corresponding
    event-indicator pair.  Cells with no link are 0.0.

    Parameters
    ----------
    impact_links_df : pd.DataFrame
        Rows with record_type == 'impact_link' from the unified dataset.
        Must contain columns: ``parent_id``, ``related_indicator``,
        ``impact_magnitude``, ``impact_direction``.
    events_df : pd.DataFrame
        Rows with record_type == 'event' from the unified dataset.
        Must contain columns: ``id``, ``indicator_code``.

    Returns
    -------
    pd.DataFrame
        Index = event IDs, columns = indicator codes.
        Values in range [-5, 5] (pp).  Empty if no valid links.

    Raises
    ------
    TypeError  : if either argument is not a DataFrame.
    ValueError : if required columns are missing in either DataFrame.
    """
    _require_dataframe(impact_links_df, "impact_links_df")
    _require_dataframe(events_df,       "events_df")
    _require_columns(
        impact_links_df,
        ["parent_id", "related_indicator"],
        "impact_links_df",
    )
    _require_columns(events_df, ["id"], "events_df")

    # Non-fatal: warn if the DataFrames are empty instead of crashing —
    # the caller might be building the matrix incrementally.
    if impact_links_df.empty:
        warnings.warn(
            "build_association_matrix: 'impact_links_df' is empty. "
            "Returning an empty matrix.",
            stacklevel=2,
        )
        return pd.DataFrame(dtype=float)

    indicators = impact_links_df["related_indicator"].dropna().unique().tolist()
    events     = impact_links_df["parent_id"].dropna().unique().tolist()

    if not indicators or not events:
        warnings.warn(
            "build_association_matrix: No valid indicator or event IDs found "
            "after dropping NaN values. Returning an empty matrix.",
            stacklevel=2,
        )
        return pd.DataFrame(dtype=float)

    matrix = pd.DataFrame(0.0, index=events, columns=indicators)

    for idx, row in impact_links_df.iterrows():
        evt = row.get("parent_id")
        ind = row.get("related_indicator")

        if pd.isna(evt) or pd.isna(ind):
            logger.debug(
                "Skipping impact_link row %s: parent_id=%r or related_indicator=%r is NaN.",
                idx, evt, ind,
            )
            continue

        # Guard: skip if event or indicator is not in our index/columns
        # (can happen if a link references an event that was filtered out).
        if evt not in matrix.index:
            warnings.warn(
                f"impact_link row {idx}: parent_id={evt!r} not found in events index. "
                f"Skipping.",
                stacklevel=2,
            )
            continue
        if ind not in matrix.columns:
            warnings.warn(
                f"impact_link row {idx}: related_indicator={ind!r} not found in "
                f"indicators columns. Skipping.",
                stacklevel=2,
            )
            continue

        mag       = _safe_magnitude(row.get("impact_magnitude"), idx)
        direction = _safe_direction(row.get("impact_direction"), idx)
        matrix.at[evt, ind] = float(direction * mag)

    logger.info(
        "build_association_matrix: %d events × %d indicators",
        len(events), len(indicators),
    )
    return matrix


def compute_cumulative_impact(
    impact_links_df: pd.DataFrame,
    indicator_code: str,
    forecast_year: float,
    *,
    realization_rate: float = DEFAULT_REALIZATION_RATE,
    lam: float = DEFAULT_LAMBDA,
) -> float:
    """Compute the cumulative event impact on *indicator_code* at *forecast_year*.

    Uses an exponential saturation model:
        δ(t) = magnitude × realization_rate × (1 − e^{−λ · Δt})

    where Δt is the time (years) since the event's effective date
    (start date + lag).  Events that have not yet reached their effective
    date contribute 0.

    Parameters
    ----------
    impact_links_df : pd.DataFrame
        Impact link records. Must contain columns:
        ``related_indicator``, ``impact_magnitude``, ``impact_direction``.
        Optionally ``lag_months`` (defaults to 12 if missing).
        Optionally ``observation_date`` for the parent event year
        (defaults to 2021 if missing).
    indicator_code : str
        Indicator to compute the impact for, e.g. ``"ACC_OWNERSHIP"``.
    forecast_year : float
        The year for which to compute the cumulative effect, e.g. ``2025.0``.
    realization_rate : float, optional
        Fraction of the theoretical maximum that materialises. Default 0.40.
    lam : float, optional
        Saturation rate constant (λ). Higher values = faster saturation.
        Default 0.18 (reaches ~53% of max within 3 years).

    Returns
    -------
    float
        Cumulative impact in percentage points (pp).
        Returns 0.0 if no relevant links are found — does not raise.

    Raises
    ------
    TypeError  : if *impact_links_df* is not a DataFrame or *indicator_code*
                 is not a string.
    ValueError : if required columns are missing, *forecast_year* is not
                 finite, or *realization_rate* / *lam* are out of range.
    """
    _require_dataframe(impact_links_df, "impact_links_df")
    indicator_code = _validate_indicator_code(indicator_code, "indicator_code")
    _require_columns(
        impact_links_df,
        ["related_indicator"],
        "impact_links_df",
    )
    _validate_realization_rate(realization_rate, "realization_rate")
    _validate_lambda(lam, "lam")

    if not np.isfinite(float(forecast_year)):
        raise ValueError(
            f"'forecast_year' must be a finite number; got {forecast_year!r}."
        )
    forecast_year = float(forecast_year)

    relevant = impact_links_df[
        impact_links_df["related_indicator"] == indicator_code
    ].copy()

    if relevant.empty:
        logger.info(
            "compute_cumulative_impact: No impact links found for indicator %r. "
            "Returning 0.0.",
            indicator_code,
        )
        return 0.0

    total = 0.0
    for idx, row in relevant.iterrows():
        mag       = _safe_magnitude(row.get("impact_magnitude"), idx)
        direction = _safe_direction(row.get("impact_direction"), idx)

        if direction == 0 or mag == 0.0:
            continue  # Neutral or unknown — skip.

        # Determine the year the event's effect starts (event year + lag).
        lag_months = float(row.get("lag_months", 12))
        if not np.isfinite(lag_months) or lag_months < 0:
            warnings.warn(
                f"Row {idx}: lag_months={lag_months!r} is invalid; defaulting to 12.",
                stacklevel=2,
            )
            lag_months = 12.0

        # Try to get the event year from observation_date in the row; fall back
        # to a neutral default so one bad row does not crash the whole model.
        raw_date = row.get("observation_date")
        try:
            if pd.isna(raw_date):
                raise ValueError("NaT")
            event_year = float(pd.Timestamp(raw_date).year)
        except Exception:
            event_year = 2021.0
            warnings.warn(
                f"Row {idx}: cannot parse observation_date={raw_date!r}; "
                f"defaulting event year to {event_year}.",
                stacklevel=2,
            )

        effective_year = event_year + lag_months / 12.0
        delta_t = forecast_year - effective_year

        if delta_t <= 0:
            # Event has not yet had time to take effect.
            continue

        saturation = 1.0 - np.exp(-lam * delta_t)
        contribution = direction * mag * realization_rate * saturation
        total += contribution

    logger.info(
        "compute_cumulative_impact: indicator=%r, forecast_year=%g, total=%.4f pp",
        indicator_code, forecast_year, total,
    )
    return float(total)


def saturation_curve(
    event_year: float,
    lag_months: float,
    magnitude: float,
    direction: int = 1,
    *,
    realization_rate: float = DEFAULT_REALIZATION_RATE,
    lam: float = DEFAULT_LAMBDA,
    years: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, effect_values) for an exponential saturation curve.

    Useful for visualising how a single event's impact builds over time.

    Parameters
    ----------
    event_year   : float  — calendar year the event occurred.
    lag_months   : float  — months before the effect begins.
    magnitude    : float  — maximum impact in pp (e.g. 5.0 for 'large').
    direction    : int    — +1 or -1.
    realization_rate : float — fraction of max that materialises.
    lam          : float  — saturation rate constant.
    years        : ndarray, optional — evaluation years. Defaults to
                   0.25-year steps from event_year-1 to event_year+8.

    Returns
    -------
    (years_arr, effect_arr) — both float64 ndarrays.

    Raises
    ------
    ValueError : if *event_year*, *lag_months*, or *magnitude* are invalid.
    """
    if not np.isfinite(float(event_year)):
        raise ValueError(f"'event_year' must be finite; got {event_year!r}.")
    if lag_months < 0:
        raise ValueError(f"'lag_months' must be ≥ 0; got {lag_months!r}.")
    if magnitude < 0:
        raise ValueError(f"'magnitude' must be ≥ 0; got {magnitude!r}.")
    _validate_realization_rate(realization_rate, "realization_rate")
    _validate_lambda(lam, "lam")

    if years is None:
        years = np.arange(float(event_year) - 1, float(event_year) + 9, 0.25)

    effective_year = float(event_year) + lag_months / 12.0
    effects = np.where(
        years < effective_year,
        0.0,
        direction * magnitude * realization_rate
        * (1.0 - np.exp(-lam * (years - effective_year))),
    )
    return years, effects


def rank_events_by_impact(
    impact_links_df: pd.DataFrame,
    indicator_code: str,
    forecast_year: float,
    *,
    realization_rate: float = DEFAULT_REALIZATION_RATE,
    lam: float = DEFAULT_LAMBDA,
) -> pd.DataFrame:
    """Return events ranked by their absolute impact on *indicator_code*.

    Convenience function for dashboard and report tables.

    Parameters
    ----------
    impact_links_df : pd.DataFrame — impact link records.
    indicator_code  : str          — indicator to rank against.
    forecast_year   : float        — year to evaluate impacts at.
    realization_rate, lam          — passed to saturation calculation.

    Returns
    -------
    pd.DataFrame with columns: parent_id, impact_pp, abs_impact_pp — sorted
        by abs_impact_pp descending.  Empty DataFrame if no relevant links.

    Raises
    ------
    TypeError  : if *impact_links_df* is not a DataFrame.
    ValueError : if required columns are missing.
    """
    _require_dataframe(impact_links_df, "impact_links_df")
    indicator_code = _validate_indicator_code(indicator_code, "indicator_code")
    _require_columns(
        impact_links_df,
        ["related_indicator", "parent_id"],
        "impact_links_df",
    )
    _validate_realization_rate(realization_rate, "realization_rate")
    _validate_lambda(lam, "lam")

    relevant = impact_links_df[
        impact_links_df["related_indicator"] == indicator_code
    ].copy()

    if relevant.empty:
        logger.info(
            "rank_events_by_impact: No impact links for indicator %r.", indicator_code
        )
        return pd.DataFrame(columns=["parent_id", "impact_pp", "abs_impact_pp"])

    rows = []
    for idx, row in relevant.iterrows():
        mag       = _safe_magnitude(row.get("impact_magnitude"), idx)
        direction = _safe_direction(row.get("impact_direction"), idx)

        lag_months = float(row.get("lag_months", 12))
        if not np.isfinite(lag_months) or lag_months < 0:
            lag_months = 12.0

        raw_date   = row.get("observation_date")
        try:
            event_year = float(pd.Timestamp(raw_date).year) if not pd.isna(raw_date) else 2021.0
        except Exception:
            event_year = 2021.0

        effective_year = event_year + lag_months / 12.0
        delta_t        = float(forecast_year) - effective_year

        if delta_t <= 0:
            contribution = 0.0
        else:
            saturation   = 1.0 - np.exp(-lam * delta_t)
            contribution = direction * mag * realization_rate * saturation

        rows.append({
            "parent_id":      row.get("parent_id", "UNKNOWN"),
            "impact_pp":      round(contribution, 4),
            "abs_impact_pp":  round(abs(contribution), 4),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("abs_impact_pp", ascending=False)
        .reset_index(drop=True)
    )
    return result
