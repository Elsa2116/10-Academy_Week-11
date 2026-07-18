"""
Data loading utilities for Ethiopia Financial Inclusion Forecasting System.
Selam Analytics

Defensive coding conventions
─────────────────────────────
• _check_file()       — raises FileNotFoundError with a clear, actionable message
• _require_columns()  — raises ValueError listing every missing column at once
• _require_dataframe()— raises TypeError when caller passes wrong type
• _require_date_col() — raises TypeError when observation_date is not datetime
All public functions call these helpers before touching any data.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── path constants ───────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent.parent / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

# ── schema contracts ─────────────────────────────────────────────────────────
UNIFIED_REQUIRED_COLS: list[str] = [
    "record_type",
    "observation_date",
    "indicator_code",
    "value_numeric",
]

REFERENCE_REQUIRED_COLS: list[str] = ["field", "code"]

VALID_RECORD_TYPES: frozenset[str] = frozenset(
    {"observation", "event", "impact_link", "target"}
)


# ════════════════════════════════════════════════════════════════════════════
#  PRIVATE VALIDATION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _check_file(path: Path) -> None:
    """Raise FileNotFoundError with an actionable message if *path* does not exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"  Expected location : {path.resolve()}\n"
            f"  Hint: run the project from the repository root, or pass an "
            f"explicit `path=` argument."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Path exists but is not a file: {path.resolve()}"
        )


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
            f"{label} is missing required column(s): {missing}\n"
            f"  Columns present: {sorted(df.columns.tolist())}"
        )


def _require_date_col(df: pd.DataFrame, col: str = "observation_date") -> None:
    """Raise TypeError if *col* is not a datetime dtype."""
    if col not in df.columns:
        return  # _require_columns already handles missing columns
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        raise TypeError(
            f"Column '{col}' must be datetime64 dtype, "
            f"got {df[col].dtype} instead.\n"
            f"  Hint: pass parse_dates=['{col}'] when calling pd.read_csv()."
        )


def _require_non_empty(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Raise ValueError if the DataFrame has no rows."""
    if df.empty:
        raise ValueError(f"{label} contains no rows. Check the source file.")


def _validate_unified(df: pd.DataFrame, label: str = "Unified dataset") -> None:
    """Run all schema checks for the unified data file."""
    _require_dataframe(df, label)
    _require_columns(df, UNIFIED_REQUIRED_COLS, label)
    # Check emptiness BEFORE date-type check: an empty CSV parsed with
    # parse_dates=[] still has object dtype on the date column, which would
    # produce a misleading TypeError instead of the correct ValueError.
    _require_non_empty(df, label)
    _require_date_col(df, "observation_date")

    # Warn (don't crash) on unexpected record_type values so the caller
    # can decide how to handle them without breaking existing pipelines.
    unknown = set(df["record_type"].dropna().unique()) - VALID_RECORD_TYPES
    if unknown:
        warnings.warn(
            f"Unexpected record_type value(s) found: {sorted(unknown)}. "
            f"Valid types are: {sorted(VALID_RECORD_TYPES)}.",
            stacklevel=3,
        )


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC LOADING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def load_unified_data(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load and validate the ethiopia_fi_unified_data.csv dataset.

    Parameters
    ----------
    path : path-like, optional
        Override the default file location. Useful in tests.

    Returns
    -------
    pd.DataFrame
        Validated dataset with ``observation_date`` parsed as datetime64.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    ValueError
        If required columns are missing or the file is empty.
    TypeError
        If ``observation_date`` is not datetime64 after parsing.
    """
    if path is None:
        path = RAW_DIR / "ethiopia_fi_unified_data.csv"
    path = Path(path)
    _check_file(path)

    try:
        df = pd.read_csv(path, parse_dates=["observation_date"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse CSV at {path}: {exc}"
        ) from exc

    _validate_unified(df, label=f"File '{path.name}'")
    logger.info("Loaded %d records from %s", len(df), path.name)
    return df


def load_reference_codes(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load and validate reference_codes.csv.

    Parameters
    ----------
    path : path-like, optional
        Override the default file location.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if path is None:
        path = RAW_DIR / "reference_codes.csv"
    path = Path(path)
    _check_file(path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse CSV at {path}: {exc}") from exc

    _require_columns(df, REFERENCE_REQUIRED_COLS, label=f"File '{path.name}'")
    _require_non_empty(df, label=f"File '{path.name}'")
    logger.info("Loaded %d reference codes from %s", len(df), path.name)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FILTER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _filter_by_record_type(df: pd.DataFrame, record_type: str) -> pd.DataFrame:
    """Shared helper: validate the DataFrame then filter by record_type."""
    _require_dataframe(df, "df")
    _require_columns(df, ["record_type"], "df")
    result = df[df["record_type"] == record_type].copy()
    if result.empty:
        logger.warning(
            "No rows with record_type=%r found in the dataset.", record_type
        )
    return result


def get_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Return only observation records.

    Raises
    ------
    TypeError  : if *df* is not a DataFrame.
    ValueError : if 'record_type' column is missing.
    """
    return _filter_by_record_type(df, "observation")


def get_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return only event records."""
    return _filter_by_record_type(df, "event")


def get_impact_links(df: pd.DataFrame) -> pd.DataFrame:
    """Return only impact_link records."""
    return _filter_by_record_type(df, "impact_link")


def get_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Return only target records."""
    return _filter_by_record_type(df, "target")


# ════════════════════════════════════════════════════════════════════════════
#  INDICATOR SERIES HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_indicator_series(
    df: pd.DataFrame,
    indicator_code: str,
    *,
    min_points: int = 1,
) -> pd.DataFrame:
    """Extract the time series for a single indicator code.

    Parameters
    ----------
    df : pd.DataFrame
        The full unified dataset (will be filtered to observations).
    indicator_code : str
        Code to look up, e.g. ``"ACC_OWNERSHIP"``.
    min_points : int, optional
        Minimum number of data points required. Raises ValueError if fewer
        than this many are found. Default 1 (at least one point must exist).

    Returns
    -------
    pd.DataFrame
        Columns: observation_date, value_numeric, confidence — sorted by date.

    Raises
    ------
    TypeError  : if *df* is not a DataFrame or *indicator_code* is not a str.
    ValueError : if required columns are missing or fewer than *min_points*
                 valid rows are found for this indicator.
    """
    _require_dataframe(df, "df")
    if not isinstance(indicator_code, str) or not indicator_code.strip():
        raise TypeError(
            f"indicator_code must be a non-empty string, "
            f"got {type(indicator_code).__name__!r}: {indicator_code!r}"
        )
    _require_columns(df, ["indicator_code", "observation_date", "value_numeric"], "df")

    obs = get_observations(df)
    series = (
        obs[obs["indicator_code"] == indicator_code][
            ["observation_date", "value_numeric", "confidence"]
        ]
        .dropna(subset=["value_numeric"])
        .sort_values("observation_date")
        .reset_index(drop=True)
    )

    if len(series) < min_points:
        raise ValueError(
            f"Indicator '{indicator_code}' has {len(series)} valid data point(s); "
            f"at least {min_points} required.\n"
            f"  Check that the indicator code is spelled correctly and that the "
            f"dataset contains sufficient observations."
        )

    if series.empty:
        logger.warning("No data found for indicator_code=%r.", indicator_code)

    return series


def get_access_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Return account ownership trajectory (Findex years).

    Wraps :func:`get_indicator_series` for ``ACC_OWNERSHIP``.
    """
    return get_indicator_series(df, "ACC_OWNERSHIP", min_points=2)


def get_mm_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Return mobile money account ownership trajectory."""
    return get_indicator_series(df, "ACC_MM_ACCOUNT", min_points=1)


def get_digital_payment_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return digital payment adoption series."""
    return get_indicator_series(df, "USG_DIGITAL_PAYMENT", min_points=1)


# ════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a quick health-check summary of the unified dataset.

    Useful for sanity-checking before running heavy analyses.

    Parameters
    ----------
    df : pd.DataFrame
        The unified dataset (output of :func:`load_unified_data`).

    Returns
    -------
    dict with keys: total_rows, record_type_counts, date_range,
                    indicators, null_value_numeric_pct
    """
    _validate_unified(df, "df")

    rt_counts = df["record_type"].value_counts().to_dict()
    obs = get_observations(df)

    date_range = (None, None)
    if not obs.empty and pd.api.types.is_datetime64_any_dtype(obs["observation_date"]):
        date_range = (
            obs["observation_date"].min().date().isoformat(),
            obs["observation_date"].max().date().isoformat(),
        )

    null_pct = (
        round(obs["value_numeric"].isna().mean() * 100, 1) if not obs.empty else None
    )

    return {
        "total_rows":             len(df),
        "record_type_counts":     rt_counts,
        "date_range":             date_range,
        "indicators":             sorted(obs["indicator_code"].dropna().unique().tolist()),
        "null_value_numeric_pct": null_pct,
    }
