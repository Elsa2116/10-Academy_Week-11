"""
Data loading utilities for Ethiopia Financial Inclusion Forecasting System.
Selam Analytics
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_unified_data(path=None):
    """Load the ethiopia_fi_unified_data.csv dataset."""
    if path is None:
        path = RAW_DIR / "ethiopia_fi_unified_data.csv"
    df = pd.read_csv(path, parse_dates=["observation_date"])
    return df


def load_reference_codes(path=None):
    """Load reference_codes.csv."""
    if path is None:
        path = RAW_DIR / "reference_codes.csv"
    return pd.read_csv(path)


def get_observations(df):
    """Filter to observation records only."""
    return df[df["record_type"] == "observation"].copy()


def get_events(df):
    """Filter to event records only."""
    return df[df["record_type"] == "event"].copy()


def get_impact_links(df):
    """Filter to impact_link records only."""
    return df[df["record_type"] == "impact_link"].copy()


def get_targets(df):
    """Filter to target records only."""
    return df[df["record_type"] == "target"].copy()


def get_indicator_series(df, indicator_code):
    """Get time series for a specific indicator code."""
    obs = get_observations(df)
    series = obs[obs["indicator_code"] == indicator_code][
        ["observation_date", "value_numeric", "confidence"]
    ].sort_values("observation_date")
    return series


def get_access_trajectory(df):
    """Return the account ownership trajectory (Findex years)."""
    return get_indicator_series(df, "ACC_OWNERSHIP")


def get_mm_trajectory(df):
    """Return mobile money account ownership trajectory."""
    return get_indicator_series(df, "ACC_MM_ACCOUNT")


def get_digital_payment_series(df):
    """Return digital payment adoption series."""
    return get_indicator_series(df, "USG_DIGITAL_PAYMENT")
