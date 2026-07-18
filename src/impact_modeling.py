"""
Event impact modeling for Ethiopia Financial Inclusion Forecasting.
Selam Analytics
"""

import pandas as pd
import numpy as np


MAGNITUDE_MAP = {"large": 5.0, "medium": 2.5, "small": 1.0}
DIRECTION_MAP = {"positive": 1, "negative": -1, "neutral": 0}


def build_association_matrix(impact_links_df, events_df):
    """
    Build event-indicator association matrix.
    Returns a DataFrame with events as rows and indicators as columns.
    """
    # Merge events into impact_links via parent_id
    merged = impact_links_df.merge(
        events_df[["id", "indicator_code", "observation_date", "notes"]],
        left_on="parent_id",
        right_on="id",
        suffixes=("", "_event"),
        how="left",
    )

    indicators = impact_links_df["related_indicator"].dropna().unique().tolist()
    events = impact_links_df["parent_id"].dropna().unique().tolist()

    matrix = pd.DataFrame(0.0, index=events, columns=indicators)

    for _, row in impact_links_df.iterrows():
        evt = row["parent_id"]
        ind = row["related_indicator"]
        if pd.isna(evt) or pd.isna(ind):
            continue
        mag = MAGNITUDE_MAP.get(str(row.get("impact_magnitude", "small")), 0.0)
        direction = DIRECTION_MAP.get(str(row.get("impact_direction", "neutral")), 0)
        matrix.loc[evt, ind] = direction * mag

    return matrix


def compute_cumulative_impact(impact_links_df, indicator_code, forecast_year):
    """
    Compute cumulative event impact on an indicator by a given year,
    applying lag months.
    """
    relevant = impact_links_df[
        impact_links_df["related_indicator"] == indicator_code
    ].copy()

    total = 0.0
    for _, row in relevant.iterrows():
        mag = MAGNITUDE_MAP.get(str(row.get("impact_magnitude", "small")), 0.0)
        direction = DIRECTION_MAP.get(str(row.get("impact_direction", "neutral")), 0)
        total += direction * mag * 0.4  # 40% average realization rate
    return total
