"""
Generate analysis-ready enriched datasets for Ethiopia FI forecasting.

The enrichment is deliberately deterministic: it does not fetch live data.
All source claims are carried forward from data/raw/ethiopia_fi_unified_data.csv
and transformed into audit-friendly fields for EDA, event overlays, and models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from data_loader import RAW_DIR, PROCESSED_DIR, load_unified_data
except ImportError:  # pragma: no cover - supports python -m src.data_enrichment
    from .data_loader import RAW_DIR, PROCESSED_DIR, load_unified_data


VERSION = "v2026-07-20"
RAW_DATA_PATH = RAW_DIR / "ethiopia_fi_unified_data.csv"
ENRICHED_DATA_PATH = PROCESSED_DIR / "ethiopia_fi_enriched.csv"
METADATA_PATH = PROCESSED_DIR / "enrichment_metadata.csv"
LOG_PATH = PROCESSED_DIR / "data_enrichment_log.md"

CONFIDENCE_SCORE = {
    "high": 1.00,
    "medium": 0.70,
    "low": 0.40,
}

SOURCE_TYPE_WEIGHT = {
    "survey": 1.00,
    "regulatory_report": 0.85,
    "operator_report": 0.75,
    "policy": 0.65,
    "product_launch": 0.65,
    "infrastructure": 0.65,
    "milestone": 0.65,
}

INDICATOR_LABELS = {
    "ACC_OWNERSHIP": "Account ownership",
    "ACC_MM_ACCOUNT": "Mobile money account ownership",
    "USG_DIGITAL_PAYMENT": "Digital payment adoption",
    "ACC_OWNERSHIP_FEMALE": "Female account ownership",
    "ACC_OWNERSHIP_MALE": "Male account ownership",
    "USG_DIGITAL_PAYMENT_FEMALE": "Female digital payment adoption",
    "USG_DIGITAL_PAYMENT_MALE": "Male digital payment adoption",
    "ACC_TELEBIRR_USERS": "Telebirr registered users",
    "ACC_MPESA_USERS": "M-Pesa registered users",
    "USG_P2P_TRANSFER": "P2P transfer value",
    "USG_ATM_WITHDRAWAL": "ATM cash withdrawal value",
    "INFRA_4G_COVERAGE": "4G coverage",
    "INFRA_AGENT_DENSITY": "Mobile money agent density",
}

PILLAR_LABELS = {
    "access": "Access",
    "usage": "Usage",
    "policy": "Policy",
}


def _year_fraction(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return dates.dt.year + (dates.dt.dayofyear - 1) / 365.25


def _coalesce_label(row: pd.Series) -> str:
    code = row.get("indicator_code")
    if code in INDICATOR_LABELS:
        return INDICATOR_LABELS[code]
    text = row.get("indicator") or row.get("value_text") or row.get("category")
    return str(text) if pd.notna(text) and str(text).strip() else str(code)


def _nearest_prior_events(
    events: pd.DataFrame,
    observation_date: pd.Timestamp,
    window_months: int = 24,
) -> str:
    if events.empty or pd.isna(observation_date):
        return ""
    lower = observation_date - pd.DateOffset(months=window_months)
    prior = events[
        (events["observation_date"] <= observation_date)
        & (events["observation_date"] >= lower)
    ].sort_values("observation_date")
    return "; ".join(prior["indicator_code"].dropna().astype(str).tolist())


def enrich_unified_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return enriched unified data plus a row-level metadata table."""
    df = raw_df.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["observation_year"] = df["observation_date"].dt.year
    df["year_fraction"] = _year_fraction(df["observation_date"]).round(3)
    df["indicator_label"] = df.apply(_coalesce_label, axis=1)
    df["pillar_label"] = df["pillar"].map(PILLAR_LABELS).fillna("")
    df["confidence_score"] = (
        df["confidence"].astype(str).str.lower().map(CONFIDENCE_SCORE).fillna(0.50)
    )
    df["source_type_weight"] = (
        df["source_type"].astype(str).str.lower().map(SOURCE_TYPE_WEIGHT).fillna(0.60)
    )
    df["source_quality_score"] = (
        df["confidence_score"] * df["source_type_weight"]
    ).round(3)
    df["source_url_present"] = df["source_url"].fillna("").astype(str).str.startswith("http")
    df["enrichment_version"] = VERSION
    df["is_enriched_record"] = True

    observations = df[df["record_type"] == "observation"].copy()
    events = df[df["record_type"] == "event"].copy()
    observations["prior_24m_events"] = observations["observation_date"].apply(
        lambda dt: _nearest_prior_events(events, dt)
    )

    observations = observations.sort_values(["indicator_code", "observation_date"])
    observations["previous_value"] = observations.groupby("indicator_code")[
        "value_numeric"
    ].shift(1)
    observations["previous_year"] = observations.groupby("indicator_code")[
        "observation_year"
    ].shift(1)
    year_gap = observations["observation_year"] - observations["previous_year"]
    observations["change_since_previous"] = (
        observations["value_numeric"] - observations["previous_value"]
    ).round(3)
    observations["annualized_change"] = (
        observations["change_since_previous"] / year_gap.replace(0, np.nan)
    ).round(3)
    observations["growth_phase"] = pd.cut(
        observations["annualized_change"],
        bins=[-np.inf, 0, 1.5, 3.5, np.inf],
        labels=["decline", "slow", "moderate", "rapid"],
    ).astype("object").fillna("baseline")
    observations["value_share_of_target_70"] = np.where(
        observations["unit"].eq("percent"),
        (observations["value_numeric"] / 70.0).round(3),
        np.nan,
    )

    for col in [
        "prior_24m_events",
        "previous_value",
        "previous_year",
        "change_since_previous",
        "annualized_change",
        "growth_phase",
        "value_share_of_target_70",
    ]:
        df[col] = np.nan if col != "prior_24m_events" and col != "growth_phase" else ""
    df.loc[observations.index, observations.columns] = observations

    metadata = _build_metadata(df)
    return df, metadata


def _build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        (
            "observation_year",
            "Extracted year from observation_date for grouping and plotting.",
            "source observation_date",
            "high",
        ),
        (
            "year_fraction",
            "Continuous year used for trend and event overlay charts.",
            "source observation_date",
            "high",
        ),
        (
            "indicator_label",
            "Human-readable indicator names for notebooks and dashboard charts.",
            "internal mapping from indicator_code",
            "high",
        ),
        (
            "confidence_score",
            "Numeric score mapped from high/medium/low confidence.",
            "source confidence",
            "high",
        ),
        (
            "source_quality_score",
            "Confidence score multiplied by source-type weight.",
            "source confidence and source_type",
            "medium",
        ),
        (
            "annualized_change",
            "Per-year change from the previous observation for the same indicator.",
            "source value_numeric and observation_date",
            "medium",
        ),
        (
            "prior_24m_events",
            "Events occurring in the 24 months before each observation.",
            "event records in unified dataset",
            "medium",
        ),
        (
            "value_share_of_target_70",
            "Percent indicators expressed as progress toward the 70% 2030 access target.",
            "NFIS-II target encoded in target records",
            "medium",
        ),
    ]
    first_url = (
        df["source_url"].dropna().astype(str).loc[lambda s: s.str.startswith("http")]
    )
    default_url = first_url.iloc[0] if not first_url.empty else ""
    rows = []
    for field, rationale, evidence_basis, confidence in fields:
        rows.append(
            {
                "enrichment_version": VERSION,
                "field": field,
                "source_url": default_url,
                "confidence": confidence,
                "evidence_basis": evidence_basis,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def write_enrichment_log(metadata: pd.DataFrame, enriched: pd.DataFrame) -> None:
    """Write a versioned Markdown data enrichment log."""
    counts = enriched["record_type"].value_counts().to_dict()
    obs = enriched[enriched["record_type"] == "observation"]
    indicators = ", ".join(sorted(obs["indicator_code"].dropna().unique()))
    lines: list[str] = [
        "# Data Enrichment Log",
        "",
        f"**Version:** {VERSION}",
        "**Project:** Ethiopia Financial Inclusion Forecasting",
        "**Generated by:** `src/data_enrichment.py`",
        "**Input:** `data/raw/ethiopia_fi_unified_data.csv`",
        "**Output:** `data/processed/ethiopia_fi_enriched.csv`",
        "",
        "## Summary",
        "",
        f"- Total records: {len(enriched)}",
        f"- Record counts: {counts}",
        f"- Observation indicators: {indicators}",
        "- Core period covered for Ethiopia FI dynamics: 2011-2024",
        "",
        "## Versioned Enrichment Fields",
        "",
        "| version | field | source_url | confidence | rationale |",
        "|---|---|---|---|---|",
    ]
    for row in metadata.itertuples(index=False):
        lines.append(
            f"| {row.enrichment_version} | {row.field} | {row.source_url} | "
            f"{row.confidence} | {row.rationale} |"
        )

    lines.extend(
        [
            "",
            "## Source Audit Rules",
            "",
            "- `source_url_present` is true only when the source URL starts with `http`.",
            "- `confidence_score` maps high/medium/low to 1.00/0.70/0.40.",
            "- `source_quality_score` combines confidence with source type weights; survey sources carry the highest weight.",
            "- `annualized_change` is calculated only within each indicator's own observed series.",
            "- `prior_24m_events` supports event overlay plots without asserting causality.",
            "",
            "## Row-Level Source Register",
            "",
            "| id | record_type | indicator_code | source_url | confidence | rationale |",
            "|---|---|---|---|---|---|",
        ]
    )
    source_rows = enriched[
        enriched["record_type"].isin(["observation", "event", "target"])
    ].sort_values(["record_type", "id"])
    for row in source_rows.itertuples(index=False):
        rationale = str(getattr(row, "notes", "") or getattr(row, "original_text", ""))
        rationale = rationale.replace("|", "/").replace("\n", " ")
        if len(rationale) > 140:
            rationale = rationale[:137] + "..."
        lines.append(
            f"| {row.id} | {row.record_type} | {row.indicator_code} | "
            f"{row.source_url} | {row.confidence} | {rationale} |"
        )

    lines.extend(
        [
            "",
            "## Rationale",
            "",
            "The enriched dataset keeps the raw unified schema intact while adding fields needed for reproducible 2011-2024 exploratory analysis: time indexing, quality scoring, event proximity, and trend-change calculations. These transformations make the EDA notebook traceable back to source rows, source URLs, confidence levels, and rationale.",
        ]
    )
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_enriched_dataset(
    raw_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = ENRICHED_DATA_PATH,
) -> pd.DataFrame:
    """Load raw unified data, enrich it, and write processed outputs."""
    raw = load_unified_data(raw_path)
    enriched, metadata = enrich_unified_data(raw)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    metadata.to_csv(METADATA_PATH, index=False)
    write_enrichment_log(metadata, enriched)
    return enriched


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    enriched = generate_enriched_dataset()
    print(f"Wrote {len(enriched)} records to {ENRICHED_DATA_PATH}")
    print(f"Wrote enrichment metadata to {METADATA_PATH}")
    print(f"Wrote versioned log to {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
