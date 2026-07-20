from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_enrichment import enrich_unified_data, generate_enriched_dataset
from data_loader import load_unified_data


RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "ethiopia_fi_unified_data.csv"


def test_enrich_unified_data_adds_analysis_fields():
    raw = load_unified_data(RAW_PATH)
    enriched, metadata = enrich_unified_data(raw)

    required = {
        "observation_year",
        "year_fraction",
        "indicator_label",
        "confidence_score",
        "source_quality_score",
        "source_url_present",
        "enrichment_version",
        "annualized_change",
        "prior_24m_events",
    }
    assert required.issubset(enriched.columns)
    assert not metadata.empty
    assert {"source_url", "confidence", "rationale"}.issubset(metadata.columns)


def test_account_ownership_growth_metrics_are_computed():
    raw = load_unified_data(RAW_PATH)
    enriched, _ = enrich_unified_data(raw)
    access = enriched[
        (enriched["record_type"] == "observation")
        & (enriched["indicator_code"] == "ACC_OWNERSHIP")
    ].sort_values("observation_date")

    latest = access.iloc[-1]
    assert latest["observation_year"] == 2024
    assert latest["change_since_previous"] == 3.0
    assert latest["annualized_change"] == 1.0
    assert latest["growth_phase"] == "slow"


def test_generate_enriched_dataset_writes_csv_and_log(tmp_path):
    output_path = tmp_path / "ethiopia_fi_enriched.csv"
    enriched = generate_enriched_dataset(RAW_PATH, output_path)

    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert len(written) == len(enriched)
    source_backed = written[written["record_type"].isin(["observation", "event", "target"])]
    assert source_backed["source_url_present"].all()
