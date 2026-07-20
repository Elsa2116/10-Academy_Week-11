# Notebooks

This directory contains analysis notebooks for the Ethiopia Financial Inclusion Forecasting project.

## Notebooks

| Notebook | Task | Description |
|----------|------|-------------|
| `task1_data_exploration_enrichment.ipynb` | Task 1 | Load, explore, and enrich the unified dataset |
| `task2_eda.ipynb` | Task 2 | Exploratory data analysis with visualizations |
| `eda_ethiopia_2011_2024.ipynb` | EDA | Dedicated 2011-2024 visual analysis with account ownership, mobile money, event overlay timeline, and five-plus documented insights |
| `task3_event_impact_modeling.ipynb` | Task 3 | Event-indicator association matrix and impact estimation |
| `task4_forecasting.ipynb` | Task 4 | Forecasting Access and Usage for 2025–2027 |

## Running Notebooks

```bash
cd ethiopia-fi-forecast
pip install -r requirements.txt
jupyter notebook notebooks/
```

Regenerate the processed enriched dataset before EDA if the raw CSV changes:

```bash
python src/data_enrichment.py
```

The dedicated EDA notebook writes supporting PNG plots to `reports/figures/`.
