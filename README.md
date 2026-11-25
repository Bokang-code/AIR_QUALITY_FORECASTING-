# Air Quality Forecasting & Environmental Intelligence

An MSc Data Science and Artificial Intelligence project exploring short-term PM2.5 forecasting for Chengdu, China. The repository contains the forecasting research workflow, model-development code, explainability utilities, and a Streamlit dashboard.

## Project overview

The project brings together:
- **Data preparation:** dataset audits, missingness analysis, feature construction, and validation.
- **Forecasting:** baseline models and machine-learning experiments across temporal, meteorological, and spatial feature sets.
- **Explainability:** SHAP-based explanations for model behaviour and individual predictions.
- **Application:** a Streamlit dashboard for exploring station observations, forecasts, and their main drivers.

> This repository contains research code and application components. Forecast quality and health interpretations should be assessed using the accompanying evaluation methodology; predictions are not medical advice.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app/` | Streamlit application and dashboard helper functions |
| `src/` | Shared explainability and PM2.5 interpretation utilities |
| `src/v4/` | Current v4 data-audit, feature-engineering, validation, baseline, and model-training scripts |
| `data/` | Source datasets used in the project |
| `configs/` | Project configuration |
| `reports/v4/` | Dataset audits, availability and missingness summaries, and experiment outputs |
| `styles/` | Dashboard styling |
| `archive/` | Earlier research iterations, notebooks, tests, and v2/v3 code |

The v4 workflow is the current development line. The `archive/` directory is retained for research traceability rather than being part of the main application path.

## Technology

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, and Jupyter.

## Running the dashboard

Install the project dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The dashboard entry point is:

```bash
streamlit run app/app.py
```

**Required generated artefacts:** the dashboard reads the processed feature dataset from `data/processed/v4/chengdu_features_v4.csv` and trained model files from `models/v4/temporal_meteorology_copollutants/`. These generated artefacts are not tracked in the repository, so they must be created or supplied before the dashboard can run successfully.

## Research workflow

The v4 scripts are organised around these stages:

1. Audit source data and forecasting/predictor availability.
2. Summarise missing observations and gaps.
3. Build the Chengdu master dataset and spatial features.
4. Engineer and validate forecasting features.
5. Establish baseline performance and train forecasting models.
6. Review model outputs and explainability reports.

See the scripts in `src/v4/` and the corresponding outputs in `reports/v4/` for implementation details.

## Notes on data and reproducibility

- Raw source files are retained under `data/`; check applicable source licences and attribution requirements before redistributing them.
- Generated datasets, model artefacts, environments, caches, and secrets should not be committed.
- The exact results depend on the source-data versions, feature-generation process, model configuration, and evaluation split. Consult the experiment outputs and dissertation for reported metrics and methodology.

## Academic context

Developed as part of an MSc Data Science and Artificial Intelligence dissertation.
