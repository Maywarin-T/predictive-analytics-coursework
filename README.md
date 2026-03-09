# Predicting Household Purchasing Power from Spending Patterns

MSIN0097 Predictive Analytics — Individual Coursework

## Overview

This project predicts a UK household's purchasing power tier (equivalised income quintile, Q1-Q5) from expenditure patterns and demographic features, using the Living Costs and Food Survey (LCFS) 2021-2023 (14,294 households).

**Equivalised income** adjusts raw household income by the OECD modified equivalence scale to account for household size and composition, providing a fairer measure of purchasing power across different household types.

Policy applications include proactive welfare targeting, cost-of-living monitoring, survey income imputation, and tax fraud detection — if a household's predicted purchasing power from spending substantially exceeds their declared income, this flags potential underreporting to HMRC.

## Project Structure

```
├── MSIN0097_code_repository.ipynb # Main analysis notebook (6 sections)
├── MSIN0097_report.pdf            # Final coursework report
├── src/
│   ├── data_loader.py             # Data loading, merging, and equivalised income computation
│   ├── preprocessing.py           # Target creation, preprocessing pipeline, splitting, PCA
│   └── evaluation.py              # Metrics, confusion matrix, calibration, feature importance
├── tests/
│   └── test_pipeline.py           # Pipeline validation tests (leakage, splitting, preprocessing)
├── data/                          # Raw LCFS data (not included — see Data Access below)
├── outputs/
│   ├── figures/                   # Saved plots
│   └── models/                    # Saved model artefacts (.pkl)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup and Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/Maywarin-T/predictive-analytics-coursework.git
cd predictive-analytics-coursework
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the data

The raw LCFS data **cannot be redistributed** under the UK Data Service End User Licence Agreement. To reproduce the analysis, you must download the data yourself:

1. Visit the **Living Costs and Food Survey series** on the UK Data Service:
   [https://datacatalogue.ukdataservice.ac.uk/series/series/2000028#access-data](https://datacatalogue.ukdataservice.ac.uk/series/series/2000028#access-data)

2. Register / log in to the UK Data Service (free for UK academic users)

3. Download the following studies in **TAB** format:
   | Study | Year | DOI |
   |-------|------|-----|
   | UKDA-9123 | 2021-2022 | https://doi.org/10.5255/UKDA-SN-9123-3 |
   | UKDA-9335 | 2022-2023 | https://doi.org/10.5255/UKDA-SN-9335-4 |
   | UKDA-9468 | 2023-2024 | https://doi.org/10.5255/UKDA-SN-9468-1 |

4. From each download, extract the **household-level** file (`dvhh_ukanon*.tab`) and place it in the `data/` directory:
   ```
   data/lcfs_2021_dvhh_ukanon.tab
   data/dvhh_ukanon_2022.tab
   data/dvhh_ukanon_v2_2023.tab
   ```

### 5. Run the notebook

```bash
jupyter notebook MSIN0097_code_repository.ipynb
```

Run all cells in order. All outputs (charts, tables, model results) are saved in the `.ipynb` file for review without re-running.

### 6. Run tests

```bash
pytest tests/test_pipeline.py -v
```

## Models Compared

| Model | Role |
|-------|------|
| Logistic Regression | Linear baseline |
| PCA + Logistic Regression | Dimensionality reduction variant (rejected) |
| Random Forest | Bagging tree ensemble |
| Gradient Boosting | Boosting ensemble (tuned via RandomizedSearchCV) |
| Neural Network (Keras) | Deep learning (selected model) |
| SVM (RBF kernel) | Non-linear comparison |

## Key Results

- **Best model**: Neural Network — selected on validation macro F1
- **Primary metric**: Macro F1 score (treats all quintiles equally)
- **Features**: 35 selected from 62 candidates (52 original + 10 engineered spending-share ratios) via importance ranking
- **Temporal split**: 2021-2022 for training/validation, 2023 for testing
- **Leakage prevention**: 227 income-derived variables explicitly excluded from features
- **Error patterns**: Adjacent quintiles (Q2-Q4) are most confused; extreme quintiles (Q1, Q5) are predicted most accurately

## Citation

Office for National Statistics, Department for Environment, Food and Rural Affairs. (2025). *Living Costs and Food Survey*. UK Data Service. Series: 2000028. Available at: https://datacatalogue.ukdataservice.ac.uk/series/series/2000028
