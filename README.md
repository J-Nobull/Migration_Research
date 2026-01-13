# Migration Analysis Project Pipeline

Modular pipeline for analyzing U.S. county-level migration patterns and housing incentive programs (2011-2021).

## Overview
This project examines the relationship between housing incentive programs and county-level migration using data from federal sources including IRS, Census Bureau, BEA, BLS, and USDA.

## Requirements to run:
    1) Three API keys (see .env.example for links and directions)
    2) 16 data files downloaded to CWD (copies provided in data/raw/)
(Built on Python 3.13 )

## Project Structure
```
migration-research/
├── py_docs                   # .ipynb and .py versions of this code
├── .env.example              # API key template
├── .gitignore
├── data_dictionary.md        # Complete variable definitions and sources
├── LICENSE
├── main.py                   # Pipeline controller
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration & API keys
├── data/
│   ├── raw/                  # 16 downloaded files 
│   └── processed/            # Generated files (gitignored)
├── src/
│   ├── acquisition/          # API data collection
│   ├── preprocessing/        # Data cleaning
│   ├── features/             # Feature engineering
│   ├── models/               # Statistical models
│   ├── analysis/             # Results & visualizations
│   └── utils/                # Helper functions
├── tests/
│   └── test_data.py          # Validation tests
└── outputs/
    ├── figures/              # Visualizations
    ├── tables/               # Result tables
    └── reports/              # Diagnostic reports
```

## Installation

1. **Clone repository**
```bash
git clone https://github.com/J-Nobull/Migration_Research.git
cd Migration-Research
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API keys**
Get API keys:
- BEA: https://apps.bea.gov/API/signup/
- Census: https://api.census.gov/data/key_signup.html
- BLS: https://data.bls.gov/registrationEngine/

```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Ensure you have the required downloaded files**

Ensure these 16 files are in `data/raw/`:
- `Incentives.csv`
- `countyinflow1112.csv` through `countyinflow2122.csv` (11 IRS files)
- `ruralurbancodes2013.xls` (USDA)
- `Ruralurbancontinuumcodes2023.csv` (USDA)
- `erscountytypology2015edition.csv` (USDA)
- `natamenf_1_.xls` (USDA)

Sources:
- Incentives: https://github.com/J-Nobull/Migration_Research/data/raw/
- IRS: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- USDA: https://www.ers.usda.gov/data-products/

## Usage

### Full Pipeline
```bash
python main.py
```

### Step-by-Step Execution
```bash
python main.py --step 1    # Data acquisition
python main.py --step 2    # Preprocessing
python main.py --step 3    # Feature engineering
python main.py --step 4    # Statistical modeling
python main.py --step 5    # Analysis & visualization
```

### Skip Validation Tests
```bash
python main.py --skip-tests
```

## Pipeline Steps

**Step 1: Data Acquisition**
- Fetch BEA (PCI, GDP, RPP)
- Fetch Census (demographics, housing)
- Fetch BLS (unemployment)
- Load IRS migration files
- Load USDA classifications
- Load Incentives

**Step 2: Preprocessing**
- Clean Census data
- Standardize FIPS codes
- Handle missing values
- Remap historical FIPS changes

**Step 3: Feature Engineering**
- Create percentage variables
- Compute derived features
- Merge datasets into panel

**Step 4: Statistical Modeling**
- Gravity model
- Panel fixed effects
- Difference-in-differences
- Dynamic panel models

**Step 5: Analysis & Visualization**
- Generate figures
- Create results tables
- Produce diagnostic reports

## Output Files

**Data:**
- `data/processed/full_panel.csv` - Complete merged dataset
- `data/processed/BEA_import.csv` - Economic indicators
- `data/processed/Census_import.csv` - Demographics
- `data/processed/IRS_gravity.csv` - Migration flows
- `data/processed/IRS_panel.csv` - For integration with other import files

**Results:**
- `outputs/tables/MODEL-*.csv` - Regression results
- `outputs/tables/Hypothesis*.csv` - Hypothesis tests
- `outputs/figures/FIG*.png` - Visualizations

**Reports:**
- `outputs/reports/KEY_FINDINGS_SUMMARY.txt`
- `outputs/reports/DIAGNOSTIC_SUMMARY.txt`

## Citation
Mendoza, Brenda; Noble, Jason; & Selva, Carlos. (2025).
National University.
The Impact of Housing Policies and Incentives on Migration: 
A Multilevel Analysis of Sustained Migration Between Counties Across the United States. 

## Contact
For questions, open an issue on GitHub
Or email jason.leo.noble@gmail.com
