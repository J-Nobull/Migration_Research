# Project Structure Guide

After downloading all files, organize them into this directory structure:

```
migration-research/
│
├── .env.example              # Rename downloaded file
├── .env                      # Copy from .env.example and add keys
├── .gitignore
├── data_dictionary.md
├── LICENSE
├── PROJECT_STRUCTURE.md      # This file
├── README.md
├── requirements.txt
├── main.py
│
├── config/
│   ├── __init__.py           # Rename: config__init__.py
│   └── settings.py           # Rename: config_settings.py
│
├── data/
│   ├── raw/                  # Create this folder
│   │   ├── Incentives.csv
│   │   ├── countyinflow1112.csv
│   │   ├── ... (10 more IRS files)
│   │   ├── erscountytypology2015edition.csv
│   │   ├── Ruralurbancontinuumcodes2023.csv
│   │   ├── ruralurbancodes2013.xls
│   │   └── natamenf_1_.xls
│   └── processed/            # Auto-created by pipeline
│
├── src/
│   ├── __init__.py           # Rename: src__init__.py
│   │
│   ├── acquisition/
│   │   ├── __init__.py       # Rename: src_acquisition__init__.py
│   │   ├── bea.py            # Rename: src_acquisition_bea.py
│   │   ├── census.py         # Rename: src_acquisition_census.py
│   │   ├── bls.py            # Rename: src_acquisition_bls.py
│   │   ├── irs.py            # Rename: src_acquisition_irs.py
│   │   └── usda.py           # Rename: src_acquisition_usda.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py       # Rename: src_preprocessing__init__.py
│   │   └── clean_data.py     # Rename: src_preprocessing_clean_data.py
│   │
│   ├── features/
│   │   ├── __init__.py       # Rename: src_features__init__.py
│   │   └── engineer_features.py  # Rename: src_features_engineer_features.py
│   │
│   ├── validation/
│   │   ├── __init__.py       # Rename: src_validation__init__.py
│   │   └── feature_selection.py  # Rename: src_validation_feature_selection.py
│   │
│   ├── models/
│   │   ├── __init__.py       # Rename: src_models__init__.py
│   │   └── run_models.py     # Rename: src_models_run_models.py
│   │
│   ├── analysis/
│   │   ├── __init__.py       # Rename: src_analysis__init__.py
│   │   ├── feature_selection # Rename: src_analysis_feature_selection.py
│   │   └── create_outputs.py # Rename: src_analysis_create_outputs.py
│   │
│   └── utils/
│       ├── __init__.py       # Rename: src_utils__init__.py
│       └── helpers.py        # Rename: src_utils_helpers.py
│
├── tests/
│   ├── __init__.py           # Rename: tests__init__.py
│   └── test_data.py          # Rename: tests_test_data.py
│
└── outputs/                  # Auto-created by pipeline
    ├── figures/
    ├── tables/
    └── reports/
```

## Quick Setup Commands

**1. Create directory structure:**
```bash
mkdir -p migration-research/{config,data/{raw,processed},src/{acquisition,analysis,features,models,preprocessing,utils,validation},tests,outputs/{figures,tables,reports}}
```

**2. Move and rename files:**

Download all files to a temporary folder, then:

```bash
# Root files (no rename needed)
mv .env.example .gitignore LICENSE README.md QUICKSTART.md requirements.txt main.py migration-research/

# Config files
mv config__init__.py migration-research/config/__init__.py
mv config_settings.py migration-research/config/settings.py

# Src files
mv src__init__.py migration-research/src/__init__.py
mv src_utils__init__.py migration-research/src/utils/__init__.py
mv src_utils_helpers.py migration-research/src/utils/helpers.py

# Acquisition files
mv src_acquisition__init__.py migration-research/src/acquisition/__init__.py
mv src_acquisition_bea.py migration-research/src/acquisition/bea.py
mv src_acquisition_census.py migration-research/src/acquisition/census.py
mv src_acquisition_bls.py migration-research/src/acquisition/bls.py
mv src_acquisition_irs.py migration-research/src/acquisition/irs.py
mv src_acquisition_usda.py migration-research/src/acquisition/usda.py

# Preprocessing files
mv src_preprocessing__init__.py migration-research/src/preprocessing/__init__.py
mv src_preprocessing_clean_data.py migration-research/src/preprocessing/clean_data.py

# Features files
mv src_features__init__.py migration-research/src/features/__init__.py
mv src_features_engineer_features.py migration-research/src/features/engineer_features.py

# Models files
mv src_models__init__.py migration-research/src/models/__init__.py
mv src_models_run_models.py migration-research/src/models/run_models.py

# Analysis files
mv src_analysis__init__.py migration-research/src/analysis/__init__.py
mv src_analysis_create_outputs.py migration-research/src/analysis/create_outputs.py

# Test files
mv tests__init__.py migration-research/tests/__init__.py
mv tests_test_data.py migration-research/tests/test_data.py
```

**3. Setup environment:**
```bash
cd migration-research
cp .env.example .env
# Edit .env and add your API keys
pip install -r requirements.txt
```

**4. Download required data files to `data/raw/`**

**5. Verify setup:**
```bash
python -m tests.test_data
```

**6. Run pipeline:**
```bash
python main.py
```

## Notes

- Downloaded filenames use underscores instead of directory separators
- You must rename them to match the structure above
- The pipeline creates `data/processed/` and `outputs/` automatically
- Place all 16 manual download files in `data/raw/`
