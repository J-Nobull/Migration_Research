"""Data cleaning and preprocessing."""
import pandas as pd
import numpy as np
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.helpers import standardize_fips, define_cols, remap_fips_changes, filter_dataframe, save_point

def clean_irs_data():
    """Clean IRS migration data."""
    print("Cleaning IRS data...")
    irs = pd.read_csv(PROCESSED_DATA_DIR / 'IRS_panel.csv')
    
    # Fill specific missing values for FIPS 20157
    irs.loc[irs['FIPS'] == '20157', ['move_out', 'agi_out', 'move_net', 'agi_net']] = \
        irs.loc[irs['FIPS'] == '20157', ['move_out', 'agi_out', 'move_net', 'agi_net']].fillna(0)
    
    save_point(irs, 'IRS_panel_clean.csv', 'IRS cleaned')
    return irs

def clean_usda_data():
    """Clean USDA data."""
    print("Cleaning USDA data...")
    
    amenities = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_Amenities.csv')
    amenities['Amenity_scale'] = amenities['Amenity_scale'].fillna(0)
    amenities.loc[amenities['FIPS'] == '12025', 'FIPS'] = '12086'  # Dade -> Miami-Dade
    save_point(amenities, 'USDA_Amenities_clean.csv', 'Amenities cleaned')
    
    rucc_2023 = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_RUCC_2023.csv')
    rucc_2023 = rucc_2023[~rucc_2023['FIPS'].astype(str).str.startswith('091')].copy()
    save_point(rucc_2023, 'USDA_RUCC_2023_clean.csv', 'RUCC 2023 cleaned')
    
    return amenities, rucc_2023

def clean_bea_data():
    """Clean BEA data."""
    print("Cleaning BEA data...")
    
    pci = pd.read_csv(PROCESSED_DATA_DIR / 'BEA_PCI.csv')
    pci['FIPS'] = pci['FIPS'].replace({'15901': '15009'})
    pci = pci[pci['FIPS'] != '55901']
    pci = pci[~pci['FIPS'].astype(str).str.startswith('519')].copy()
    save_point(pci, 'BEA_PCI_clean.csv', 'PCI cleaned')
    
    gdp = pd.read_csv(PROCESSED_DATA_DIR / 'BEA_GDP.csv')
    gdp['FIPS'] = gdp['FIPS'].replace({'15901': '15009'})
    gdp = gdp[~gdp['FIPS'].astype(str).str.startswith('519')].copy()
    save_point(gdp, 'BEA_GDP_clean.csv', 'GDP cleaned')
    
    return pci, gdp

def clean_census_data():
    """Clean Census data."""
    print("Cleaning Census data...")
    
    census = pd.read_csv(PROCESSED_DATA_DIR / 'Census_import.csv')
    census = standardize_fips(census)
    census = remap_fips_changes(census, fips_cols=['FIPS'])
    census = filter_dataframe(census, 'Census Data')
    census = census.replace(-666666666, np.nan)
    
    # Ensure median columns are Int64
    median_vars = ['median_property_taxes', 'median_hh_income', 'median_home_value']
    census[median_vars] = census[median_vars].apply(
        lambda col: pd.to_numeric(col, errors='coerce').astype('Int64'))
    
    # Helper functions for imputation
    def avg_years(df, fips, col, y1, y2):
        vals = df.loc[(df['FIPS'] == fips) & df['Year'].isin([y1, y2]), col]
        return int(round(vals.mean())) if vals.notna().all() else np.nan
    
    def fill(df, fips, col, year, value):
        df.loc[(df['FIPS'] == fips) & (df['Year'] == year), col] = value
    
    # Impute FIPS 35039, year 2018
    for col in census.columns:
        if census.loc[(census['FIPS'] == '35039') & (census['Year'] == 2018), col].isna().any():
            val = avg_years(census, '35039', col, 2017, 2019)
            fill(census, '35039', col, 2018, val)
    
    # Impute 2015 gaps in median variables
    for col in median_vars:
        missing_fips = census.loc[(census['Year'] == 2015) & (census[col].isna()), 'FIPS'].unique()
        for fips in missing_fips:
            val = avg_years(census, fips, col, 2014, 2016)
            if pd.notna(val):
                fill(census, fips, col, 2015, val)
    
    census = define_cols(census)
    save_point(census, 'Census_clean.csv', f"{len(census):,} cleaned observations")
    
    return census

def load_incentives():
    """Load housing incentive programs data."""
    print("Loading housing incentives...")
    
    incentives = pd.read_csv(RAW_DATA_DIR / 'Incentives.csv', dtype={'FIPS': str})
    incentives = incentives[['FIPS', 'Year', 'has_incentive', 'Incentive_CAT', 'COVID_program']]
    incentives = standardize_fips(incentives)
    incentives = filter_dataframe(incentives, 'Housing Incentive Programs')
    
    save_point(incentives, 'Incentives_clean.csv', f"{incentives['FIPS'].nunique()} counties with programs")
    
    return incentives

def run():
    """Main preprocessing function."""
    print("="*49)
    print("DATA CLEANING & PREPROCESSING")
    print("="*49 + "\n")
    
    # Clean all datasets
    census = clean_census_data()
    irs = clean_irs_data()
    amenities, rucc_2023 = clean_usda_data()
    pci, gdp = clean_bea_data()
    incentives = load_incentives()
    
    print("\nPreprocessing complete\n")

if __name__ == '__main__':
    run()
