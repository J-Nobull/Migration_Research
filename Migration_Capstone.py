# ===========================================================
# SECTION 1: CONFIGURATION & ENVIRONMENT
# ===========================================================
import pandas as pd
import numpy as np
import os
import requests
import time
import warnings
from io import BytesIO
from census import Census
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import geopandas as gpd
from datetime import datetime
from statsmodels.regression.linear_model import OLS
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels import PooledOLS
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.formula.api as smf
from libpysal.weights import Queen, KNN, DistanceBand
from esda.moran import Moran
from linearmodels.iv import IV2SLS
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from graphviz import Digraph
from IPython.display import Image, display
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
# %matplotlib inline
print('\nEnvironment Ready')
 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
 
# API key official sources
# BEA:      https://apps.bea.gov/API/signup/
# Census:   https://api.census.gov/data/key_signup.html
# BLS:      https://data.bls.gov/registrationEngine/

# API Keys - REPLACE 'Key-Here' WITH ACTUAL KEYS
#API_KEY_BEA     = 'Key-Here'
#API_KEY_CENSUS  = 'Key-Here'
#API_KEY_BLS     = 'Key-Here'
 
# Validate API keys
if API_KEY_BEA == 'Key-Here':
    raise ValueError("ERROR: Replace API_KEY_BEA with your actual Bureau of Economic Analysis API key")
if API_KEY_CENSUS == 'Key-Here':
    raise ValueError("ERROR: Replace API_KEY_CENSUS with your actual U.S. Census Bureau API key")
if API_KEY_BLS == 'Key-Here':
    raise ValueError("ERROR: Replace API_KEY_BLS with your actual Bureau of Labor Statistics API key")
 
# Study parameters
YEARS = list(range(2011, 2022))  # 2011-2021 inclusive
CWD = Path.cwd()
 
print(f"Configuration loaded successfully")
print(f"Study period: {YEARS[0]}-{YEARS[-1]}")
print(f"Working directory: {CWD}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
 
# ===========================================================
# SECTION 2: REQUIRED FILES
# ===========================================================
'''
The following 16 files must be manually downloaded from source websites
to the working directory before running this script.
All other files are generated via API calls.
 
***** MANUAL DOWNLOADS REQUIRED *****
 
1. Incentives.csv
   Source:
# https://github.com/J-Nobull/Migration_Research/blob/main/Data_files/Incentives.csv
   Contains: Housing incentive program details for 2016-2021
 
2-12. IRS County-to-County Migration Files (11 years × inflow format)
   Source: https://www.irs.gov/statistics/soi-tax-stats-migration-data
   Files: countyinflow1112.csv through countyinflow2122.csv
   Note: Inflow format selected because analysis examines incentive programs
         as attraction mechanisms (pull factors) rather than out-migration
 
13. ruralurbancodes2013.xls
14. Ruralurbancontinuumcodes2023.csv
    Source: https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/
 
15. erscountytypology2015edition.csv
    Source: https://www.ers.usda.gov/data-products/county-typology-codes/
 
16. natamenf_1_.xls
    Source: https://www.ers.usda.gov/data-products/natural-amenities-scale/
 
API-GENERATED FILES (created by this script):
- BEA_RPP_Metro.csv
- BEA_RPP_NONmetro.csv
- BEA_PCI.csv
- BEA_GDP.csv
- CENSUS_import.csv
- BLS_import.csv
- IRS_gravity.csv
- IRS_panel.csv
- full_panel.csv
'''
# Verify required manual download files exist
REQUIRED_FILES = [
    'Incentives.csv', 'countyinflow1112.csv', 'countyinflow1213.csv',
    'countyinflow1314.csv', 'countyinflow1415.csv', 'countyinflow1516.csv',
    'countyinflow1617.csv', 'countyinflow1718.csv', 'countyinflow1819.csv',
    'countyinflow1920.csv', 'countyinflow2021.csv', 'countyinflow2122.csv',
    'ruralurbancodes2013.xls', 'Ruralurbancontinuumcodes2023.csv',
    'erscountytypology2015edition.csv', 'natamenf_1_.xls']
# IRS file list
IRS_FILES = [
    f for f in REQUIRED_FILES if f.startswith('countyinflow')]
missing_files = [
    f for f in REQUIRED_FILES if not (CWD / f).exists()]
if missing_files:
    print("ERROR: These required files are missing from CWD:")
    for f in missing_files:
        print(f"  - {f}")
    raise FileNotFoundError("Download required files before running script")
print(" All 16 required files found in working directory\n")
 
# ==========================================================
# SECTION 3: UTILITY FUNCTIONS
# ==========================================================
#   Utility 1
def standardize_fips(df, state_col=None, county_col=None, fips_col='FIPS'):
  
    df = df.copy()
    if state_col and county_col:
        df[fips_col] = (df[state_col].astype(str).str.zfill(2) +
                       df[county_col].astype(str).str.zfill(3))
    elif fips_col in df.columns:
        df[fips_col] = df[fips_col].astype(str).str.zfill(5)
    else:
        raise ValueError(
            f"Must provide either (state_col, county_col) or fips_col")
    return df
 
#   Utility 2
def define_cols(
        df, exclude_cols=['FIPS', 'origin_FIPS',
                          'state', 'county', 'STATE']):
 
    df = df.copy()
    for col in df.columns:
        if col in exclude_cols:
            continue
        elif col == 'Year':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif col in ['median_age', 'Amenity_scale', 'RPP', 'unemploy_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif df[col].dtype == 'object':
            df[col] = df[col].replace('-', np.nan)
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                if df[col].dropna().apply(float.is_integer).all():
                    df[col] = df[col].astype('Int64')
            except:
                pass
    return df
 
#   Utility 3
def remap_fips_changes(df, fips_cols=['FIPS']):
    df = df.copy()
    # Alaska (02): consolidate all boroughs/census areas
    alaska_codes = [f'02{str(i).zfill(3)}' for i in range(1, 999)]
    # Connecticut (09): planning regions → counties
    ct_remap = {
        '09110': '09003', '09120': '09001', '09130': '09007', 
        '09140': '09009', '09150': '09015', '09160': '09005', 
        '09170': '09009', '09180': '09011', '09190': '09001'} 
    for col in fips_cols:
    # Alaska
        df.loc[df[col].isin(alaska_codes), col] = '02001'
    # Connecticut
        df[col] = df[col].replace(ct_remap)
    # Drop Kalawao County, HI
        df = df[df[col] != '15005']
    # South Dakota rename
        df.loc[df[col] == '46113', col] = '46102'
    # Drop Bedford City, VA
        df = df[df[col] != '51515']
    return df
 
#   Utility 4
def filter_dataframe(df, name='DataFrame'):
    # Filter out FIPS > 56999 (Puerto Rico, US Territories)
    print(f"\n{'='*49}")
    print(f"{name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    if 'FIPS' in df.columns:
        before_count = len(df)
        df = df[df['FIPS'].str[:2] != '72'].copy()
        df = df[df['FIPS'].astype(str).str[:5].astype(
            int) < 57000].copy()
        after_count = len(df)
        if before_count != after_count:
            print(f"Removed {before_count - after_count:,} rows with FIPS > 56999")
        print(f"Unique FIPS: {df['FIPS'].nunique():,}")
    if 'Year' in df.columns:
        print(f"Years: {df['Year'].min()}-{df['Year'].max()}")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values: {missing.sum():,} ({missing.sum()/df.size*100:.1f}%)")
    # Display concise summary of DataFrame structure and content.
    print(f"{'='*49}\n")
    return df
 
#   Utility 5
def save_point(df, filename, description=""):
    # Save DataFrame checkpoint with consistent formatting.
    filepath = CWD / filename
    df.to_csv(filepath, index=False)
    print(f"Saved: {filename}")
    if description:
        print(f"  {description}")
print("5 Utility Functions Created")
 
# ==========================================================
# SECTION 4: DATA ACQUISITION
# ==========================================================
print("="*49)
print("SECTION 4: DATA ACQUISITION")
print("="*49 + "\n")
 
# ----------------------------------------------------------
# 4.1: Bureau of Economic Analysis (BEA) Data
# ----------------------------------------------------------
print("4.1: Acquiring BEA data via API...")
 
def get_bea_data(dataset, table, line_code, geo_type, years):
    # Fetch BEA data via API.
    base_url = "https://apps.bea.gov/api/data"
 
    results = []
    for year in years:
        params = {
            'UserID': API_KEY_BEA,
            'method': 'GetData',
            'datasetname': dataset,
            'TableName': table,
            'LineCode': line_code,
            'GeoFips': geo_type,
            'Year': year,
            'ResultFormat': 'JSON'}
 
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Add error checking for nested keys
            if 'BEAAPI' in data and 'Results' in data['BEAAPI']:
                if 'Data' in data['BEAAPI']['Results']:
                    results.extend(data['BEAAPI']['Results']['Data'])
                else:
                    print(f" ❌  Warning: No data returned for {table} year {year}")
            else:
                print(f" ❌  Warning: Invalid API response for {table} year {year}")
        else:
            print(f" ❌  Warning: API request failed (status {response.status_code}) for {table} year {year}")
    return pd.DataFrame(results) if results else pd.DataFrame()
 
# Regional Price Parities (RPP)
print("  - Regional Price Parities (RPP)...")
rpp_metro = get_bea_data('Regional', 'MARPP', '3', 'MSA', YEARS)
rpp_nonmetro = get_bea_data('Regional', 'PARPP', '3', 'PORT', YEARS)
 
if not rpp_metro.empty:
    rpp_metro = rpp_metro[[
        'GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'MSA_Code',
            'TimePeriod': 'Year',
            'DataValue': 'RPP_Metro'})
    rpp_metro = define_cols(rpp_metro, exclude_cols=['MSA_Code'])
    save_point(rpp_metro, 'BEA_RPP_Metro.csv', f"{len(rpp_metro):,}")
 
if not rpp_nonmetro.empty:
    rpp_nonmetro = rpp_nonmetro[[
        'GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'State_FIPS',
            'TimePeriod': 'Year',
            'DataValue': 'RPP_NonMetro'})
    rpp_nonmetro = define_cols(rpp_nonmetro, exclude_cols=['State_FIPS'])
    save_point(rpp_nonmetro, 'BEA_RPP_NONmetro.csv', f"{len(rpp_nonmetro):,}")
 
# Per Capita Income (PCI)
print("  - Per Capita Income...")
pci = get_bea_data('Regional', 'CAINC1', '3', 'COUNTY', YEARS)
 
if not pci.empty:
    pci = pci[[
        'GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'FIPS',
            'TimePeriod': 'Year',
            'DataValue': 'BEA_PCI'})
    pci = standardize_fips(pci)
    pci = define_cols(pci)
    pci = remap_fips_changes(pci, fips_cols=['FIPS'])
    pci = filter_dataframe(pci)
    save_point(pci, 'BEA_PCI.csv', f"{len(pci):,} county-year obs")

# Real GDP by County
print("  - Real GDP by County...")
gdp = get_bea_data('Regional', 'CAGDP1', '1', 'COUNTY', YEARS)
 
if not gdp.empty:
    gdp = gdp[[
        'GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'FIPS',
            'TimePeriod': 'Year',
            'DataValue': 'BEA_GDP'})
    gdp = standardize_fips(gdp)
    gdp = define_cols(gdp)
    gdp = remap_fips_changes(gdp, fips_cols=['FIPS'])
    gdp = filter_dataframe(gdp)
    save_point(gdp, 'BEA_GDP.csv', f"{len(gdp):,} county-year obs")
 
# Convert MSA codes to county FIPS using CBSA crosswalk
print("  - Downloading CBSA delineation crosswalk...")
cbsa_url = 'https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2013/delineation-files/list1.xls'
response = requests.get(cbsa_url)
cbsa = pd.read_excel(BytesIO(response.content), skiprows=2)
 
# Fix CBSA float columns
cbsa['FIPS State Code'] = pd.to_numeric(cbsa[
    'FIPS State Code'], errors='coerce').astype('Int64')
cbsa['FIPS County Code'] = pd.to_numeric(cbsa[
    'FIPS County Code'], errors='coerce').astype('Int64')
cbsa = standardize_fips(cbsa,
                        state_col='FIPS State Code',
                        county_col='FIPS County Code',
                        fips_col='FIPS')
cbsa.rename(columns={'CBSA Code': 'MSA_Code'}, inplace=True)
if not rpp_metro.empty:
    rpp_metro['MSA_Code'] = rpp_metro['MSA_Code'].astype(str).str.zfill(5)
    cbsa['MSA_Code'] = cbsa['MSA_Code'].astype(str).str.zfill(5)
    rpp_metro = rpp_metro.merge(cbsa[[
        'MSA_Code', 'FIPS']], on='MSA_Code', how='left')
    rpp_metro = rpp_metro[[
        'FIPS', 'Year', 'RPP_Metro']].dropna(subset=['FIPS'])
 
print(" BEA data acquisition complete\n")
 
# ----------------------------------------------------------
# 4.2: U.S. Census Bureau - American Community Survey (ACS-5)
# ----------------------------------------------------------
print("4.2: Acquiring Census ACS data via API...")
 
# ACS 5-year estimates variable list (67 variables)
ACS_VARS = {
    'B01003_001E': 'total_population',
    'B01002_001E': 'median_age',
    'B25003_001E': 'housing_total',
    'B25003_002E': 'owner_occupied',
    'B25003_003E': 'renter_occupied',
    'B19013_001E': 'median_hh_income',
    'B25077_001E': 'median_home_value',
    'B25103_001E': 'median_property_taxes',
    'B11001_002E': 'family_households',
    'B12001_001E': 'marital_total',
    'B12001_003E': 'never_married_male',
    'B12001_004E': 'now_married_male',
    'B12001_009E': 'widowed_male',
    'B12001_010E': 'divorced_male',
    'B12001_012E': 'never_married_female',
    'B12001_013E': 'now_married_female',
    'B12001_018E': 'widowed_female',
    'B12001_019E': 'divorced_female',
    'B09001_002E': 'under_18_in_hh',
    'B03002_003E': 'white',
    'B03002_004E': 'black',
    'B03002_005E': 'native',
    'B03002_006E': 'asian',
    'B03002_007E': 'pacific_islander',
    'B03002_008E': 'other_race',
    'B03002_009E': 'mixed_non_h',
    'B03002_012E': 'hispanic',
    'B15002_001E': 'education_total_sex',
    'B15002_011E': 'male_complete_hs',
    'B15002_012E': 'male_less1yr_college',
    'B15002_013E': 'male_more1yr_college',
    'B15002_014E': 'male_associates',
    'B15002_015E': 'male_bachelors',
    'B15002_016E': 'male_masters',
    'B15002_017E': 'male_professional',
    'B15002_018E': 'male_doctorate',
    'B15002_028E': 'female_complete_hs',
    'B15002_029E': 'female_less1yr_college',
    'B15002_030E': 'female_more1yr_college',
    'B15002_031E': 'female_associates',
    'B15002_032E': 'female_bachelors',
    'B15002_033E': 'female_masters',
    'B15002_034E': 'female_professional',
    'B15002_035E': 'female_doctorate',
    'B08303_002E': 'commute_less_5min',
    'B08303_003E': 'commute_5_9min',
    'B08303_004E': 'commute_10_14min',
    'B08303_005E': 'commute_15_19min',
    'B08303_006E': 'commute_20_24min',
    'B08303_007E': 'commute_25_29min',
    'B08303_008E': 'commute_30_34min',
    'B08303_009E': 'commute_35_39min',
    'B08303_010E': 'commute_40_44min',
    'B08303_011E': 'commute_45_59min',
    'B08303_012E': 'commute_60_89min',
    'B08303_013E': 'commute_90_plus_min',
    'B08137_020E': 'work_in_owned_home',
    'B08137_021E': 'work_in_rental',
    'C24060_001E': 'occupation_total',
    'C24060_002E': 'Mgmt_Biz_Sci_Arts',
    'C24060_003E': 'Services',
    'C24060_004E': 'Sales_Admin',
    'C24060_005E': 'Nat-rsrc_Constr_Maint',
    'C24060_006E': 'Prod_Transp_Mvng'}
 
def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
 
def fetch_census_batch(year: int, var_codes: list[str]) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{int(year)}/acs/acs5"
    params = {'get': ','.join(var_codes),
              'for': 'county:*',
              'key': API_KEY_CENSUS}
 
    r = requests.get(url, params=params, timeout=120)
    if r.status_code != 200:
        print(f"❌ Census HTTP {r.status_code} for {year}")
        print(r.text[:1500])
        return pd.DataFrame()
    try:
        data = r.json()
    except ValueError:
        print(f"❌ Census JSON decode error for {year}")
        print(r.text[:1500])
        return pd.DataFrame()
    if isinstance(data, dict) and 'error' in data:
        print(f"❌ Census API error for {year}: {data['error']}")
        return pd.DataFrame()
    if not data or len(data) <= 1:
        print(f"❌ Warning: Census returned no rows for {year}")
        return pd.DataFrame()
    return pd.DataFrame(data[1:], columns=data[0])
 
def download_census_acs(years: list[int] | tuple[int, ...],
                        batch_size: int = 45,
                        sleep_s: float = 0.25) -> pd.DataFrame:
    all_vars = list(ACS_VARS.keys())
    batches = _chunk_list(all_vars, batch_size)
    frames = []
    for year in years:
        print(f"Fetching ACS {year}...")
        parts = []
        for b in batches:
            dfb = fetch_census_batch(year, ['NAME'] + b)
            if dfb.empty:
                parts = []
                break
            parts.append(dfb)
            time.sleep(sleep_s)
        if not parts:
            continue
        year_df = parts[0]
        for part in parts[1:]:
            year_df = year_df.merge(
                part, on=['NAME', 'state', 'county'], how='outer')
        year_df = year_df.rename(columns=ACS_VARS)
        year_df = standardize_fips(
            year_df,
            state_col='state',
            county_col='county',
            fips_col='FIPS',)
        year_df['Year'] = int(year)
        year_df = year_df.drop(
            columns=['NAME', 'state', 'county'], errors='ignore')
        year_df = define_cols(year_df)
        frames.append(year_df)
        print(f"Saved {len(year_df):,} rows")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    keep_cols = [
        'FIPS', 'Year'] + [c for c in ACS_VARS.values() if c in out.columns]
    return out[keep_cols].copy()
 
# Fetch data
print("Downloading Census data (2011–2021)...\n")
CENSUS_DF = download_census_acs(YEARS)
if CENSUS_DF.empty:
    print("\n❌ Error: No Census data was downloaded")
else:
    CENSUS_DF.to_csv('Census_import.csv', index=False)
    print("\n" + "=" * 30)
    print("CENSUS DOWNLOAD COMPLETE")
    print("=" * 30)
    print(f"Saved {len(CENSUS_DF):,} rows")
    print(f"Counties: {CENSUS_DF['FIPS'].nunique()}")
    print(f"Years: {CENSUS_DF['Year'].min()}-{CENSUS_DF['Year'].max()}")
    print(f"Variables: {len([c for c in ACS_VARS.values() if c in CENSUS_DF.columns])}")
    print(CENSUS_DF.info())
 
# ----------------------------------------------------------
# 4.3: Bureau of Labor Statistics (BLS) - Local Area Unemployment
# ----------------------------------------------------------
print("4.3: Acquiring BLS LAUS data via API...")
 
fips_list = CENSUS_DF['FIPS'].dropna().unique()
 
def get_bls_unemployment(fips_list, years, batch_size=50, sleep_s=1.0):
# Fetch county-level annual unemployment rates (LAUS) from BLS API (batched).
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
 
    fips_list = pd.Series(fips_list).astype(str).str.zfill(5).unique().tolist()
    all_series = [f"LAUCN{fips}0000000003" for fips in fips_list]
    total_batches = (len(all_series) + (batch_size - 1)) // batch_size
    rows = []
 
    for i in range(0, len(all_series), batch_size):
        batch = all_series[i:i + batch_size]
        batch_num = i // batch_size + 1
        payload = {
            'seriesid': batch,
            'startyear': str(min(years)),
            'endyear': str(max(years)),
            'registrationkey': API_KEY_BLS,
            'annualaverage': True,}
 
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"❌ BLS HTTP {response.status_code} in batch {batch_num}/{total_batches}")
            print(response.text[:1000])
            continue
 
        data = response.json()
 
        if data.get('status') != 'REQUEST_SUCCEEDED':
            msg = data.get('message', 'Unknown error')
            print(f"❌ Batch {batch_num}/{total_batches} error: {msg}")
            continue
 
        for series in data.get('Results', {}).get('series', []):
            series_id = series.get('seriesID', "")
            fips = series_id[5:10] if len(series_id) >= 10 else None
            if not fips:
                continue
 
            for item in series.get('data', []):
                if item.get('period') == 'M13':  # annual average
                    rows.append({
                        'FIPS': fips,
                        'Year': int(item['year']),
                        'unemploy_rate': pd.to_numeric(item.get(
                            'value'), errors='coerce'),})
 
        print(f"  Batch {batch_num}/{total_batches}")
        time.sleep(sleep_s)
 
    return pd.DataFrame(rows)
 
BLS_DF = get_bls_unemployment(fips_list, YEARS)
save_point(BLS_DF, 'BLS_import.csv',
            f"{len(BLS_DF):,} annual umemployment rate obs")
bls_data = BLS_DF.copy()
bls_data = standardize_fips(bls_data)
bls_data = define_cols(bls_data)
bls_data = remap_fips_changes(bls_data, fips_cols=['FIPS'])
if not bls_data.empty:
    print(f"Unique FIPS: {bls_data['FIPS'].nunique():,}")
print(" BLS data acquisition complete\n")
 
# -----------------------------------------------------------
# 4.4: USDA Data Files
# -----------------------------------------------------------
print("4.4: Loading USDA data files...")
 
# Rural-Urban Continuum Codes
 
rucc_2013 = pd.read_excel(CWD / 'ruralurbancodes2013.xls',
                          dtype={'FIPS': str})
rucc_2013 = standardize_fips(rucc_2013)
rucc_2013 = rucc_2013[['FIPS', 'RUCC_2013']]
rucc_2013 = remap_fips_changes(rucc_2013, fips_cols=['FIPS'])
rucc_2013 = filter_dataframe(rucc_2013, 'RUCC 2013')
 
rucc_2023_long = pd.read_csv(
    CWD / 'Ruralurbancontinuumcodes2023.csv',
    dtype={'FIPS': str},
    encoding='latin-1')
# Extract only rucc_2023 values
rucc_2023 = rucc_2023_long[
    rucc_2023_long['Attribute'] == 'RUCC_2023'].copy()
rucc_2023 = rucc_2023[[
    'FIPS', 'Value']].rename(columns={'Value': 'RUCC_2023'})
rucc_2023 = standardize_fips(rucc_2023)
rucc_2023['RUCC_2023'] = pd.to_numeric(
    rucc_2023['RUCC_2023'], errors='coerce')
rucc_2023 = remap_fips_changes(rucc_2023, fips_cols=['FIPS'])
rucc_2023 = filter_dataframe(rucc_2023, 'RUCC 2023')
 
# County Typology
typology = pd.read_csv(CWD / 'erscountytypology2015edition.csv',
                       dtype={'FIPStxt': str},
                       encoding='latin-1')
# Drop duplicate identification columns
typology.drop(columns=['State', 'County_name', 'Economic_Type_Label',
                       'Metro-nonmetro status, 2013 0=Nonmetro 1=Metro'],
              inplace=True, errors='ignore')
 
typology.rename(columns={
    'FIPStxt': 'FIPS',
    'Economic Types Type_2015_Update non-overlapping': 'Industry_type',
    'Farming_2015_Update': 'Farming',
    'Mining_2015-Update': 'Mining',
    'Manufacturing_2015_Update': 'Mfging',
    'Government_2015_Update': 'Govt',
    'Recreation_2015_Update': 'Rec',
    'Nonspecialized_2015_Update': 'Nonspec',
    'Low_Education_2015_Update': 'Low_Ed_cnty',
    'Low_Employment_Cnty_2008_2012_25_64': 'Low_employ_cnty',
    'Retirement_Dest_2015_Update': 'Retire_dest_cnty',
    'Persistent_Poverty_2013': 'Persistent_Pov_cnty',
    'Persistent_Related_Child_Poverty_2013': 'Pers_chld_pov_cnty'},
    inplace=True)
typology = standardize_fips(typology)
typology = define_cols(typology)
typology = remap_fips_changes(typology)
typology = filter_dataframe(typology, 'Typology 2015')
print(f"Typology 2015: {len(typology):,} counties, {len(typology.columns)} variables")
 
# Natural Amenities Scale - Data starts at row 106
amenities = pd.read_excel(CWD / 'natamenf_1_.xls',
                          dtype={'for measures': str},
                          header=104)
amenities.rename(columns={
    'for measures': 'FIPS',
    'Scale': 'Amenity_scale'}, inplace=True)
amenities = amenities.drop_duplicates(
    subset='FIPS', keep='first')
# Keep only amenity variables, drop components
amenities = amenities[['FIPS', 'Amenity_scale']]
amenities = standardize_fips(amenities)
amenities = define_cols(amenities)
amenities = remap_fips_changes(amenities)
amenities = filter_dataframe(amenities, 'Natural Amenities')
print(f"Natural Amenities: {len(amenities):,} counties")
 
print("\n **** USDA data loading complete\n")
 
# ----------------------------------------------------------
# 4.5: IRS County-to-County Migration Data
# ----------------------------------------------------------
print("4.5: Processing IRS migration data...")
 
def process_irs_migration(filename):
    df = pd.read_csv(CWD / filename, dtype=str, encoding='latin-1')
    # Extract year from filename: countyinflow1112.csv -> 2011
    year_suffix = filename[12:14]
    year = 2000 + int(year_suffix)
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    rename_map = {
        'n2': 'movers',
        'agi': 'movers_agi'}
    df.rename(columns=rename_map, inplace=True)
 
    # Create FIPS codes
    df = standardize_fips(df, state_col='y1_statefips',
                            county_col='y1_countyfips',
                            fips_col='origin_FIPS')
    df = standardize_fips(df, state_col='y2_statefips',
                            county_col='y2_countyfips',
                            fips_col='dest_FIPS')
 
    # Drop state totals (XX000) and non-movers (origin=dest):
    ST_TOT_dest = (
        df['dest_FIPS'].astype('Int64') % 1000 == 0)
    ST_TOT_origin = (
        df['origin_FIPS'].astype('Int64') % 1000 == 0)
    non_movers = (df['origin_FIPS'] == df['dest_FIPS'])
    drop_mask = (ST_TOT_dest | ST_TOT_origin | non_movers)
    df = df[~drop_mask].copy()
 
    # Exclude FIPS > 56999 (Territorie, Puerto Rico, Foreign)
    df = df[df['origin_FIPS'].astype('Int64') < 57000]
    df = df[df['dest_FIPS'].astype('Int64') < 57000]
 
    # Ensure FIPS are objects
    df['dest_FIPS'] = df['dest_FIPS'].astype(str)
    df['origin_FIPS'] = df['origin_FIPS'].astype(str)
    df['Year'] = year
    return df
 
# Process all years
print("  - Processing 11 years of bilateral flows...")
bilateral_flows = []
 
for filename in IRS_FILES:
    df = process_irs_migration(filename)
    bilateral_flows.append(df)
    print(f"    {filename}: {len(df):,} flows (Year {df['Year'].iloc[0]})")
 
bilateral = pd.concat(bilateral_flows, ignore_index=True)
bilateral = define_cols(bilateral)
bilateral = remap_fips_changes(bilateral, fips_cols=['dest_FIPS'])
bilateral = remap_fips_changes(bilateral, fips_cols=['origin_FIPS'])
bilateral = filter_dataframe(bilateral, 'Bilateral Flows')
IRS_bilateral = bilateral[['origin_FIPS', 'Year', 'dest_FIPS', 'movers', 'movers_agi']]
 
# Creates two datasets:
#    1. Bilateral flows: County-to-county migration for gravity model
#    2. Aggregated net migration: County-year totals for panel models
 
# Save Full Migration with flows
save_point(IRS_bilateral, 'IRS_gravity.csv',
               f"{len(IRS_bilateral):,}")
 
inflow = IRS_bilateral.groupby(['dest_FIPS', 'Year'], as_index=False).agg({
    'movers': 'sum',
    'movers_agi': 'sum'}).rename(columns={
        'dest_FIPS': 'FIPS',
        'movers': 'move_in',
        'movers_agi': 'agi_in'})
outflow = IRS_bilateral.groupby(['origin_FIPS', 'Year'], as_index=False).agg({
    'movers': 'sum',
    'movers_agi': 'sum'}).rename(columns={
        'origin_FIPS': 'FIPS',
        'movers': 'move_out',
        'movers_agi': 'agi_out'})
inflow['FIPS'] = inflow['FIPS'].astype(str).str.zfill(5)
outflow['FIPS'] = outflow['FIPS'].astype(str).str.zfill(5)
 
# Merge and calculate net
IRS_migration = pd.merge(inflow, outflow, on=['FIPS', 'Year'], how='outer').fillna(0)
IRS_migration['move_net'] = IRS_migration['move_in'] - IRS_migration['move_out']
IRS_migration['agi_net'] = IRS_migration['agi_in'] - IRS_migration['agi_out']
IRS_migration = remap_fips_changes(IRS_migration, fips_cols=['FIPS'])
 
# Save Full Migration aggregated for panel
save_point(IRS_migration, 'IRS_panel.csv',
               f"{len(IRS_migration):,} panel-ready obs")
 
# Display
print(f"  Panel rows: {len(IRS_migration):,}")
print(f"  Unique counties: {IRS_migration['FIPS'].nunique():,}")
 
# -----------------------------------------------------------
# 4.6: Housing Incentive Programs Database
# -----------------------------------------------------------
print("4.6: Loading housing incentive programs data...")
 
incentives = pd.read_csv(CWD / 'Incentives.csv', dtype={'FIPS': str})
incentives = incentives[[
    'FIPS', 'Year', 'has_incentive', 'Incentive_CAT', 'COVID_program']]
incentives = standardize_fips(incentives)
incentives = filter_dataframe(incentives, 'Housing Incentive Programs')
 
print(f"  - Incentive programs: {incentives['FIPS'].nunique():,} counties")
print("\n Incentive data loading complete\n")
print("="*49)
print("SECTION 4: DATA ACQUISITION COMPLETE")
print("="*49 + "\n")
 
# ===========================================================
# SECTION 5: DATA CLEANING (Non-Census)
# ===========================================================
print("="*49)
print("SECTION 5: DATA CLEANING & HARMONIZING")
print("="*49 + "\n")
 
#1 IRS data
IRS_migration.loc[
    IRS_migration['FIPS'] == '20157', [
        'move_out', 'agi_out',
        'move_net', 'agi_net']] = IRS_migration.loc[
    IRS_migration['FIPS'] == '20157', [
        'move_out', 'agi_out',
        'move_net', 'agi_net']].fillna(0)
 
#2 USDA data
amenities['Amenity_scale'] = amenities['Amenity_scale'].fillna(0)
# Align FIPS: Dade, FL → Miami-Dade, FL
amenities.loc[amenities['FIPS'] == '12025', 'FIPS'] = '12086'
rucc_2023 = rucc_2023[~rucc_2023['FIPS'].str.startswith('091')].copy()
rucc_2023 = rucc_2023[~rucc_2023['FIPS'].str.startswith('091')].copy()
 
#3 BEA data
# Change '15901' to '15009' in pci and gdp
pci['FIPS'] = pci['FIPS'].replace({'15901': '15009'})
gdp['FIPS'] = gdp['FIPS'].replace({'15901': '15009'})
# Remove pci FIPS=55901, unique and empty
pci = pci[pci['FIPS'] != '55901']
# Drop FIPS '519XX' BEA lists some VA cities and counties
pci = pci[~pci['FIPS'].str.startswith('519')].copy()
gdp = gdp[~gdp['FIPS'].str.startswith('519')].copy()
 
print("Clean Census data next\n")
 
CENSUS_DF.info()
 
# Clean CENSUS
CENSUS_clean = CENSUS_DF.copy()
CENSUS_clean = standardize_fips(CENSUS_clean)
CENSUS_clean = remap_fips_changes(CENSUS_clean, fips_cols=['FIPS'])
CENSUS_clean = filter_dataframe(CENSUS_clean)
CENSUS_clean = CENSUS_clean.replace(-666666666, np.nan)
 
# Ensure median columns are Int64
median_vars = ['median_property_taxes', 'median_hh_income', 'median_home_value']
CENSUS_clean[median_vars] = CENSUS_clean[median_vars].apply(
    lambda col: pd.to_numeric(col, errors='coerce').astype('Int64'))
 
# Helper functions
def avg_years(df, fips, col, y1, y2):
    vals = df.loc[(df['FIPS'] == fips) & df['Year'].isin([y1, y2]), col]
    return int(round(vals.mean())) if vals.notna().all() else np.nan
 
def fill(df, fips, col, year, value):
    df.loc[(df['FIPS'] == fips) & (df['Year'] == year), col] = value
 
# Impute for specific FIPS/year (all columns)
for col in CENSUS_clean.columns:
    if CENSUS_clean.loc[(CENSUS_clean['FIPS'] == '35039') & (
        CENSUS_clean['Year'] == 2018), col].isna().any():
        val = avg_years(CENSUS_clean, '35039', col, 2017, 2019)
        fill(CENSUS_clean, '35039', col, 2018, val)

# Impute 2015 gaps in median variables
for col in median_vars:
    missing_fips = CENSUS_clean.loc[
        (CENSUS_clean['Year'] == 2015) & (
            CENSUS_clean[col].isna()), 'FIPS'].unique()
    for fips in missing_fips:
        val = avg_years(CENSUS_clean, fips, col, 2014, 2016)
        if pd.notna(val):
            fill(CENSUS_clean, fips, col, 2015, val)
 
# Recompute final features
CENSUS_clean = define_cols(CENSUS_clean)
 
# ===========================================================
# SECTION 6: EDA
# ===========================================================
print("-"*49)
print("SECTION 6: EDA")
print("-"*49 + "\n")
 
# Pearson, Spearman, and XICOR correlations computed to account for linear, monotonic, and nonlinear associations respectively.
# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
 
print("DESCRIPTIVE STATISTICS")
print("-"*49)
print(CENSUS_clean.describe())
 
print("-"*49)
print("SECTION 6: Varience and VIF")
print("-"*49 + "\n")
check_num = CENSUS_clean.select_dtypes(include=np.number)
 
# Calculate variances
variances = check_num.var()
 
# Sort variances in ascending order
var_sorted = variances.sort_values(ascending=True)
 
# Set display format
pd.set_option('display.float_format', '{:.2f}'.format)
 
# Show low variance features (consider dropping if < 0.05)
print('FEATURE VARIANCES (Lowest 20)')
print('= '*25)
print(var_sorted.head(20))
print('\n')
print('FEATURE VARIANCES (Highest 20)')
print('= '*25)
print(var_sorted.tail(20))
 
# Identify very low variance features
low_var = var_sorted[var_sorted < 0.05]
print(f"\n\nFeatures with variance < 0.05: {len(low_var)}")
if len(low_var) > 0:
    print(low_var)
 
# Prepare features (drop dependent variable and identifiers)
exclude_for_vif = ['FIPS', 'Year', 'move_net', 'net_agi'] + [
    col for col in CENSUS_clean.columns if col not in check_num.columns]
 
VIF_feat = check_num.drop(
    columns=[col for col in exclude_for_vif if col in check_num.columns])
 
# Add constant for VIF calculation
VIF_const = sm.add_constant(VIF_feat)
VIF_const = VIF_const.astype(float).dropna()
 
print('COMPUTING VIF (This may take a few minutes...)')
print('='*30)
 
# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = VIF_const.columns
vif_data['VIF'] = [
    variance_inflation_factor(VIF_const.values, i)
    for i in range(VIF_const.shape[1])]
 
# Sort by VIF descending
vif_data = vif_data.sort_values(by='VIF', ascending=False)
 
print('\nVIF Results (VIF > 10 indicates multicollinearity):')
print(vif_data.head(30))
 
# ===========================================================
# SECTION 7: FEATURE ENGINEERING
# ===========================================================
print("="*49)
print("SECTION 7: FEATURE ENGINEERING")
print("="*49 + "\n")
 
# 7.1: Census aggregates
print("Creating census aggregates...")
POP_TOT = CENSUS_clean['total_population']
MAR_TOT = CENSUS_clean['marital_total']
 
# Home Tenure: Rent vs Own
CENSUS_clean['%owner_occupied'] = (
    CENSUS_clean['owner_occupied'] / CENSUS_clean['housing_total'] * 100)
# Marital status
CENSUS_clean['%never_married_male'] = (
    CENSUS_clean['never_married_male'] / MAR_TOT) * 100
CENSUS_clean['%now_married_male'] = (
    CENSUS_clean['now_married_male'] / MAR_TOT) * 100
CENSUS_clean['%divorced_male'] = (
    CENSUS_clean['divorced_male'] / MAR_TOT) * 100
CENSUS_clean['%never_married_female'] = (
    CENSUS_clean['never_married_female'] / MAR_TOT) * 100
CENSUS_clean['%now_married_female'] = (
    CENSUS_clean['now_married_female'] / MAR_TOT) * 100
CENSUS_clean['%divorced_female'] = (
    CENSUS_clean['divorced_female'] / MAR_TOT) * 100
CENSUS_clean['%widowed_female'] = (
    CENSUS_clean['widowed_female'] / MAR_TOT) * 100
# Race and Ethnicity
CENSUS_clean['%white'] = (
    CENSUS_clean['white'] / POP_TOT) * 100
CENSUS_clean['%black'] = (
    CENSUS_clean['black'] / POP_TOT) * 100
CENSUS_clean['%native'] = (
    CENSUS_clean['native'] / POP_TOT) * 100
CENSUS_clean['%asian'] = (
    CENSUS_clean['asian'] / POP_TOT) * 100
CENSUS_clean['%pacific_islander'] = (
    CENSUS_clean['pacific_islander'] / POP_TOT) * 100
CENSUS_clean['%other_race'] = (
    CENSUS_clean['other_race'] / POP_TOT) * 100
CENSUS_clean['%hispanic'] = (
    CENSUS_clean['hispanic'] / POP_TOT) * 100
# College degree holder: At least 25 years of age 
CENSUS_clean['%college_degree'] = (
    CENSUS_clean['male_associates'] + CENSUS_clean['female_associates'] +
    CENSUS_clean['male_bachelors'] + CENSUS_clean['female_bachelors'] +
    CENSUS_clean['male_masters'] + CENSUS_clean['female_masters'] +
    CENSUS_clean['male_professional'] + CENSUS_clean['female_professional'] +
    CENSUS_clean['male_doctorate'] + CENSUS_clean['female_doctorate'] / 
    CENSUS_clean['education_total_sex']) * 100
# Graduated High School or has some college, but no degree: Aged 25 or older
CENSUS_clean['%HSorCollege_NOdegree'] = (
    CENSUS_clean['male_complete_hs'] + CENSUS_clean['female_complete_hs'] +
    CENSUS_clean['male_less1yr_college'] + CENSUS_clean['female_less1yr_college'] +
    CENSUS_clean['male_more1yr_college'] + CENSUS_clean['female_more1yr_college'] /
    CENSUS_clean['education_total_sex']) * 100
# Industry: White collar
CENSUS_clean['%white_collar'] = (
    CENSUS_clean['Mgmt_Biz_Sci_Arts'] + CENSUS_clean['Services'] +
    CENSUS_clean['Sales_Admin']) / CENSUS_clean['occupation_total'] * 100

print("Created 18 CENSUS features")
 
# Optimize percentage features
for col in CENSUS_clean.filter(regex=r'^%').columns:
    CENSUS_clean[col] = CENSUS_clean[col].round(2)
    CENSUS_clean[col] = pd.to_numeric(CENSUS_clean[col], errors='coerce')
 
# 7.2 Cleanup: Drop raw variables
drop_vars = [
    'family_households', 'education_total_sex',
    'marital_total',
    'male_complete_hs', 'female_complete_hs',
    'male_less1yr_college', 'female_less1yr_college',
    'male_more1yr_college', 'female_more1yr_college',
    'male_associates', 'female_associates',
    'male_bachelors', 'female_bachelors',
    'male_masters', 'female_masters',
    'male_professional', 'female_professional',
    'male_doctorate', 'female_doctorate',
    'Mgmt_Biz_Sci_Arts', 'Sales_Admin', 'occupation_total',
    'Nat-rsrc_Constr_Maint', 'Services', 'Prod_Transp_Mvng',
    'owner_occupied', 'housing_total', 'renter_occupied',
    'white', 'black', 'native', 'asian', 'pacific_islander',
    'other_race', 'mixed_non_h', 'hispanic',
    'never_married_male', 'never_married_female',
    'now_married_male', 'now_married_female',
    'divorced_male', 'divorced_female',
    'widowed_male', 'widowed_female']
CENSUS_reduced = CENSUS_clean.drop(columns=drop_vars)
print(f"Dropped {len(drop_vars)} raw variables")
print("\n Feature engineering complete")
 
# Second VIF with reduced features
exclude_for_vif = ['FIPS', 'Year', 'move_net', 'net_agi'] + [
    col for col in CENSUS_reduced.columns if col not in check_num.columns]
VIF_feat = check_num.drop(
    columns=[col for col in exclude_for_vif if col in check_num.columns])
VIF_const = sm.add_constant(VIF_feat)
VIF_const = VIF_const.astype(float).dropna()
vif_data = pd.DataFrame()
vif_data['Feature'] = VIF_const.columns
vif_data['VIF'] = [
    variance_inflation_factor(VIF_const.values, i)
    for i in range(VIF_const.shape[1])]
vif_data = vif_data.sort_values(by='VIF', ascending=False)
print('\nVIF Results (VIF > 10 indicates multicollinearity):')
print(vif_data.head(30))
 
# ============================================================
# SECTION 8: MERGE & PANEL CONSTRUCTION
# ============================================================
print("="*49)
print("SECTION 7: MERGE DATASETS & PANEL CONSTRUCTION")
print("="*49 + "\n")
 
print("Constructing county-year analytical panel...")
 
# 8.1 Merge into Panel - Start with Census (panel baseline)
panel = CENSUS_clean.copy()
print(f"  Starting panel: {len(panel):,} observations ({panel['FIPS'].nunique()} counties)")
 
# USDA: RUCC (temporal assignment: 2013→2011-2019, 2023→2020-2021)
rucc_2013_panel = pd.concat([rucc_2013.assign(
    Year=y) for y in range(2011, 2020)], ignore_index=True)
rucc_2013_panel = rucc_2013_panel.drop_duplicates(['FIPS', 'Year'])
rucc_2023_panel = pd.concat([rucc_2023.assign(
    Year=y) for y in range(2020, 2022)], ignore_index=True)
rucc_2023_panel = rucc_2023_panel.drop_duplicates(['FIPS', 'Year'])
panel = panel.merge(rucc_2013_panel[[
    'FIPS', 'Year', 'RUCC_2013']], on=[
        'FIPS', 'Year'], how='left')
panel = panel.merge(rucc_2023_panel[[
    'FIPS', 'Year', 'RUCC_2023']], on=[
        'FIPS', 'Year'], how='left')
# Combine
panel['RUC_code'] = panel['RUCC_2013'].combine_first(panel['RUCC_2023'])
if panel['RUC_code'].notna().all():
    panel['RUC_code'] = panel['RUC_code'].astype('Int64')
else:
    missing_rucc = panel[panel['RUC_code'].isna()]
    print(" Missing RUCC values found:")
    print(missing_rucc[['FIPS', 'Year']].drop_duplicates())
print(f"  + RUCC: {len(panel):,} obs")
 
# USDA: typology, amenities
typology_panel = pd.concat([typology.assign(
    Year=y) for y in range(2011, 2022)], ignore_index=True)
typology_panel = typology_panel.drop_duplicates(['FIPS', 'Year'])
amenities_panel = pd.concat([amenities.assign(
    Year=y) for y in range(2011, 2022)], ignore_index=True)
amenities_panel = amenities_panel.drop_duplicates(['FIPS', 'Year'])
panel = panel.merge(typology_panel, on=['FIPS', 'Year'], how='left')
panel = panel.merge(amenities_panel, on=['FIPS', 'Year'], how='left')
typology_cols = [
    c for c in typology_panel.columns if c not in ['FIPS', 'Year']]
amenities_cols = [
    c for c in amenities_panel.columns if c not in ['FIPS', 'Year']]
panel[typology_cols] = panel[typology_cols].fillna(0)
panel[amenities_cols] = panel[amenities_cols].fillna(0)
print(f"  + USDA (typology, amenities): {len(panel):,} obs")
 
# BEA
pci = pci.drop_duplicates(['FIPS', 'Year'])
gdp = gdp.drop_duplicates(['FIPS', 'Year'])
panel = panel.merge(pci[[
    'FIPS', 'Year', 'BEA_PCI']], on=[
        'FIPS', 'Year'], how='left')
panel = panel.merge(gdp[[
    'FIPS', 'Year', 'BEA_GDP']], on=[
        'FIPS', 'Year'], how='left')
panel['BEA_PCI'] = panel['BEA_PCI'].fillna(0)
panel['BEA_GDP'] = panel['BEA_GDP'].fillna(0)
 
# Extract STATE from panel FIPS
panel['STATE'] = panel['FIPS'].str[:2]
# Merge metro RPP
rpp_metro = rpp_metro.drop_duplicates(['FIPS', 'Year'])
panel = panel.merge(rpp_metro[['FIPS', 'Year', 'RPP_Metro']], 
                    on=['FIPS', 'Year'], how='left')
# Prepare non-metro state RPP
rpp_nonmetro['STATE'] = rpp_nonmetro[
    'State_FIPS'].astype(str).str[:2]
rpp_nonmetro = rpp_nonmetro[rpp_nonmetro[
    'State_FIPS'].astype(str).str.endswith('999')].copy()
rpp_nonmetro = rpp_nonmetro.drop_duplicates([
    'STATE', 'Year'])  # ← Add
panel = panel.merge(rpp_nonmetro[[
    'STATE', 'Year', 'RPP_NonMetro']], on=['STATE', 'Year'], how='left')
# Fill metro with non-metro
panel['RPP'] = panel['RPP_Metro'].fillna(panel['RPP_NonMetro'])
panel = panel.drop(columns=['RPP_Metro', 'RPP_NonMetro', 'STATE'])
print(f"  + BEA (PCI, GDP, RPP): {len(panel):,} obs")
 
# BLS
bls_data = bls_data.drop_duplicates([
    'FIPS', 'Year'])
panel = panel.merge(bls_data[[
    'FIPS', 'Year', 'unemploy_rate']], on=[
        'FIPS', 'Year'], how='left')
panel['unemploy_rate'] = panel['unemploy_rate'].fillna(0)
print(f"  + BLS: {len(panel):,} obs")
 
# IRS Migration
IRS_migration = IRS_migration.drop_duplicates(['FIPS', 'Year'])  # ← Add
panel = panel.merge(IRS_migration, on=['FIPS', 'Year'], how='left')
migration_cols = [c for c in IRS_migration.columns if c not in ['FIPS', 'Year']]
panel[migration_cols] = panel[migration_cols].fillna(0)
print(f"  + IRS Migration: {len(panel):,} obs")
 
# Incentives
incentives = incentives.drop_duplicates(['FIPS', 'Year'])  # ← Add
panel = panel.merge(incentives, on=['FIPS', 'Year'], how='left')
incentive_cols = [c for c in incentives.columns if c not in ['FIPS', 'Year']]
panel[incentive_cols] = panel[incentive_cols].fillna(0)
print(f"  + Incentives: {len(panel):,} obs")
 
print(f"\nFinal panel: {len(panel):,} obs, {panel['FIPS'].nunique()} counties, {panel['Year'].nunique()} years")
print(f"Remaining NaNs: {panel.isna().sum().sum()}")
 
# 8.2 Clean panel for modeling
less_vars = [
    'RUCC_2013', 'RUCC_2023', 'Industry_type',
    'move_in', 'move_out', 'agi_in', 'agi_out']
panel = panel.drop(columns=less_vars)
 
save_point(panel, 
           'full_panel.csv', f"{len(panel):,} county-year observations, {panel['FIPS'].nunique():,} counties")
print("Merge Complete, panel dataframe ready for model")
 
# ===========================================================
# SECTION 9: STATISTICAL MODELS
# ===========================================================
print("="*49)
print("SECTION 9: STATISTICAL MODELS")
print("="*49 + "\n")
 
# ===========================================================
# MODEL 1 - GRAVITY MODEL
# ===========================================================
print("MODEL 1: GRAVITY MODEL\n")
 
# Assign RUCC codes to years
rucc_2013_panel = pd.concat([
    rucc_2013.assign(Year=year) for year in range(2011, 2020)],
        ignore_index=True).rename(columns={'RUCC_2013': 'RUC_code'})
 
rucc_2023_panel = pd.concat([
    rucc_2023.assign(Year=year) for year in range(2020, 2022)],
        ignore_index=True).rename(columns={'RUCC_2023': 'RUC_code'})
 
rucc_all = pd.concat([rucc_2013_panel, rucc_2023_panel], ignore_index=True)
rucc_all['RUC_code'] = pd.to_numeric(rucc_all['RUC_code'], 
                                     errors='coerce').astype('Int64')
rucc_all['FIPS'] = rucc_all['FIPS'].astype(str)
 
# Prepare gravity data
gravity_df = IRS_bilateral.copy()
gravity_df['dest_FIPS'] = gravity_df['dest_FIPS'].astype(str).str.zfill(5)
 
gravity_df = gravity_df.merge(
    rucc_all.rename(columns={'FIPS': 'dest_FIPS', 'RUC_code': 'RUC_code_dest'}),
    on=['dest_FIPS', 'Year'], how='left')
 
gravity_df = gravity_df.merge(
    rucc_all.rename(columns={'FIPS': 'origin_FIPS', 'RUC_code': 'RUC_code_origin'}),
    on=['origin_FIPS', 'Year'], how='left')
 
gravity_df = gravity_df.drop(columns=['movers_agi']).dropna(subset=['RUC_code_origin', 'RUC_code_dest'])
 
# Create variables
gravity_df['log_movers'] = np.log(gravity_df['movers'].astype(float) + 1)
gravity_df['rucc_pair'] = (gravity_df['RUC_code_origin'].astype(str) + '→' +
                            gravity_df['RUC_code_dest'].astype(str))
 
# Estimate
X_grav = pd.get_dummies(gravity_df[['RUC_code_origin', 'RUC_code_dest']], drop_first=True)
X_grav = sm.add_constant(X_grav).astype(float)
y_grav = gravity_df['log_movers'].astype(float)
model1 = sm.OLS(y_grav, X_grav).fit()
 
# Save
save_point(pd.DataFrame({
    'Variable': model1.params.index,
    'Coefficient': model1.params.values,
    'Std_Error': model1.bse,
    'P_value': model1.pvalues,
    'R2': model1.rsquared}),
    'MODEL-1_gravity_results.csv', 'Gravity Model')
 
print(" Model 1 complete\n")

# ===========================================================
# MODEL 2 - PANEL FIXED EFFECTS
# ===========================================================
print("MODEL 2: PANEL FIXED EFFECTS\n")
 
model_2 = panel.copy().set_index(['FIPS', 'Year'])
 
# Define variables (removed collinear percentage pairs)
panel_X_vars = [
    'total_population', 'median_age', 'under_18_in_hh',
    'median_hh_income', 'median_home_value', 'median_property_taxes',
    'commute_less_5min', 'commute_5_9min', 'commute_10_14min',
    'commute_15_19min', 'commute_20_24min', 'commute_25_29min',
    'commute_30_34min', 'commute_35_39min', 'commute_40_44min',
    'commute_45_59min', 'commute_60_89min', 'commute_90_plus_min',
    'college_degree', '%white_collar', '%owner_occupied',
    '%white', '%black', '%native', '%asian', '%pacific_islander',
    '%other_race', '%hispanic',
    '%never_married_male', '%never_married_female',
    '%now_married_male', '%now_married_female',
    '%divorced_male', '%divorced_female', # Removed 'Nonspec'
    'RUC_code', 'Farming', 'Mining', 'Mfging', 'Govt', 'Rec',
    'Low_Ed_cnty', 'Low_employ_cnty', 'Pop_Loss_2010',
    'Retire_dest_cnty', 'Persistent_Pov_cnty', 'Pers_chld_pov_cnty',
    'Amenity_scale', 'BEA_PCI', 'BEA_GDP', 'RPP', 'unemploy_rate']
 
# Time-varying only (for FE)
X_time_varying = model_2[[
    'total_population', 'median_hh_income', 'median_home_value',
    'median_property_taxes', 'college_degree',
    'BEA_PCI', 'BEA_GDP', 'RPP', 'unemploy_rate']]
 
y = model_2['move_net']
X = model_2[panel_X_vars]
# Specification A: Pooled OLS
pooled_model = PooledOLS(y, X).fit(cov_type='clustered', cluster_entity=True)
 
# Specification B: County FE
fe_model = PanelOLS(y, X_time_varying, entity_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
 
# Specification C: Two-Way FE
twoway_model = PanelOLS(y, X_time_varying, entity_effects=True, time_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
 
# Save comparison
save_point(pd.DataFrame({
    'Model': ['Pooled_OLS', 'County_FE', 'TwoWay_FE'],
    'R2_Overall': [pooled_model.rsquared, fe_model.rsquared, twoway_model.rsquared],
    'R2_Within': [pooled_model.rsquared, fe_model.rsquared_within, twoway_model.rsquared_within],
    'N_Obs': [pooled_model.nobs, fe_model.nobs, twoway_model.nobs]}),
    'MODEL-2_panel_fe_comparison.csv', 'Panel FE specifications')
 
print(" Model 2 complete\n")
 
# ===========================================================
# MODEL 3 - DIFFERENCE-IN-DIFFERENCES
# ===========================================================
print("MODEL 3: DIFFERENCE-IN-DIFFERENCES\n")
 
model_3 = panel.copy()
model_3['PULL'] = model_3['has_incentive']
model_3['POST'] = (model_3['Year'] >= model_3['Incentive_CAT']).fillna(0).astype('Int64')
model_3['pull_x_post'] = model_3['PULL'] * model_3['POST']
 
did_data = model_3.dropna(subset=['move_net'] + panel_X_vars).set_index(['FIPS', 'Year'])
y_did = did_data['move_net']
 
# Specification A: Simple
X_did_simple = did_data[['pull_x_post']]
model3a = PanelOLS(y_did, X_did_simple, entity_effects=True, time_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
 
# Specification B: With controls
X_did_controls = did_data[['pull_x_post'] + panel_X_vars]
model3b = PanelOLS(y_did, X_did_controls, drop_absorbed=True,
                   entity_effects=True, time_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
 
# Specification C: Heterogeneous by category
if 'Incentive_CAT' in model_3.columns:
    model_3['pull_CAT1'] = ((model_3['Incentive_CAT'] == 1) * model_3['POST']).astype('Int64')
    model_3['pull_CAT2'] = ((model_3['Incentive_CAT'] == 2) * model_3['POST']).astype('Int64')
    model_3['pull_CAT3'] = ((model_3['Incentive_CAT'] == 3) * model_3['POST']).astype('Int64')
 
    did_data_het = model_3.dropna(subset=['move_net'] + panel_X_vars).set_index(['FIPS', 'Year'])
    X_did_het = did_data_het[['pull_CAT1', 'pull_CAT2', 'pull_CAT3'] + panel_X_vars]
 
    model3c = PanelOLS(y_did, X_did_het, entity_effects=True,
                       time_effects=True, drop_absorbed=True).fit(
        cov_type='clustered', cluster_entity=True)
 
    cat1 = model3c.params.get('pull_CAT1', np.nan)
    cat1_se = model3c.std_errors.get('pull_CAT1', np.nan)
    cat1_p = model3c.pvalues.get('pull_CAT1', np.nan)
    cat2 = model3c.params.get('pull_CAT2', np.nan)
    cat2_se = model3c.std_errors.get('pull_CAT2', np.nan)
    cat2_p = model3c.pvalues.get('pull_CAT2', np.nan)
    cat3 = model3c.params.get('pull_CAT3', np.nan)
    cat3_se = model3c.std_errors.get('pull_CAT3', np.nan)
    cat3_p = model3c.pvalues.get('pull_CAT3', np.nan)
else:
    cat1 = cat1_se = cat1_p = np.nan
    cat2 = cat2_se = cat2_p = np.nan
    cat3 = cat3_se = cat3_p = np.nan
 
save_point(pd.DataFrame({
    'Specification': ['Simple', 'Controls', 'CAT1', 'CAT2', 'CAT3'],
    'Treatment_Effect': [
        model3a.params.get('pull_x_post', np.nan),
        model3b.params.get('pull_x_post', np.nan),
        cat1, cat2, cat3],
    'Std_Error': [
        model3a.std_errors.get('pull_x_post', np.nan),
        model3b.std_errors.get('pull_x_post', np.nan),
        cat1_se, cat2_se, cat3_se],
    'P_value': [
        model3a.pvalues.get('pull_x_post', np.nan),
        model3b.pvalues.get('pull_x_post', np.nan),
        cat1_p, cat2_p, cat3_p],
    'R2_Within': [
        model3a.rsquared, model3b.rsquared,
        model3c.rsquared if 'Incentive_CAT' in model_3.columns else np.nan,
        model3c.rsquared if 'Incentive_CAT' in model_3.columns else np.nan,
        model3c.rsquared if 'Incentive_CAT' in model_3.columns else np.nan]}),
    'MODEL-3_did_comparison.csv', 'DiD specifications')
 
print(" Model 3 complete\n")
 
# ===========================================================
# MODEL 4 - DYNAMIC PANEL
# ===========================================================
print("MODEL 4: DYNAMIC PANEL\n")
 
dynamic_panel = panel.copy().sort_values(['FIPS', 'Year'])
dynamic_panel['move_net_lag1'] = dynamic_panel.groupby('FIPS')['move_net'].shift(1)
 
dynamic_data = dynamic_panel.dropna(
    subset=['move_net', 'move_net_lag1'] + panel_X_vars).set_index(['FIPS', 'Year'])
 
y_dynamic = dynamic_data['move_net']
X_dynamic = dynamic_data[['move_net_lag1'] + panel_X_vars]
 
model4 = PanelOLS(y_dynamic, X_dynamic, entity_effects=True,
                  drop_absorbed=True, time_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
 
persistence = model4.params['move_net_lag1']
lr_multiplier = 1 / (1 - persistence) if abs(persistence) < 1 else np.inf
 
save_point(pd.DataFrame({
    'Variable': model4.params.index,
    'Coefficient': model4.params.values,
    'Std_Error': model4.std_errors.values,
    'P_value': model4.pvalues.values,
    'R2_Within': model4.rsquared,
    'Persistence': persistence,
    'LR_Multiplier': lr_multiplier}),
    'MODEL-4_dynamic_results.csv', 'Dynamic panel')
 
print(" Model 4 complete\n")
print("="*49)
print("SECTION 9 COMPLETE")
print("="*49 + "\n")
 
# ===========================================================
# SECTION 10: CONSOLIDATED RESULTS & OUTPUT
# ===========================================================
print("="*49)
print("SECTION 10: CONSOLIDATED RESULTS & OUTPUT")
print("="*49 + "\n")
'''
Consolidate all model results and generate publication-ready tables:
1. Master Model Comparison Table
2. Hypothesis 1: Program timing effects (Pre-COVID vs COVID-era)
3. Hypothesis 2: Incentive vs non-incentive comparison
4. Hypothesis 3: Rural-urban migration flows
5. Descriptive statistics by incentive status
6. Key findings summary
'''
# ===========================================================
# 10.1: MASTER MODEL COMPARISON TABLE
# ===========================================================
print("10.1: Generating master model comparison table...\n")
 
master_comparison = pd.DataFrame({
    'Model': [
        'Gravity_Flow',
        'Panel_Pooled_OLS',
        'Panel_County_FE',
        'Panel_TwoWay_FE',
        'DiD_Simple',
        'DiD_Controls',
        'DiD_CAT1',
        'DiD_CAT2',
        'DiD_CAT3',
        'Dynamic_Panel'],
    'N_Obs': [
        int(model1.nobs),
        int(pooled_model.nobs),
        int(fe_model.nobs),
        int(twoway_model.nobs),
        int(model3a.nobs),
        int(model3b.nobs),
        int(model3c.nobs) if 'Incentive_CAT' in model_3.columns else np.nan,
        int(model3c.nobs) if 'Incentive_CAT' in model_3.columns else np.nan,
        int(model3c.nobs) if 'Incentive_CAT' in model_3.columns else np.nan,
        int(model4.nobs)],
    'N_Variables': [
        len(model1.params),
        len(pooled_model.params),
        len(fe_model.params),
        len(twoway_model.params),
        len(model3a.params),
        len(model3b.params),
        len(model3c.params) if 'Incentive_CAT' in model_3.columns else np.nan,
        len(model3c.params) if 'Incentive_CAT' in model_3.columns else np.nan,
        len(model3c.params) if 'Incentive_CAT' in model_3.columns else np.nan,
        len(model4.params)],
    'R2': [
        model1.rsquared,
        pooled_model.rsquared,
        fe_model.rsquared,
        twoway_model.rsquared,
        model3a.rsquared,
        model3b.rsquared,
        model3c.rsquared if 'Incentive_CAT' in model_3.columns else np.nan,
        model3c.rsquared if 'Incentive_CAT' in model_3.columns else np.nan,
        model3c.rsquared if 'Incentive_CAT' in model_3.columns else np.nan,
        model4.rsquared],
    'Treatment_Effect': [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        model3a.params.get('pull_x_post', np.nan),
        model3b.params.get('pull_x_post', np.nan),
        cat1,
        cat2,
        cat3,
        np.nan],
    'P_Value_Treatment': [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        model3a.pvalues.get('pull_x_post', np.nan),
        model3b.pvalues.get('pull_x_post', np.nan),
        cat1_p,
        cat2_p,
        cat3_p,
        np.nan],
    'Specification': [
        'RUCC origin-dest flows',
        'All variables',
        'Time-varying only',
        'Time-varying + year FE',
        'Treatment only',
        'Treatment + controls',
        'Category 1 (≤$5K)',
        'Category 2 ($5K-$10K)',
        'Category 3 (>$10K)',
        'Lagged DV + controls']})
save_point(master_comparison, '10-1_MASTER_MODEL_COMPARISON.csv',
           'Master comparison of all models')
 
print(master_comparison.to_string(index=False))
print()
 
# ===========================================================
# 10.2: HYPOTHESIS 1 - PROGRAM TIMING EFFECTS
# ===========================================================
print("10.2: Generating Hypothesis 1 results (Program Timing)...\n")
 
if 'COVID_program' in panel.columns:
    precovid_counties = panel[panel['COVID_program'] == 0]['FIPS'].unique()
    covid_counties = panel[panel['COVID_program'] == 1]['FIPS'].unique()
 
    covid_period = panel[panel['Year'] >= 2020]
 
    precovid_effect = covid_period[covid_period['FIPS'].isin(precovid_counties)]['move_net'].mean()
    covid_effect = covid_period[covid_period['FIPS'].isin(covid_counties)]['move_net'].mean()
 
    h1_results = pd.DataFrame({
        'Program_Timing': ['Pre-COVID (2011-2019)', 'COVID-Era (2020-2021)'],
        'Mean_Net_Migration': [precovid_effect, covid_effect],
        'N_Counties': [len(precovid_counties), len(covid_counties)],
        'Difference': [np.nan, covid_effect - precovid_effect]})
 
    save_point(h1_results, 'Hypothesis1_program_timing.csv',
               'H1: Program timing effects')
 
    print(h1_results.to_string(index=False))
    print()
 
# ===========================================================
# 10.3: HYPOTHESIS 2 - INCENTIVE EFFECTIVENESS
# ===========================================================
print("10.3: Generating Hypothesis 2 results (Incentive vs Non-Incentive)...\n")
 
h2_summary = panel.groupby('has_incentive').agg({
    'move_net': ['mean', 'median', 'std', 'count'],
    'total_population': 'mean',
    'median_hh_income': 'mean',
    'unemploy_rate': 'mean',
    'FIPS': 'nunique'}).round(2)
 
h2_summary.columns = ['_'.join(col) for col in h2_summary.columns]
h2_summary.reset_index(inplace=True)
h2_summary['has_incentive'] = h2_summary['has_incentive'].map({0: 'No_Incentive', 1: 'Has_Incentive'})
 
save_point(h2_summary, 'H2_incentive_comparison.csv',
           'H2: Incentive effectiveness')
 
print(h2_summary.to_string(index=False))
print()

# Add DiD treatment effects
print("DiD Treatment Effects (from Model 3):")
print(f"  Simple DiD: {model3a.params.get('pull_x_post', np.nan):.2f} (p={model3a.pvalues.get('pull_x_post', np.nan):.4f})")
print(f"  DiD + Controls: {model3b.params.get('pull_x_post', np.nan):.2f} (p={model3b.pvalues.get('pull_x_post', np.nan):.4f})")
print()
 
# ===========================================================
# 10.4: HYPOTHESIS 3 - RURAL-URBAN MIGRATION FLOWS
# ===========================================================
print("10.4: Generating Hypothesis 3 results (Rural-Urban Flows)...\n")
 
if 'RUC_code_origin' in gravity_df.columns and 'RUC_code_dest' in gravity_df.columns:
    gravity_df['origin_type'] = gravity_df['RUC_code_origin'].apply(
        lambda x: 'Urban' if x <= 3 else 'Rural')
    gravity_df['dest_type'] = gravity_df['RUC_code_dest'].apply(
        lambda x: 'Urban' if x <= 3 else 'Rural')
 
    flow_summary = gravity_df.groupby(['origin_type', 'dest_type'])['movers'].sum().reset_index()
    flow_summary.columns = ['Origin', 'Destination', 'Total_Movers']
    flow_summary['Pct_of_Total'] = (flow_summary['Total_Movers'] /
                                     flow_summary['Total_Movers'].sum() * 100).round(2)
 
    save_point(flow_summary, 'H3_flow_analysis.csv',
               'H3: Rural-urban migration flows')
 
    print(flow_summary.to_string(index=False))
    print()
 
# ===========================================================
# 10.5: DESCRIPTIVE STATISTICS BY INCENTIVE STATUS
# ===========================================================
print("10.5: Generating descriptive statistics tables...\n")
 
desc_vars = [
    'total_population', 'move_net', 'RUC_code',
    'median_hh_income', 'median_home_value', 'median_property_taxes',
    'college_degree', 'unemploy_rate',
    'Amenity_scale', 'BEA_PCI', 'RPP']
 
descriptive_stats = panel.groupby('has_incentive')[desc_vars].agg(['mean', 'std', 'count'])
descriptive_stats.columns = ['_'.join(col) for col in descriptive_stats.columns]
descriptive_stats.reset_index(inplace=True)
descriptive_stats['has_incentive'] = descriptive_stats['has_incentive'].map({0: 'Control', 1: 'Treatment'})
 
save_point(descriptive_stats, '10-5_DESCRIPTIVE_STATS_BY_INCENTIVE.csv',
           'Descriptive statistics by group')
 
print(descriptive_stats.to_string(index=False))
print()

# ===========================================================
# 10.6: KEY FINDINGS SUMMARY
# ===========================================================
print("10.6: Key Findings Summary\n")
print("="*49)
 
key_findings = f"""
KEY FINDINGS SUMMARY
====================
 
SAMPLE CHARACTERISTICS:
  • Total observations: {len(panel):,}
  • Counties: {panel['FIPS'].nunique():,}
  • Years: {panel['Year'].min()}-{panel['Year'].max()}
  • Incentive counties: {panel[panel['has_incentive']==1]['FIPS'].nunique():,}
  • Control counties: {panel[panel['has_incentive']==0]['FIPS'].nunique():,}
 
MODEL PERFORMANCE:
  • Gravity Model R²: {model1.rsquared:.4f}
  • Panel Two-Way FE R²: {twoway_model.rsquared:.4f}
  • DiD (with controls) R²: {model3b.rsquared:.4f}
  • Dynamic Panel R²: {model4.rsquared:.4f}
 
TREATMENT EFFECTS (DiD):
  • Simple specification: {model3a.params.get('pull_x_post', np.nan):.2f} net migrants
    (p-value: {model3a.pvalues.get('pull_x_post', np.nan):.4f})
 
  • With controls: {model3b.params.get('pull_x_post', np.nan):.2f} net migrants
    (p-value: {model3b.pvalues.get('pull_x_post', np.nan):.4f})
 
PERSISTENCE (Dynamic Panel):
  • Coefficient (ρ): {persistence:.4f}
  • Long-run multiplier: {lr_multiplier:.4f}
  • Interpretation: {"Positive persistence - gains compound" if persistence > 0 else "Mean reversion - gains followed by losses"}
 
STATISTICAL SIGNIFICANCE:
  • Treatment effect significant: {'Yes' if model3b.pvalues.get('pull_x_post', 1) < 0.05 else 'No'}
  • Persistence significant: {'Yes' if model4.pvalues.get('move_net_lag1', 1) < 0.05 else 'No'}
"""
 
print(key_findings)
 
# Save to file
with open('10-6:KEY_FINDINGS_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(key_findings)
 
print(" Key findings saved to KEY_FINDINGS_SUMMARY.txt")
print()

# ===========================================================
# 10.7: OUTPUT FILES GENERATED
# ===========================================================
print("10.7: Output files generated:\n")
print("-" * 60)
 
output_files = [
    'MASTER_MODEL_COMPARISON.csv',
    'model1_gravity_results.csv',
    'model2_panel_fe_comparison.csv',
    'model3_did_comparison.csv',
    'model4_dynamic_results.csv',
    'hypothesis1_program_timing.csv',
    'hypothesis2_incentive_comparison.csv',
    'hypothesis3_flow_analysis.csv',
    'DESCRIPTIVE_STATS_BY_INCENTIVE.csv',
    'KEY_FINDINGS_SUMMARY.txt']
 
for f in output_files:
    print(f" {f}")
 
print("\n" + "="*49)
print("SECTION 10 COMPLETE")
print("="*49 + "\n")
 
# ===========================================================
# SECTION 11: VISUALIZATIONS
# ===========================================================
 
# -------------------------------------------------
# FIGURE 1 - MIGRATION STATUS VISUALIZATION
# -------------------------------------------------
print("="*49)
print("COUNTY MIGRATION STATUS ANALYSIS")
print("="*49 + "\n")
 
ANN = panel.copy()
 
# Annual net migration summary
annual_net = ANN.groupby('Year')['move_net'].agg(['sum', 'mean', 'median', 'std'])
print("Annual Net Migration Summary:")
print(annual_net.to_string())
print()
 
# Create migration status categories
ANN['migration_status'] = ANN['move_net'].apply(
    lambda x: 'Gaining' if x > 0 else ('Losing' if x < 0 else 'Stable'))
 
# Count by status and year
status_counts = ANN.groupby(['Year', 'migration_status']).size().unstack(fill_value=0)
 
# Create single plot
fig, ax = plt.subplots(figsize=(12, 7))
 
status_counts.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    color=['#d62728', '#2ca02c', '#ff7f0e'],
    edgecolor='black',
    linewidth=0.5)
 
ax.set_title('Counties by Migration Status (2011-2021)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Number of Counties', fontsize=14)
ax.legend(title='Migration Status', fontsize=11, title_fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3, linestyle='--')
 
# Add COVID marker
ax.axvline(x=9, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.text(9, ax.get_ylim()[1]*0.95, 'COVID-19',
        ha='center', fontsize=10, fontweight='bold')
 
plt.tight_layout()
plt.savefig('FIG1_county_migration_status.png', dpi=300, bbox_inches='tight')
print(" Figure saved: county_migration_status.png\n")
plt.show()
 
# Summary statistics
print("Overall Migration Summary:")
print(f"  Counties gaining population: {(ANN['move_net'] > 0).sum():,} ({(ANN['move_net'] > 0).sum()/len(ANN)*100:.1f}%)")
print(f"  Counties losing population: {(ANN['move_net'] < 0).sum():,} ({(ANN['move_net'] < 0).sum()/len(ANN)*100:.1f}%)")
print(f"  Counties stable (net=0): {(ANN['move_net'] == 0).sum():,} ({(ANN['move_net'] == 0).sum()/len(ANN)*100:.1f}%)")
print()
 
# By year breakdown
print("Migration Status by Year:")
status_pct = status_counts.div(status_counts.sum(axis=1), axis=0) * 100
print(status_pct.round(1).to_string())
 
print("\n" + "="*49)
 
# -------------------------------------------------
# FIGURE 2 - NET MIGRATION BY RUCC and RUCC TRENDS
# -------------------------------------------------
Urban_migrate = panel.copy()
# RUCC categories
rucc_labels = {
    1: 'Metro >1M',
    2: 'Metro 250K-1M',
    3: 'Metro <250K',
    4: 'Suburban 20K+, Metro',
    5: 'Suburban 20K+, Non-metro',
    6: 'Suburban 2.5-20K, Metro',
    7: 'Rural 2.5-20K, Non-metro',
    8: 'Rural <2.5K, Metro',
    9: 'Rural <2.5K, Non-metro'}
 
Urban_migrate['RUCC_label'] = Urban_migrate['RUC_code'].map(rucc_labels)
 
# Migration by RUCC
rucc_migration = Urban_migrate.groupby('RUC_code').agg({
    'move_net': ['mean', 'median', 'sum'],
    'total_population': 'mean',
    'FIPS': 'nunique'}).round(2)
rucc_migration.columns = ['_'.join(col) for col in rucc_migration.columns]
print("\nMigration by RUCC Category:")
print(rucc_migration)
 
# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
 
# Average net migration by RUCC
rucc_avg = Urban_migrate.groupby('RUCC_label')['move_net'].mean().sort_values()
rucc_avg.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0].set_title('Average Net Migration by RUCC Category', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Average Net Movers')
axes[0].set_ylabel('')
 
# RUCC over time
rucc_time = Urban_migrate.groupby(['Year', 'RUC_code'])['move_net'].mean().unstack()
rucc_time[[1, 3, 5, 7, 9]].plot(ax=axes[1], marker='o', linewidth=2)
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].set_title('Net Migration Trends by RUCC', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Net Movers')
axes[1].legend(title='RUCC', labels=[
    'Metro >1M',
    'Metro <250K',
    'Urban 20K+ Non-metro',
    'Urban 2.5-20K Non-metro',
    'Rural Non-metro'])
axes[1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('FIG2_3_RUCC_patterns.png', dpi=300, bbox_inches='tight')
plt.show()
print("ECONOMIC FACTORS")

# -------------------------------------------------
# FIGURE 3: IRS BILATERAL MIGRATION FLOWS (EXPANDED 3×3)
# -------------------------------------------------
print("="*49)
print("FIGURE 3: IRS BILATERAL MIGRATION ANALYSIS (9 FLOW TYPES)")
print("="*49 + "\n")
 
# = = = = = = = = = = = = = = = = = = = = = = = = =
# VARIATION A: State-to-State Flow Matrix (UNCHANGED)
# = = = = = = = = = = = = = = = = = = = = = = = = =
print("Creating Figure 3A: State-to-State Flow Matrix...\n")
 
# Extract state FIPS (first 2 digits)
IRS_bilateral['origin_state'] = IRS_bilateral['origin_FIPS'].astype(str).str[:2]
IRS_bilateral['dest_state'] = IRS_bilateral['dest_FIPS'].astype(str).str[:2]
 
# Aggregate by state
state_flows = IRS_bilateral.groupby(['origin_state', 'dest_state'])['movers'].sum().reset_index()
 
# Get top 20 states by total migration volume
top_states_origin = state_flows.groupby('origin_state')['movers'].sum().nlargest(20).index
top_states_dest = state_flows.groupby('dest_state')['movers'].sum().nlargest(20).index
top_states = list(set(top_states_origin) | set(top_states_dest))[:20]
 
# Filter to top states
state_flows_top = state_flows[
    state_flows['origin_state'].isin(top_states) &
    state_flows['dest_state'].isin(top_states)]
 
# Create pivot table
flow_matrix_state = state_flows_top.pivot_table(
    values='movers',
    index='origin_state',
    columns='dest_state',
    aggfunc='sum',
    fill_value=0)
 
# State FIPS to abbreviation mapping (top 20)
state_map = {
    '01': 'AL', '04': 'AZ', '06': 'CA', '08': 'CO', '12': 'FL',
    '13': 'GA', '17': 'IL', '18': 'IN', '26': 'MI', '27': 'MN',
    '29': 'MO', '34': 'NJ', '36': 'NY', '37': 'NC', '39': 'OH',
    '42': 'PA', '48': 'TX', '51': 'VA', '53': 'WA', '55': 'WI'}
 
flow_matrix_state.index = flow_matrix_state.index.map(lambda x: state_map.get(x, x))
flow_matrix_state.columns = flow_matrix_state.columns.map(lambda x: state_map.get(x, x))
 
# Plot
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(flow_matrix_state,
            cmap='YlOrRd',
            annot=False,
            cbar_kws={'label': 'Total Migrants (2011-2021)'},
            linewidths=0.5,
            linecolor='white',
            square=True,
            ax=ax)
 
ax.set_title('State-to-State Migration Flows (Top 20 States, 2011-2021)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Destination State', fontsize=12)
ax.set_ylabel('Origin State', fontsize=12)
 
plt.tight_layout()
plt.savefig('fig3a_state_flow_matrix.png', dpi=300, bbox_inches='tight')
print(" Figure 3A saved: fig3a_state_flow_matrix.png\n")
plt.close()
 
# = = = = = = = = = = = = = = = = = = = = = = = = =
# VARIATION B: RUCC Flow Matrix (9×9 - UNCHANGED)
# = = = = = = = = = = = = = = = = = = = = = = = = =
print("Creating Figure 3: RUCC Flow Matrix (9×9)...\n")
 
# Ensure FIPS formatting matches
IRS_bilateral['origin_FIPS'] = IRS_bilateral['origin_FIPS'].astype(str).str.zfill(5)
IRS_bilateral['dest_FIPS'] = IRS_bilateral['dest_FIPS'].astype(str).str.zfill(5)
 
# Check if rucc_all exists
if 'rucc_all' not in dir():
    print(" Creating rucc_all from panel...")
    rucc_all = panel[['FIPS', 'Year', 'RUCcode']].copy()
    rucc_all['FIPS'] = rucc_all['FIPS'].astype(str).str.zfill(5)
else:
    rucc_all['FIPS'] = rucc_all['FIPS'].astype(str).str.zfill(5)
    # Handle column name variations
    if 'RUC_code' in rucc_all.columns:
        rucc_all.rename(columns={'RUC_code': 'RUCcode'}, inplace=True)
 
# Merge RUCC codes for origin
bilateral_rucc = IRS_bilateral.merge(
    rucc_all.rename(columns={'FIPS': 'origin_FIPS', 'RUCcode': 'origin_RUCC'}),
    on=['origin_FIPS', 'Year'],
    how='left')
 
# Merge RUCC codes for destination
bilateral_rucc = bilateral_rucc.merge(
    rucc_all.rename(columns={'FIPS': 'dest_FIPS', 'RUCcode': 'dest_RUCC'}),
    on=['dest_FIPS', 'Year'],
    how='left')
 
# Check merge success
print(f"  After merge: {len(bilateral_rucc)} rows")
print(f"  With origin_RUCC: {bilateral_rucc['origin_RUCC'].notna().sum()}")
print(f"  With dest_RUCC: {bilateral_rucc['dest_RUCC'].notna().sum()}")
 
# Remove missing
bilateral_rucc = bilateral_rucc.dropna(subset=['origin_RUCC', 'dest_RUCC'])
print(f"  After dropna: {len(bilateral_rucc)} rows\n")
 
if len(bilateral_rucc) > 0:
    # Aggregate by RUCC
    rucc_flows = bilateral_rucc.groupby(['origin_RUCC', 'dest_RUCC'])['movers'].sum().reset_index()
 
    # Create pivot
    flow_matrix_rucc = rucc_flows.pivot_table(
        values='movers',
        index='origin_RUCC',
        columns='dest_RUCC',
        aggfunc='sum',
        fill_value=0)
 
    # Ensure all 1-9 are present
    for i in range(1, 10):
        if i not in flow_matrix_rucc.index:
            flow_matrix_rucc.loc[i] = 0
        if i not in flow_matrix_rucc.columns:
            flow_matrix_rucc[i] = 0
 
    flow_matrix_rucc = flow_matrix_rucc.sort_index().sort_index(axis=1)
 
    # Plot
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(flow_matrix_rucc,
                cmap='YlOrRd',
                annot=True,
                fmt='.0f',
                annot_kws={'fontsize': 7},
                cbar_kws={'label': 'Total Migrants (2011-2021)'},
                linewidths=1,
                linecolor='gray',
                square=True,
                ax=ax)
 
    ax.set_title('Migration Flows by RUCC Classification (2011-2021)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Destination RUCC (1-3: Urban | 4-6: Suburban | 7-9: Rural)', fontsize=11)
    ax.set_ylabel('Origin RUCC (1-3: Urban | 4-6: Suburban | 7-9: Rural)', fontsize=11)
 
    # Add dividing lines to show Urban/Suburban/Rural blocks
    ax.axhline(y=3, color='black', linewidth=2, alpha=0.7)
    ax.axhline(y=6, color='black', linewidth=2, alpha=0.7)
    ax.axvline(x=3, color='black', linewidth=2, alpha=0.7)
    ax.axvline(x=6, color='black', linewidth=2, alpha=0.7)
 
    plt.tight_layout()
    plt.savefig('fig3b_rucc_flow_matrix.png', dpi=300, bbox_inches='tight')
    print(" Figure 3 saved: fig3b_rucc_flow_matrix.png\n")
    plt.close()
else:
    print(" Figure 3c skipped: RUCC merge returned no data\n")
 
# = = = = = = = = = = = = = = = = = = = = = = = = =
# VARIATION C: Time Evolution Heatmap (EXPANDED TO 9 TYPES)
# = = = = = = = = = = = = = = = = = = = = = = = = =
print("Creating Figure 3C: Time Evolution Heatmap (9 Flow Types)...\n")
 
if len(bilateral_rucc) > 0:
    # Create 3-category classification
    bilateral_time = bilateral_rucc.copy()
 
    # Classify origin
    bilateral_time['origin_category'] = 'Rural'
    bilateral_time.loc[bilateral_time['origin_RUCC'] <= 3, 'origin_category'] = 'Urban'
    bilateral_time.loc[(bilateral_time['origin_RUCC'] > 3) &
                       (bilateral_time['origin_RUCC'] <= 6), 'origin_category'] = 'Suburban'
 
    # Classify destination
    bilateral_time['dest_category'] = 'Rural'
    bilateral_time.loc[bilateral_time['dest_RUCC'] <= 3, 'dest_category'] = 'Urban'
    bilateral_time.loc[(bilateral_time['dest_RUCC'] > 3) &
                       (bilateral_time['dest_RUCC'] <= 6), 'dest_category'] = 'Suburban'
 
    # Create 9 flow types
    bilateral_time['flow_type'] = (bilateral_time['origin_category'] + '→' +
                                   bilateral_time['dest_category'])
 
    # Aggregate by year and flow type
    time_flows = bilateral_time.groupby(['Year', 'flow_type'])['movers'].sum().reset_index()
 
    # Pivot
    flow_matrix_time = time_flows.pivot(
        index='flow_type',
        columns='Year',
        values='movers')
 
    # Reorder rows for logical presentation
    flow_order = [
        'Urban→Urban', 'Urban→Suburban', 'Urban→Rural',
        'Suburban→Urban', 'Suburban→Suburban', 'Suburban→Rural',
        'Rural→Urban', 'Rural→Suburban', 'Rural→Rural']
 
    flow_matrix_time = flow_matrix_time.reindex(
        [f for f in flow_order if f in flow_matrix_time.index])
 
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(flow_matrix_time,
                cmap='YlGnBu',
                annot=True,
                fmt='.0f',
                annot_kws={'fontsize': 7},
                cbar_kws={'label': 'Total Migrants'},
                linewidths=0.5,
                linecolor='white',
                ax=ax)
 
    ax.set_title('Migration Flow Trends by Type (2011-2021)\nUrban (1-3) | Suburban (4-6) | Rural (7-9)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Flow Type', fontsize=12)
 
    # Add horizontal dividers between origin categories
    ax.axhline(y=3, color='black', linewidth=2, alpha=0.7)
    ax.axhline(y=6, color='black', linewidth=2, alpha=0.7)
 
    # Add COVID annotation
    covid_col_index = list(flow_matrix_time.columns).index(2020) if 2020 in flow_matrix_time.columns else 9
    ax.axvline(x=covid_col_index + 0.5, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.text(covid_col_index + 0.5, -0.7, 'COVID-19', color='red', fontsize=11,
            ha='center', fontweight='bold')
 
    plt.tight_layout()
    plt.savefig('fig3c_time_evolution_heatmap_9types.png', dpi=300, bbox_inches='tight')
    print(" Figure 3C saved: fig3c_time_evolution_heatmap_9types.png\n")
    plt.close()
 
    # = = = = = = = = = = = = = = = = = = = = = = = = =
    # Summary Statistics (EXPANDED TO 9 TYPES)
    # = = = = = = = = = = = = = = = = = = = = = = = = =
    print("Summary Statistics for Bilateral Flows (9 Flow Types):")
    print("-" * 60)
 
    total_flows = IRS_bilateral['movers'].sum()
    print(f"Total migration events (2011-2021): {total_flows:,.0f}\n")
 
    flow_summary = bilateral_time.groupby('flow_type')['movers'].sum().sort_values(ascending=False)
    flow_pct = (flow_summary / flow_summary.sum() * 100).round(2)
 
    summary_df = pd.DataFrame({
        'Flow_Type': flow_summary.index,
        'Total_Movers': flow_summary.values,
        'Percentage': flow_pct.values})
 
    print("Migration by Flow Type (9 Categories):")
    print(summary_df.to_string(index=False))
    print()
 
    # Aggregate to 3 origin categories
    origin_summary = bilateral_time.groupby('origin_category')['movers'].sum().sort_values(ascending=False)
    origin_pct = (origin_summary / origin_summary.sum() * 100).round(2)
 
    print("\nMigration by Origin Category:")
    print(f"  Urban:    {origin_summary.get('Urban', 0):>12,.0f}  ({origin_pct.get('Urban', 0):>5.1f}%)")
    print(f"  Suburban: {origin_summary.get('Suburban', 0):>12,.0f}  ({origin_pct.get('Suburban', 0):>5.1f}%)")
    print(f"  Rural:    {origin_summary.get('Rural', 0):>12,.0f}  ({origin_pct.get('Rural', 0):>5.1f}%)")
    print()
 
    # Aggregate to 3 destination categories
    dest_summary = bilateral_time.groupby('dest_category')['movers'].sum().sort_values(ascending=False)
    dest_pct = (dest_summary / dest_summary.sum() * 100).round(2)
 
    print("Migration by Destination Category:")
    print(f"  Urban:    {dest_summary.get('Urban', 0):>12,.0f}  ({dest_pct.get('Urban', 0):>5.1f}%)")
    print(f"  Suburban: {dest_summary.get('Suburban', 0):>12,.0f}  ({dest_pct.get('Suburban', 0):>5.1f}%)")
    print(f"  Rural:    {dest_summary.get('Rural', 0):>12,.0f}  ({dest_pct.get('Rural', 0):>5.1f}%)")
    print()
 
    save_point(summary_df, 'fig3_flow_type_summary_9types.csv',
               'Migration flow summary statistics (9 types)')
else:
    print(" Figure 3C skipped: No RUCC data available\n")
 
print("="*49)
print("FIGURE 3 COMPLETE (9 FLOW TYPES)")
print("="*49)
print()
 
print("FILES GENERATED:")
print("   fig3a_state_flow_matrix.png")
if len(bilateral_rucc) > 0:
    print("   fig3b_rucc_flow_matrix.png (9×9 with dividers)")
    print("   fig3c_time_evolution_heatmap_9types.png (9 rows)")
    print("   fig3_flow_type_summary_9types.csv")
else:
    print("   fig3b and fig3c skipped (RUCC merge issue)")
print()
 
# -------------------------------------------------
# FIGURE 4 - MODEL FIT COMPARISON
# -------------------------------------------------
print("="*49)
print("SECTION 11: VISUALIZATIONS")
print("="*49 + "\n")
 
r2_data = master_comparison[['Model', 'R2']].dropna()
plt.figure(figsize=(10,6))
plt.barh(r2_data['Model'], r2_data['R2'])
plt.xlabel('R²')
plt.title('Model Fit Comparison')
plt.xlim(0, 1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_model_comparison.png', dpi=300, bbox_inches='tight')
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
 
# -------------------------------------------------
# FIGURE 5: Q-Q Plot for Residual Normality
# -------------------------------------------------
# Use model's built-in residuals
residuals = twoway_model.resids
fig, ax = plt.subplots(figsize=(10, 8))

stats.probplot(residuals, dist="norm", plot=ax)
 
ax.set_title('Q-Q Plot: Residual Normality Assessment',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Theoretical Quantiles', fontsize=14)
ax.set_ylabel('Ordered Values', fontsize=14)
ax.grid(alpha=0.3, linestyle='--')
 
# Add statistics box
textstr = f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}\nN: {len(residuals):,}'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
 
# Add interpretation note
fig.text(0.5, 0.02,
         'Note: Deviations in tails indicate outliers. With N=34,553 and cluster-robust SE, FE estimates remain consistent.',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
 
plt.tight_layout()
plt.savefig('fig5_qq_plot.png', dpi=300, bbox_inches='tight')
print(" Figure 5 saved: fig6_qq_plot.png\n")
plt.show()
 
# ===========================================================
# SECTION 12: DIAGNOSTICS
# ===========================================================
print("="*49)
print("SECTION 12: DIAGNOSTICS")
print("="*49 + "\n")
'''
  0. VIF: Completed during EDA
  1. Residual analysis
  2. Parallel trends test
  3. Treatment-control balance
  4. Outlier detection
'''
avg_diff = np.nan
std_diff = np.nan
outlier_treatment_pct = np.nan
overall_treatment_pct = np.nan
 
# -------------------------------------------------
# 12.1: Residual Analysis
# -------------------------------------------------
print("12.1: Residual Analysis\n")
 
residuals = twoway_model.resids
residual_stats = pd.DataFrame({
    'Statistic': ['Mean', 'Std_Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
    'Value': [
        residuals.mean(),
        residuals.std(),
        residuals.min(),
        residuals.max(),
        residuals.skew(),
        residuals.kurtosis()]}).round(2)
 
print(residual_stats.to_string(index=False))
print()
 
# Normality test
from scipy.stats import shapiro, jarque_bera
 
if len(residuals) <= 5000:
    stat, pval = shapiro(residuals.sample(min(5000, len(residuals))))
    test_name = "Shapiro-Wilk"
else:
    stat, pval = jarque_bera(residuals)
    test_name = "Jarque-Bera"
 
print(f"{test_name} normality test: stat={stat:.4f}, p={pval:.4f}")
 
if pval < 0.05:
    print("  Residuals deviate from normality (p < 0.05)")
    print("  Note: With large samples, FE estimators remain consistent")
else:
    print("   Residuals approximately normal (p ≥ 0.05)")
 
print()
 
save_point(residual_stats, '12-1_diagnostics_residuals.csv', 'Residual statistics')
 
# -------------------------------------------------
# 12.2: Parallel Trends Test (DiD Assumption)
# -------------------------------------------------
print("12.2: Parallel Trends Test (DiD Assumption)\n")
 
if 'PULL' in panel.columns:
    trends = panel.groupby(['Year', 'PULL'])['move_net'].mean().reset_index()
    trends_pivot = trends.pivot(index='Year', columns='PULL', values='move_net')
 
    pre_treatment = trends_pivot[trends_pivot.index < 2020]
 
    print("Pre-treatment period (2011-2019):")
    print(pre_treatment.to_string())
    print()
 
    # Calculate trend differences
    pre_treatment['Difference'] = pre_treatment[1] - pre_treatment[0]
    avg_diff = pre_treatment['Difference'].mean()
    std_diff = pre_treatment['Difference'].std()
 
    print(f"Average pre-treatment difference: {avg_diff:.2f}")
    print(f"Std dev of differences: {std_diff:.2f}")
    print()
 
    if abs(avg_diff) < 50:
        print("Parallel trends assumption reasonable")
        print("  Average difference small relative to variation")
    else:
        print(" Warning: Pre-treatment trends may diverge")
        print("  Consider event study or propensity score matching")
 
    print()
 
    save_point(trends, '12-2_diagnostics_parallel_trends.csv',
               'Parallel trends data')
else:
    print(" PULL variable not found in panel")
    print()
 
# -------------------------------------------------
# 12.3: Treatment-Control Balance
# -------------------------------------------------
print("12.3: Treatment-Control Balance Check\n")
 
print("12.3: Treatment-Control Balance\n")
 
# Baseline comparison: year 2014(post-2008 housing crisis, pre-2015 large city emigration)
baseline = panel[panel['Year'] == 2014].copy()
 
# Select variables for balance check
balance_vars = ['total_population', 'median_hh_income', 'unemploy_rate',
                'median_home_value', 'college_degree', 'BEA_pci']
 
# Calculate means by group
balance_list = []
treat_var = 'has_incentive' if 'has_incentive' in panel.columns else 'PULL'
 
for var in balance_vars:
    if var not in baseline.columns:
        continue
    
    treat_mean = baseline[baseline[treat_var] == 1][var].mean()
    control_mean = baseline[baseline[treat_var] == 0][var].mean()
    diff = treat_mean - control_mean
    pct_diff = (diff / control_mean * 100) if control_mean != 0 else 0
    
    balance_list.append({
        'Variable': var,
        'Treatment': treat_mean,
        'Control': control_mean,
        'Difference': diff,         # ← CREATE this column FIRST
        'Pct_Diff': pct_diff})
 
# Create DataFrame
balance = pd.DataFrame(balance_list)
balance = balance.set_index('Variable')
 
# NOW convert to numeric (after DataFrame is created with all columns)
balance['Difference'] = pd.to_numeric(balance['Difference'], errors='coerce')
balance['Pct_Diff'] = pd.to_numeric(balance['Pct_Diff'], errors='coerce')
 
print("Treatment-Control Balance:")
print(balance.round(2))
print()
 
# Check for large imbalances
large_imbalances = balance[abs(balance['Pct_Diff']) > 20]
if len(large_imbalances) > 0:
    print(" Large imbalances detected (>20%):")
    print(large_imbalances[['Pct_Diff']].round(1))
    print()
 
save_point(balance.reset_index().rename(columns={'index': 'Variable'}),
           '12-3_diagnostics_balance.csv', 'Treatment-control balance')
 
# -------------------------------------------------
# 12.4: Outlier Detection
# -------------------------------------------------
print("12.4: Outlier Detection\n")
 
outlier_threshold = 3
z_scores = np.abs((panel['move_net'] - panel['move_net'].mean()) /
                  panel['move_net'].std())
 
outliers = panel[z_scores > outlier_threshold].copy()
outliers['Z_Score'] = z_scores[z_scores > outlier_threshold]
 
print(f"Observations beyond {outlier_threshold} std dev: {len(outliers):,} ({len(outliers)/len(panel)*100:.2f}%)")
print()
 
if len(outliers) > 0:
    top_outliers = outliers.nlargest(10, 'move_net')[
        ['FIPS', 'Year', 'move_net', 'Z_Score', 'has_incentive']]
 
    print("Top 10 outlier counties (highest net migration):")
    print(top_outliers.to_string(index=False))
    print()
 
    # Check if outliers concentrated in treatment group
    outlier_treatment_pct = (outliers['has_incentive'].sum() /
                             len(outliers) * 100)
    overall_treatment_pct = (panel['has_incentive'].sum() /
                            len(panel) * 100)
 
    print(f"Outliers in treatment group: {outlier_treatment_pct:.1f}%")
    print(f"Overall treatment rate: {overall_treatment_pct:.1f}%")
 
    if abs(outlier_treatment_pct - overall_treatment_pct) > 10:
        print("            Outliers disproportionately in treatment/control")
        print("Consider robustness check excluding outliers")
    else:
        print("Outliers balanced across groups")
 
    print()
 
    save_point(outliers[['FIPS', 'Year', 'move_net', 'Z_Score', 'has_incentive']],
               '12-4_diagnostics_outliers.csv', 'Outlier observations')
else:
    print(" No extreme outliers detected")
    print()
 
# -------------------------------------------------
# 12.5: Diagnostic Summary Report
# -------------------------------------------------
print("12.5: Diagnostic Summary Report\n")
print("="*49)
 
diagnostic_summary = f"""
POST-ESTIMATION DIAGNOSTIC REPORT
==================================
 
1. RESIDUAL ANALYSIS:
   Mean: {residuals.mean():.2f} (target: ~0)
   Std Dev: {residuals.std():.2f}
   Normality: {' Normal' if pval >= 0.05 else ' Non-normal'}
 
   Assessment: {'Residuals meet standard assumptions' if pval >= 0.05 and abs(residuals.mean()) < 10 else 'Review residual patterns'}
 
2. PARALLEL TRENDS (DiD):
   Pre-treatment avg difference: {avg_diff:.2f} if 'PULL' in panel.columns else 'N/A'
   Pre-treatment std dev: {std_diff:.2f} if 'PULL' in panel.columns else 'N/A'
 
   Assessment: {' Parallel trends reasonable' if 'PULL' in panel.columns and abs(avg_diff) < 50 else ' Review trends'}
 
3. TREATMENT-CONTROL BALANCE:
   Population difference: {balance.loc['total_population', 'Pct_Diff']:.1f}%
   Income difference: {balance.loc['median_hh_income', 'Pct_Diff']:.1f}%
 
   Assessment: {' Groups reasonably balanced' if abs(balance['Pct_Diff']).max() < 20 else ' Large imbalances exist'}
 
4. OUTLIERS:
   Count: {len(outliers):,} ({len(outliers)/len(panel)*100:.2f}%)
   Treatment concentration: {outlier_treatment_pct:.1f}% (overall: {overall_treatment_pct:.1f}%)
 
   Assessment: {' Minimal outliers' if len(outliers)/len(panel) < 0.05 else ' Review outliers'}
 
OVERALL ASSESSMENT:
{' Models meet standard assumptions for causal inference' if (pval >= 0.05 and abs(avg_diff) < 50 and len(outliers)/len(panel) < 0.05) else ' Some assumptions violated - interpret with caution'}
 
RECOMMENDATIONS:
- {'Proceed with reported results' if pval >= 0.05 else 'Consider robust standard errors'}
- {'No additional robustness checks needed' if len(outliers)/len(panel) < 0.01 else 'Run sensitivity analysis excluding outliers'}
- {'DiD estimates credible' if 'PULL' in panel.columns and abs(avg_diff) < 50 else 'Consider alternative identification strategies'}
"""
 
print(diagnostic_summary)
with open('12-5:DIAGNOSTIC_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(diagnostic_summary)
 
print("\n Diagnostic summary saved to DIAGNOSTIC_SUMMARY.txt")
 
print("="*49)
print("SECTION 12 COMPLETE")
print("="*49 + "\n")
 
print("DIAGNOSTIC FILES GENERATED:")
print("  diagnostics_residuals.csv")
print("  diagnostics_parallel_trends.csv")
print("  diagnostics_balance.csv")
print("  diagnostics_outliers.csv")
print("  DIAGNOSTIC_SUMMARY.txt")
 
""" END """