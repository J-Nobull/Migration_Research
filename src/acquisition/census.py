"""U.S. Census Bureau - American Community Survey (ACS-5) data acquisition."""
import pandas as pd
import requests
import time
import zipfile
import io
from config.settings import API_KEY_CENSUS, YEARS
from src.utils.helpers import (api_request, standardize_fips,  
                               define_cols, save_point)

# ACS 5-year estimates variable list (64 variables)
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

def _chunk_list(items, chunk_size):
    """Split list into chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def fetch_census_batch(year, var_codes):
    """Fetch Census batch for one year."""
    url = f"https://api.census.gov/data/{int(year)}/acs/acs5"
    params = {
        'get': ','.join(var_codes),
        'for': 'county:*',
        'key': API_KEY_CENSUS}
    
    response = api_request(url, params=params)
    
    if response is None:
        return pd.DataFrame()
    try:
        data = response.json()
    except ValueError:
        print(f"  ⚠️  JSON decode error for {year}")
        return pd.DataFrame()
    
    # Check for API errors
    if isinstance(data, dict) and 'error' in data:
        print(f"  ⚠️  API error for {year}: {data['error']}")
        return pd.DataFrame()
    
    # Check for empty response
    if not data or len(data) <= 1:
        print(f"  ⚠️  No rows returned for {year}")
        return pd.DataFrame()
    
    return pd.DataFrame(data[1:], columns=data[0])

def run():
    """Main Census data acquisition function."""
    print("="*49)
    print("CENSUS ACS DATA ACQUISITION")
    print("="*49 + "\n")
    
    all_vars = list(ACS_VARS.keys())
    batches = _chunk_list(all_vars, 45)
    frames = []
    
    for year in YEARS:
        print(f"Fetching ACS {year}...")
        parts = []
        
        for batch in batches:
            batch_df = fetch_census_batch(year, ['NAME'] + batch)
            if batch_df.empty:
                parts = []
                break
            parts.append(batch_df)
            time.sleep(0.25) 
        
        if not parts:
            print(f"  ⚠️  Skipping year {year} due to errors")
            continue
        
        # Merge batches for this year
        year_df = parts[0]
        for part in parts[1:]:
            year_df = year_df.merge(part, on=['NAME', 'state', 'county'], how='outer')
        
        # Rename columns and process
        year_df = year_df.rename(columns=ACS_VARS)
        year_df = standardize_fips(year_df, state_col='state', county_col='county', fips_col='FIPS')
        year_df['Year'] = int(year)
        year_df = year_df.drop(columns=['NAME', 'state', 'county'], errors='ignore')
        year_df = define_cols(year_df)
        
        frames.append(year_df)
        print(f"  Saved {len(year_df):,} rows")
    
    if not frames:
        print("\n❌ No Census data downloaded")
        return
    
    # Combine all years
    out = pd.concat(frames, ignore_index=True)
    keep_cols = ['FIPS', 'Year'] + [c for c in ACS_VARS.values() if c in out.columns]
    out = out[keep_cols].copy()
    
    save_point(out, 'Census_import.csv', f"{len(out):,} county-year observations")
    
    print(f"\nCensus download complete")
    print(f"Counties: {out['FIPS'].nunique():,}")
    print(f"Years: {out['Year'].min()}-{out['Year'].max()}")
    print(f"Variables: {len([c for c in ACS_VARS.values() if c in out.columns])}\n")

    download_centroids()

def download_centroids():
    """Download county centroids (lat/lon) for use in model-1 gravity flow ."""
    print("\n" + "-"*49)
    print("Downloading county centroids (lat/lon)")
    print("-"*49)
    
    gaz_url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2021_Gazetteer/2021_Gaz_counties_national.zip"
    
    print(f"\nDownloading: {gaz_url}")
    response = api_request(gaz_url)
    
    if response is None:
        print("❌ Failed to download centroids")
        return
    
    print("Downloaded, extracting...")
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('2021_Gaz_counties_national.txt') as f:
            centroids = pd.read_csv(f, sep='\t', encoding='latin1')
    
    # Strip whitespace from column names
    centroids.columns = centroids.columns.str.strip()
    
    # Extract FIPS, lat, lon
    centroids['FIPS'] = centroids['GEOID'].astype(str).str.zfill(5)
    centroids = centroids[['FIPS', 'INTPTLAT', 'INTPTLONG']].rename(
        columns={'INTPTLAT': 'lat', 'INTPTLONG': 'lon'})
    
    # Drop Puerto Rico 
    centroids = centroids[~centroids['FIPS'].str.startswith('72')]
    
    # Alaska: keep only Anchorage (02020), the most populous location
    centroids = centroids[~((centroids['FIPS'].str.startswith('02')) & (centroids['FIPS'] != '02020'))]
    # Rename to 02001 to match AK consolidation
    centroids.loc[centroids['FIPS'] == '02020', 'FIPS'] = '02001'
    centroids = centroids.sort_values('FIPS').reset_index(drop=True)
    save_point(centroids, 'County_centroids.csv', f"{len(centroids):,} county centroids")
    
if __name__ == '__main__':
    run()
