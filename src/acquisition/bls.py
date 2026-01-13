"""Bureau of Labor Statistics (BLS) - Local Area Unemployment data acquisition."""
import pandas as pd
import time
from config.settings import API_KEY_BLS, YEARS, PROCESSED_DATA_DIR
from src.utils.helpers import (api_request, standardize_fips, define_cols, 
                               remap_fips_changes, save_point)

def get_bls_unemployment(fips_list, years, batch_size=50, sleep_s=1.0):
    """Fetch county-level annual unemployment rates from BLS API."""
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
            'annualaverage': True}
        
        response = api_request(url, method='POST', json_data=payload)
        
        if response is None:
            print(f"  ⚠️  Failed to fetch batch {batch_num}/{total_batches}")
            continue
        try:
            data = response.json()
        except ValueError:
            print(f"  ⚠️  JSON decode error in batch {batch_num}/{total_batches}")
            continue
        
        if data.get('status') != 'REQUEST_SUCCEEDED':
            msg = data.get('message', ['Unknown error'])
            error_msg = msg[0] if isinstance(msg, list) else msg
            print(f"  ⚠️  Batch {batch_num}/{total_batches} error: {error_msg}")
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
                        'unemploy_rate': pd.to_numeric(item.get('value'), errors='coerce')})
        
        print(f"  Batch {batch_num}/{total_batches}")
        time.sleep(sleep_s)
    
    return pd.DataFrame(rows)

def run():
    """Main BLS data acquisition function."""
    print("="*49)
    print("BLS LAUS DATA ACQUISITION")
    print("="*49 + "\n")
    
    # Get FIPS list from Census data
    census_file = PROCESSED_DATA_DIR / 'Census_import.csv'
    
    if not census_file.exists():
        print("❌ Census data not found")
        print("   Run Census acquisition first (Step 1)\n")
        return
    
    print("Loading FIPS list from Census data...")
    census_df = pd.read_csv(census_file)
    fips_list = census_df['FIPS'].dropna().unique()
    
    print(f"Fetching unemployment data for {len(fips_list):,} counties...\n")
    bls_df = get_bls_unemployment(fips_list, YEARS)
    
    if bls_df.empty:
        print("\n❌ No BLS data downloaded")
        return
    
    # Clean and standardize
    bls_df = standardize_fips(bls_df)
    bls_df = define_cols(bls_df)
    bls_df = remap_fips_changes(bls_df, fips_cols=['FIPS'])
    
    save_point(bls_df, 'BLS_import.csv', f"{len(bls_df):,} annual unemployment observations")
    print(f"\nBLS data acquisition complete")
    print(f"Unique FIPS: {bls_df['FIPS'].nunique():,}\n")

if __name__ == '__main__':
    run()
