"""Helper functions for data cleaning and processing."""
import pandas as pd
import numpy as np
import requests
import time
from typing import Optional, Dict, Any
from config.settings import PROCESSED_DATA_DIR

# API request with retry logic and error handling
def api_request(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    method: str = 'GET',
    json_data: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_delay: int = 2,
    timeout: int = 120) -> Optional[requests.Response]:

    for attempt in range(max_retries):
        try:
            if method.upper() == 'POST':
                response = requests.post(url, json=json_data, timeout=timeout)
            else:
                response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"  ⚠️  Timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
            else:
                print(f"  ❌ Timeout after {max_retries} attempts")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                print(f"  ⚠️  Rate limited, waiting {retry_delay * 2}s...")
                time.sleep(retry_delay * 2)
            else:
                print(f"  ❌ HTTP {e.response.status_code}: {e}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request error: {e}")
            return None
    return None

def standardize_fips(df, state_col=None, county_col=None, fips_col='FIPS'):
    """Standardize FIPS codes to 5-digit strings."""
    df = df.copy()
    if state_col and county_col:
        df[fips_col] = (df[state_col].astype(str).str.zfill(2) +
                       df[county_col].astype(str).str.zfill(3))
    elif fips_col in df.columns:
        df[fips_col] = df[fips_col].astype(str).str.zfill(5)
    else:
        raise ValueError("Must provide either (state_col, county_col) or fips_col")
    return df

def define_cols(df, exclude_cols=['FIPS', 'origin_FIPS', 'dest_FIPS', 'state', 'county', 'STATE']):
    """Define column types based on content."""
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
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            if (df[col].dropna() % 1 == 0).all():
                df[col] = df[col].astype('Int64')
    return df

def remap_fips_changes(df, fips_cols=['FIPS']):
    """Handle historical FIPS code changes."""
    df = df.copy()
    
    # Alaska consolidation
    alaska_codes = [f'02{str(i).zfill(3)}' for i in range(1, 999)]
    
    # Connecticut remap
    ct_remap = {
        '09110': '09003', '09120': '09001', '09130': '09007',
        '09140': '09009', '09150': '09015', '09160': '09005',
        '09170': '09009', '09180': '09011', '09190': '09013'}
    
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

def filter_dataframe(df, name='DataFrame'):
    """Filter and display DataFrame summary."""
    print(f"\n{'='*49}")
    print(f"{name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    if 'FIPS' in df.columns:
        before_count = len(df)
        df = df[df['FIPS'].str[:2] != '72'].copy()
        df = df[df['FIPS'].astype(str).str[:5].astype(int) < 57000].copy()
        after_count = len(df)
        if before_count != after_count:
            print(f"Removed {before_count - after_count:,} rows with FIPS > 56999")
        print(f"Unique FIPS: {df['FIPS'].nunique():,}")
    
    if 'Year' in df.columns:
        print(f"Years: {df['Year'].min()}-{df['Year'].max()}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values: {missing.sum():,} ({missing.sum()/df.size*100:.1f}%)")
    
    print(f"{'='*49}\n")
    return df

def save_point(df, filename, description=""):
    """Save DataFrame checkpoint."""
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Saved: {filename}")
    if description:
        print(f"  {description}")

print("5 Utility Functions Loaded")
