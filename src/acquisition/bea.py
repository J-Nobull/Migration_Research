"""Bureau of Economic Analysis (BEA) data acquisition."""
import pandas as pd
import requests
from io import BytesIO
from config.settings import API_KEY_BEA, YEARS
from src.utils.helpers import standardize_fips, define_cols, remap_fips_changes, filter_dataframe, save_point

def get_bea_data(dataset, table, line_code, geo_type, years):
    """Fetch BEA data via API."""
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
            if 'BEAAPI' in data and 'Results' in data['BEAAPI']:
                if 'Data' in data['BEAAPI']['Results']:
                    results.extend(data['BEAAPI']['Results']['Data'])
                else:
                    print(f"  ⚠ No data for {table} year {year}")
            else:
                print(f"  ⚠ Invalid response for {table} year {year}")
        else:
            print(f"  ⚠ HTTP {response.status_code} for {table} year {year}")
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def run():
    """Main BEA data acquisition function."""
    print("="*49)
    print("BEA DATA ACQUISITION")
    print("="*49 + "\n")
    
    # Regional Price Parities (RPP)
    print("Fetching Regional Price Parities...")
    rpp_metro = get_bea_data('Regional', 'MARPP', '3', 'MSA', YEARS)
    rpp_nonmetro = get_bea_data('Regional', 'PARPP', '3', 'PORT', YEARS)
    
    if not rpp_metro.empty:
        rpp_metro = rpp_metro[['GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'MSA_Code',
            'TimePeriod': 'Year',
            'DataValue': 'RPP_Metro'})
        rpp_metro = define_cols(rpp_metro, exclude_cols=['MSA_Code'])
        save_point(rpp_metro, 'BEA_RPP_Metro.csv', f"{len(rpp_metro):,} metro RPP records")
    
    if not rpp_nonmetro.empty:
        rpp_nonmetro = rpp_nonmetro[['GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'State_FIPS',
            'TimePeriod': 'Year',
            'DataValue': 'RPP_NonMetro'})
        rpp_nonmetro = define_cols(rpp_nonmetro, exclude_cols=['State_FIPS'])
        save_point(rpp_nonmetro, 'BEA_RPP_NONmetro.csv', f"{len(rpp_nonmetro):,} non-metro RPP records")
    
    # Per Capita Income
    print("\nFetching Per Capita Income...")
    pci = get_bea_data('Regional', 'CAINC1', '3', 'COUNTY', YEARS)
    
    if not pci.empty:
        pci = pci[['GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'FIPS',
            'TimePeriod': 'Year',
            'DataValue': 'BEA_PCI'})
        pci = standardize_fips(pci)
        pci = define_cols(pci)
        pci = remap_fips_changes(pci, fips_cols=['FIPS'])
        pci = filter_dataframe(pci, 'BEA PCI')
        save_point(pci, 'BEA_PCI.csv', f"{len(pci):,} county-year observations")
    
    # Real GDP
    print("\nFetching Real GDP...")
    gdp = get_bea_data('Regional', 'CAGDP1', '1', 'COUNTY', YEARS)
    
    if not gdp.empty:
        gdp = gdp[['GeoFips', 'TimePeriod', 'DataValue']].rename(columns={
            'GeoFips': 'FIPS',
            'TimePeriod': 'Year',
            'DataValue': 'BEA_GDP'})
        gdp = standardize_fips(gdp)
        gdp = define_cols(gdp)
        gdp = remap_fips_changes(gdp, fips_cols=['FIPS'])
        gdp = filter_dataframe(gdp, 'BEA GDP')
        save_point(gdp, 'BEA_GDP.csv', f"{len(gdp):,} county-year observations")
    
    # Download CBSA crosswalk for metro RPP
    print("\nDownloading CBSA delineation crosswalk...")
    cbsa_url = 'https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2013/delineation-files/list1.xls'
    response = requests.get(cbsa_url)
    cbsa = pd.read_excel(BytesIO(response.content), skiprows=2)
    
    cbsa['FIPS State Code'] = pd.to_numeric(cbsa['FIPS State Code'], errors='coerce').astype('Int64')
    cbsa['FIPS County Code'] = pd.to_numeric(cbsa['FIPS County Code'], errors='coerce').astype('Int64')
    cbsa = standardize_fips(cbsa, state_col='FIPS State Code', county_col='FIPS County Code', fips_col='FIPS')
    cbsa.rename(columns={'CBSA Code': 'MSA_Code'}, inplace=True)
    
    if not rpp_metro.empty:
        rpp_metro['MSA_Code'] = rpp_metro['MSA_Code'].astype(str).str.zfill(5)
        cbsa['MSA_Code'] = cbsa['MSA_Code'].astype(str).str.zfill(5)
        rpp_metro = rpp_metro.merge(cbsa[['MSA_Code', 'FIPS']], on='MSA_Code', how='left')
        rpp_metro = rpp_metro[['FIPS', 'Year', 'RPP_Metro']].dropna(subset=['FIPS'])
        save_point(rpp_metro, 'BEA_RPP_Metro.csv', f"{len(rpp_metro):,} county-year RPP records")
    
    print("\nBEA data acquisition complete\n")

if __name__ == '__main__':
    run()
