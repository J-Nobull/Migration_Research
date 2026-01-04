"""IRS County-to-County Migration data processing."""
import pandas as pd
from config.settings import RAW_DATA_DIR, IRS_FILES
from src.utils.helpers import standardize_fips, define_cols, remap_fips_changes, filter_dataframe, save_point

def process_irs_migration(filename):
    """Process single IRS migration file."""
    df = pd.read_csv(RAW_DATA_DIR / filename, dtype=str, encoding='latin-1')
    
    # Extract year from filename: countyinflow1112.csv -> 2011
    year_suffix = filename[12:14]
    year = 2000 + int(year_suffix)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    rename_map = {'n2': 'movers', 'agi': 'movers_agi'}
    df.rename(columns=rename_map, inplace=True)
    
    # Create FIPS codes
    df = standardize_fips(df, state_col='y1_statefips', county_col='y1_countyfips', fips_col='origin_FIPS')
    df = standardize_fips(df, state_col='y2_statefips', county_col='y2_countyfips', fips_col='dest_FIPS')
    
    # Drop state totals and non-movers
    ST_TOT_dest = (df['dest_FIPS'].astype('Int64') % 1000 == 0)
    ST_TOT_origin = (df['origin_FIPS'].astype('Int64') % 1000 == 0)
    non_movers = (df['origin_FIPS'] == df['dest_FIPS'])
    drop_mask = (ST_TOT_dest | ST_TOT_origin | non_movers)
    df = df[~drop_mask].copy()
    
    # Exclude territories
    df = df[df['origin_FIPS'].astype('Int64') < 57000]
    df = df[df['dest_FIPS'].astype('Int64') < 57000]
    
    df['dest_FIPS'] = df['dest_FIPS'].astype(str)
    df['origin_FIPS'] = df['origin_FIPS'].astype(str)
    df['Year'] = year
    
    return df

def run():
    """Main IRS data processing function."""
    print("="*49)
    print("IRS MIGRATION DATA PROCESSING")
    print("="*49 + "\n")
    
    print("Processing 11 years of bilateral flows...")
    bilateral_flows = []
    
    for filename in IRS_FILES:
        df = process_irs_migration(filename)
        bilateral_flows.append(df)
        print(f"  ✓ {filename}: {len(df):,} flows (Year {df['Year'].iloc[0]})")
    
    bilateral = pd.concat(bilateral_flows, ignore_index=True)
    bilateral = define_cols(bilateral)
    bilateral = remap_fips_changes(bilateral, fips_cols=['dest_FIPS'])
    bilateral = remap_fips_changes(bilateral, fips_cols=['origin_FIPS'])
    bilateral = filter_dataframe(bilateral, 'Bilateral Flows')
    
    IRS_bilateral = bilateral[['origin_FIPS', 'Year', 'dest_FIPS', 'movers', 'movers_agi']]
    
    # Save bilateral flows for gravity model
    save_point(IRS_bilateral, 'IRS_gravity.csv', f"{len(IRS_bilateral):,} bilateral flows")
    
    # Create aggregated panel data
    print("\nCreating aggregated panel data...")
    
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
    
    IRS_migration = pd.merge(inflow, outflow, on=['FIPS', 'Year'], how='outer').fillna(0)
    IRS_migration['move_net'] = IRS_migration['move_in'] - IRS_migration['move_out']
    IRS_migration['agi_net'] = IRS_migration['agi_in'] - IRS_migration['agi_out']
    IRS_migration = remap_fips_changes(IRS_migration, fips_cols=['FIPS'])
    
    save_point(IRS_migration, 'IRS_panel.csv', f"{len(IRS_migration):,} panel-ready observations")
    
    print(f"\nIRS data processing complete")
    print(f"Panel rows: {len(IRS_migration):,}")
    print(f"Unique counties: {IRS_migration['FIPS'].nunique():,}\n")

if __name__ == '__main__':
    run()
