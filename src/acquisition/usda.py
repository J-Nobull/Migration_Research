"""USDA data loading from manual download files."""
import pandas as pd
from config.settings import RAW_DATA_DIR
from src.utils.helpers import standardize_fips, define_cols, remap_fips_changes, filter_dataframe, save_point

def run():
    """Main USDA data loading function."""
    print("="*49)
    print("USDA DATA LOADING")
    print("="*49 + "\n")
    
    # Rural-Urban Continuum Codes 2013
    print("Loading RUCC 2013...")
    rucc_2013 = pd.read_excel(RAW_DATA_DIR / 'ruralurbancodes2013.xls', dtype={'FIPS': str})
    rucc_2013 = standardize_fips(rucc_2013)
    rucc_2013 = rucc_2013[['FIPS', 'RUCC_2013']]
    rucc_2013 = remap_fips_changes(rucc_2013, fips_cols=['FIPS'])
    rucc_2013 = filter_dataframe(rucc_2013, 'RUCC 2013')
    save_point(rucc_2013, 'USDA_RUCC_2013.csv', f"{len(rucc_2013)} counties")
    
    # Rural-Urban Continuum Codes 2023
    print("\nLoading RUCC 2023...")
    rucc_2023_long = pd.read_csv(RAW_DATA_DIR / 'Ruralurbancontinuumcodes2023.csv', 
                                  dtype={'FIPS': str}, encoding='latin-1')
    rucc_2023 = rucc_2023_long[rucc_2023_long['Attribute'] == 'RUCC_2023'].copy()
    rucc_2023 = rucc_2023[['FIPS', 'Value']].rename(columns={'Value': 'RUCC_2023'})
    rucc_2023 = standardize_fips(rucc_2023)
    rucc_2023['RUCC_2023'] = pd.to_numeric(rucc_2023['RUCC_2023'], errors='coerce')
    rucc_2023 = remap_fips_changes(rucc_2023, fips_cols=['FIPS'])
    rucc_2023 = filter_dataframe(rucc_2023, 'RUCC 2023')
    save_point(rucc_2023, 'USDA_RUCC_2023.csv', f"{len(rucc_2023)} counties")
    
    # County Typology
    print("\nLoading County Typology 2015...")
    typology = pd.read_csv(RAW_DATA_DIR / 'erscountytypology2015edition.csv',
                          dtype={'FIPStxt': str}, encoding='latin-1')
    
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
    save_point(typology, 'USDA_Typology.csv', f"{len(typology)} counties")
    
    # Natural Amenities Scale
    print("\nLoading Natural Amenities...")
    amenities = pd.read_excel(RAW_DATA_DIR / 'natamenf_1_.xls',
                             dtype={'for measures': str}, header=104)
    amenities.rename(columns={'for measures': 'FIPS', 'Scale': 'Amenity_scale'}, inplace=True)
    amenities = amenities.drop_duplicates(subset='FIPS', keep='first')
    amenities = amenities[['FIPS', 'Amenity_scale']]
    amenities = standardize_fips(amenities)
    amenities = define_cols(amenities)
    amenities = remap_fips_changes(amenities)
    amenities = filter_dataframe(amenities, 'Natural Amenities')
    save_point(amenities, 'USDA_Amenities.csv', f"{len(amenities)} counties")
    
    print("\n✅ USDA data loading complete\n")

if __name__ == '__main__':
    run()
