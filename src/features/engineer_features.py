"""Feature engineering and panel construction."""
import pandas as pd
from config.settings import PROCESSED_DATA_DIR
from src.utils.helpers import save_point

def create_census_features(census):
    """Create derived features from Census data."""
    print("Creating Census-derived features...")
    
    POP_TOT = census['total_population']
    MAR_TOT = census['marital_total']
    
    # Home Tenure
    census['pct_owner_occupied'] = (census['owner_occupied'] / census['housing_total'] * 100)
    # Marital status
    census['pct_never_married_male'] = (census['never_married_male'] / MAR_TOT) * 100
    census['pct_now_married_male'] = (census['now_married_male'] / MAR_TOT) * 100
    census['pct_divorced_male'] = (census['divorced_male'] / MAR_TOT) * 100
    census['pct_never_married_female'] = (census['never_married_female'] / MAR_TOT) * 100
    census['pct_now_married_female'] = (census['now_married_female'] / MAR_TOT) * 100
    census['pct_divorced_female'] = (census['divorced_female'] / MAR_TOT) * 100
    census['pct_widowed_female'] = (census['widowed_female'] / MAR_TOT) * 100
    # Race and Ethnicity
    census['pct_white'] = (census['white'] / POP_TOT) * 100
    census['pct_black'] = (census['black'] / POP_TOT) * 100
    census['pct_native'] = (census['native'] / POP_TOT) * 100
    census['pct_asian'] = (census['asian'] / POP_TOT) * 100
    census['pct_pacific_islander'] = (census['pacific_islander'] / POP_TOT) * 100
    census['pct_other_race'] = (census['other_race'] / POP_TOT) * 100
    census['pct_hispanic'] = (census['hispanic'] / POP_TOT) * 100
    # Education
    census['pct_college_degree'] = (
        census['male_associates'] + census['female_associates'] +
        census['male_bachelors'] + census['female_bachelors'] +
        census['male_masters'] + census['female_masters'] +
        census['male_professional'] + census['female_professional'] +
        census['male_doctorate'] + census['female_doctorate']) / census['education_total_sex'] * 100
    census['pct_HSorCollege_NOdegree'] = (
        census['male_complete_hs'] + census['female_complete_hs'] +
        census['male_less1yr_college'] + census['female_less1yr_college'] +
        census['male_more1yr_college'] + census['female_more1yr_college']) / census['education_total_sex'] * 100
    # Occupation
    census['pct_white_collar'] = (
        census['Mgmt_Biz_Sci_Arts'] + census['Services'] +
        census['Sales_Admin']) / census['occupation_total'] * 100
    # Optimize percentage features
    for col in census.filter(regex=r'^pct_').columns:
        census[col] = census[col].round(2)
        census[col] = pd.to_numeric(census[col], errors='coerce')
    
    print(f"  Created {len(census.filter(regex=r'^pct_').columns)} percentage features")
    
    return census

def drop_raw_variables(census):
    """Drop raw variables after creating derived features."""
    drop_vars = [
        'family_households', 'education_total_sex', 'marital_total',
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
    
    census_reduced = census.drop(columns=drop_vars, errors='ignore')
    print(f"  Dropped {len(drop_vars)} raw variables")
    
    return census_reduced

def merge_panel():
    """Merge all datasets into analytical panel."""
    print("\nConstructing county-year analytical panel...")
    
    # Load cleaned data
    census = pd.read_csv(PROCESSED_DATA_DIR / 'Census_clean.csv')
    irs = pd.read_csv(PROCESSED_DATA_DIR / 'IRS_panel_clean.csv')
    bls = pd.read_csv(PROCESSED_DATA_DIR / 'BLS_import.csv')
    pci = pd.read_csv(PROCESSED_DATA_DIR / 'BEA_PCI_clean.csv')
    gdp = pd.read_csv(PROCESSED_DATA_DIR / 'BEA_GDP_clean.csv')
    rucc_2013 = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_RUCC_2013.csv')
    rucc_2023 = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_RUCC_2023_clean.csv')
    typology = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_Typology.csv')
    amenities = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_Amenities_clean.csv')
    incentives = pd.read_csv(PROCESSED_DATA_DIR / 'Incentives_clean.csv')
    rpp_metro = pd.read_csv(PROCESSED_DATA_DIR / 'BEA_RPP_Metro.csv')
    rpp_nonmetro = pd.read_csv(PROCESSED_DATA_DIR / 'BEA_RPP_NONmetro.csv')
    
    # Standardize FIPS across all dataframes
    for df in [census, irs, bls, pci, gdp, rucc_2013, rucc_2023, 
               typology, amenities, incentives, rpp_metro]:
        if 'FIPS' in df.columns:
            df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(int)
    
    # Start with Census as base
    panel = census.copy()
    print(f"  Starting: {len(panel):,} obs ({panel['FIPS'].nunique()} counties)")
    
    # Prepare Temporal assignments
    rucc_2013_panel = pd.concat([rucc_2013.assign(Year=y) for y in range(2011, 2020)], ignore_index=True)
    rucc_2023_panel = pd.concat([rucc_2023.assign(Year=y) for y in range(2020, 2022)], ignore_index=True)
    typology_panel = pd.concat([typology.assign(Year=y) for y in range(2011, 2022)], ignore_index=True)
    amenities_panel = pd.concat([amenities.assign(Year=y) for y in range(2011, 2022)], ignore_index=True)

    for df in [rucc_2013_panel, rucc_2023_panel, typology_panel, amenities_panel, 
               pci, gdp, rpp_metro, bls, irs, incentives]:
        df.drop_duplicates(['FIPS', 'Year'], inplace=True)
    
    # USDA
    panel = panel.merge(rucc_2013_panel[['FIPS', 'Year', 'RUCC_2013']], on=['FIPS', 'Year'], how='left')
    panel = panel.merge(rucc_2023_panel[['FIPS', 'Year', 'RUCC_2023']], on=['FIPS', 'Year'], how='left')
    panel['RUCC_code'] = panel['RUCC_2013'].combine_first(panel['RUCC_2023'])
    panel['RUCC_code'] = panel['RUCC_code'].astype('Int64')
    print(f"  + RUCC: {len(panel):,} obs")
   
    panel = panel.merge(typology_panel, on=['FIPS', 'Year'], how='left')
    typology_cols = [c for c in typology_panel.columns if c not in ['FIPS', 'Year']]
    panel[typology_cols] = panel[typology_cols].fillna(0)
    
    panel = panel.merge(amenities_panel, on=['FIPS', 'Year'], how='left')
    amenities_cols = [c for c in amenities_panel.columns if c not in ['FIPS', 'Year']]
    panel[amenities_cols] = panel[amenities_cols].fillna(0)
    print(f"  + USDA: {len(panel):,} obs")
    
    # RPP (metro and non-metro)
    panel['STATE'] = panel['FIPS'].str[:2]

    # Metro RPP
    rpp_metro = rpp_metro.drop_duplicates(['FIPS', 'Year'])
    panel = panel.merge(rpp_metro[['FIPS', 'Year', 'RPP_Metro']], on=['FIPS', 'Year'], how='left')

    # Non-Metro RPP (state-level)
    rpp_nonmetro['STATE'] = rpp_nonmetro['State_FIPS'].astype(str).str[:2]
    rpp_nonmetro = rpp_nonmetro[rpp_nonmetro['State_FIPS'].astype(str).str.endswith('999')].copy()
    rpp_nonmetro = rpp_nonmetro.drop_duplicates(['STATE', 'Year'])
    panel = panel.merge(rpp_nonmetro[['STATE', 'Year', 'RPP_NonMetro']], on=['STATE', 'Year'], how='left')

    # Combine Metro and Non-Metro RPP
    panel['RPP'] = panel['RPP_Metro'].fillna(panel['RPP_NonMetro'])
    panel = panel.drop(columns=['RPP_Metro', 'RPP_NonMetro', 'STATE'])

    # GDP
    panel = panel.merge(gdp[['FIPS', 'Year', 'BEA_GDP']], on=['FIPS', 'Year'], how='left')
    panel['BEA_GDP'] = panel['BEA_GDP'].fillna(0)
    
    # PCI
    panel = panel.merge(pci[['FIPS', 'Year', 'BEA_PCI']], on=['FIPS', 'Year'], how='left')
    panel['BEA_PCI'] = panel['BEA_PCI'].fillna(0)
    print(f"  + BEA: {len(panel):,} obs")
        
    # BLS
    panel = panel.merge(bls[['FIPS', 'Year', 'unemploy_rate']], on=['FIPS', 'Year'], how='left')
    panel['unemploy_rate'] = panel['unemploy_rate'].fillna(0)
    print(f"  + BLS: {len(panel):,} obs")
    
    # IRS Migration
    panel = panel.merge(irs, on=['FIPS', 'Year'], how='left')
    migration_cols = [c for c in irs.columns if c not in ['FIPS', 'Year']]
    panel[migration_cols] = panel[migration_cols].fillna(0)
    print(f"  + IRS: {len(panel):,} obs")
    
    # Incentives
    panel = panel.merge(incentives, on=['FIPS', 'Year'], how='left')
    incentive_cols = [c for c in incentives.columns if c not in ['FIPS', 'Year']]
    panel[incentive_cols] = panel[incentive_cols].fillna(0)
    print(f"  + Incentives: {len(panel):,} obs")
    
    # Drop unnecessary columns
    drop_cols = ['RUCC_2013', 'RUCC_2023', 'Industry_type', 
                 'move_in', 'move_out', 'agi_in', 'agi_out']
    panel = panel.drop(columns=drop_cols, errors='ignore')
    
    print(f"\nFinal panel: {len(panel):,} obs, {panel['FIPS'].nunique()} counties, {panel['Year'].nunique()} years")
    
    return panel

def run():
    """Main feature engineering function."""
    print("="*49)
    print("FEATURE ENGINEERING")
    print("="*49 + "\n")
    
    census = pd.read_csv(PROCESSED_DATA_DIR / 'Census_prep.csv')
    census = create_census_features(census)
    census_reduced = drop_raw_variables(census)
    save_point(census_reduced, 'Census_clean.csv')
    panel = merge_panel()
    save_point(panel, 'full_panel.csv', f"{len(panel):,} county-year observations")
    
    print("\nFeature engineering complete\n")

if __name__ == '__main__':
    run()
