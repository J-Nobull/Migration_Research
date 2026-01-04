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
    census['%owner_occupied'] = (census['owner_occupied'] / census['housing_total'] * 100)
    
    # Marital status
    census['%never_married_male'] = (census['never_married_male'] / MAR_TOT) * 100
    census['%now_married_male'] = (census['now_married_male'] / MAR_TOT) * 100
    census['%divorced_male'] = (census['divorced_male'] / MAR_TOT) * 100
    census['%never_married_female'] = (census['never_married_female'] / MAR_TOT) * 100
    census['%now_married_female'] = (census['now_married_female'] / MAR_TOT) * 100
    census['%divorced_female'] = (census['divorced_female'] / MAR_TOT) * 100
    census['%widowed_female'] = (census['widowed_female'] / MAR_TOT) * 100
    
    # Race and Ethnicity
    census['%white'] = (census['white'] / POP_TOT) * 100
    census['%black'] = (census['black'] / POP_TOT) * 100
    census['%native'] = (census['native'] / POP_TOT) * 100
    census['%asian'] = (census['asian'] / POP_TOT) * 100
    census['%pacific_islander'] = (census['pacific_islander'] / POP_TOT) * 100
    census['%other_race'] = (census['other_race'] / POP_TOT) * 100
    census['%hispanic'] = (census['hispanic'] / POP_TOT) * 100
    
    # Education
    census['%college_degree'] = (
        census['male_associates'] + census['female_associates'] +
        census['male_bachelors'] + census['female_bachelors'] +
        census['male_masters'] + census['female_masters'] +
        census['male_professional'] + census['female_professional'] +
        census['male_doctorate'] + census['female_doctorate']) / census['education_total_sex'] * 100
    
    census['%HSorCollege_NOdegree'] = (
        census['male_complete_hs'] + census['female_complete_hs'] +
        census['male_less1yr_college'] + census['female_less1yr_college'] +
        census['male_more1yr_college'] + census['female_more1yr_college']) / census['education_total_sex'] * 100
    
    # Occupation
    census['%white_collar'] = (
        census['Mgmt_Biz_Sci_Arts'] + census['Services'] +
        census['Sales_Admin']) / census['occupation_total'] * 100
    
    # Optimize percentage features
    for col in census.filter(regex=r'^%').columns:
        census[col] = census[col].round(2)
        census[col] = pd.to_numeric(census[col], errors='coerce')
    
    print(f"  Created {len(census.filter(regex=r'^%').columns)} percentage features")
    
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
    census = pd.read_csv(PROCESSED_DATA_DIR / 'Census_reduced.csv')
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
               typology, amenities, incentives, rpp_metro, rpp_nonmetro]:
        if 'FIPS' in df.columns:
            df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(int)
    
    # Start with Census as base
    panel = census.copy()
    print(f"  Starting: {len(panel):,} obs ({panel['FIPS'].nunique()} counties)")
    
    # RUCC temporal assignment
    rucc_2013_panel = pd.concat([rucc_2013.assign(Year=y) for y in range(2011, 2020)], ignore_index=True)
    rucc_2013_panel = rucc_2013_panel.drop_duplicates(['FIPS', 'Year'])
    
    rucc_2023_panel = pd.concat([rucc_2023.assign(Year=y) for y in range(2020, 2022)], ignore_index=True)
    rucc_2023_panel = rucc_2023_panel.drop_duplicates(['FIPS', 'Year'])
    
    panel = panel.merge(rucc_2013_panel[['FIPS', 'Year', 'RUCC_2013']], on=['FIPS', 'Year'], how='left')
    panel = panel.merge(rucc_2023_panel[['FIPS', 'Year', 'RUCC_2023']], on=['FIPS', 'Year'], how='left')
    panel['RUC_code'] = panel['RUCC_2013'].combine_first(panel['RUCC_2023'])
    panel['RUC_code'] = panel['RUC_code'].astype('Int64')
    print(f"  + RUCC: {len(panel):,} obs")
    
    # USDA typology and amenities
    typology_panel = pd.concat([typology.assign(Year=y) for y in range(2011, 2022)], ignore_index=True)
    typology_panel = typology_panel.drop_duplicates(['FIPS', 'Year'])
    
    amenities_panel = pd.concat([amenities.assign(Year=y) for y in range(2011, 2022)], ignore_index=True)
    amenities_panel = amenities_panel.drop_duplicates(['FIPS', 'Year'])
    
    panel = panel.merge(typology_panel, on=['FIPS', 'Year'], how='left')
    panel = panel.merge(amenities_panel, on=['FIPS', 'Year'], how='left')
    
    typology_cols = [c for c in typology_panel.columns if c not in ['FIPS', 'Year']]
    amenities_cols = [c for c in amenities_panel.columns if c not in ['FIPS', 'Year']]
    panel[typology_cols] = panel[typology_cols].fillna(0)
    panel[amenities_cols] = panel[amenities_cols].fillna(0)
    print(f"  + USDA: {len(panel):,} obs")
    
    # BEA - drop duplicates before merge
    pci = pci.drop_duplicates(['FIPS', 'Year'])
    gdp = gdp.drop_duplicates(['FIPS', 'Year'])
    
    panel = panel.merge(pci[['FIPS', 'Year', 'BEA_PCI']], on=['FIPS', 'Year'], how='left')
    panel = panel.merge(gdp[['FIPS', 'Year', 'BEA_GDP']], on=['FIPS', 'Year'], how='left')
    panel['BEA_PCI'] = panel['BEA_PCI'].fillna(0)
    panel['BEA_GDP'] = panel['BEA_GDP'].fillna(0)
    
    # RPP (metro and non-metro)
    panel['STATE'] = panel['FIPS'].str[:2]
    
    rpp_metro = rpp_metro.drop_duplicates(['FIPS', 'Year'])
    panel = panel.merge(rpp_metro[['FIPS', 'Year', 'RPP_Metro']], on=['FIPS', 'Year'], how='left')
    
    rpp_nonmetro['STATE'] = rpp_nonmetro['State_FIPS'].astype(str).str[:2]
    rpp_nonmetro = rpp_nonmetro[rpp_nonmetro['State_FIPS'].astype(str).str.endswith('999')].copy()
    rpp_nonmetro = rpp_nonmetro.drop_duplicates(['STATE', 'Year'])
    panel = panel.merge(rpp_nonmetro[['STATE', 'Year', 'RPP_NonMetro']], on=['STATE', 'Year'], how='left')
    
    panel['RPP'] = panel['RPP_Metro'].fillna(panel['RPP_NonMetro'])
    panel = panel.drop(columns=['RPP_Metro', 'RPP_NonMetro', 'STATE'])
    print(f"  + BEA: {len(panel):,} obs")
    
    # BLS
    bls = bls.drop_duplicates(['FIPS', 'Year'])
    panel = panel.merge(bls[['FIPS', 'Year', 'unemploy_rate']], on=['FIPS', 'Year'], how='left')
    panel['unemploy_rate'] = panel['unemploy_rate'].fillna(0)
    print(f"  + BLS: {len(panel):,} obs")
    
    # IRS Migration
    irs = irs.drop_duplicates(['FIPS', 'Year'])
    panel = panel.merge(irs, on=['FIPS', 'Year'], how='left')
    migration_cols = [c for c in irs.columns if c not in ['FIPS', 'Year']]
    panel[migration_cols] = panel[migration_cols].fillna(0)
    print(f"  + IRS: {len(panel):,} obs")
    
    # Incentives
    incentives = incentives.drop_duplicates(['FIPS', 'Year'])
    panel = panel.merge(incentives, on=['FIPS', 'Year'], how='left')
    incentive_cols = [c for c in incentives.columns if c not in ['FIPS', 'Year']]
    panel[incentive_cols] = panel[incentive_cols].fillna(0)
    print(f"  + Incentives: {len(panel):,} obs")
    
    # Drop temporary columns
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
    
    # Load Census data
    census = pd.read_csv(PROCESSED_DATA_DIR / 'Census_clean.csv')
    
    # Create features
    census = create_census_features(census)
    census = drop_raw_variables(census)
    
    # Save reduced Census
    save_point(census, 'Census_reduced.csv', 'Census with derived features')
    
    # Merge all datasets
    panel = merge_panel()
    
    # Save final panel
    save_point(panel, 'full_panel.csv', f"{len(panel):,} county-year observations")
    
    print("\nFeature engineering complete\n")

if __name__ == '__main__':
    run()
