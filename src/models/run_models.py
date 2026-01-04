"""Statistical models for migration analysis."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels import PooledOLS
from config.settings import PROCESSED_DATA_DIR, OUTPUTS_DIR
from src.utils.helpers import save_point

def load_panel():
    """Load full panel dataset."""
    return pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')

def model_1_gravity():
    """Model 1: Gravity Model - RUCC-based migration flows."""
    print("\n" + "="*49)
    print("MODEL 1: GRAVITY MODEL")
    print("="*49 + "\n")
    
    # Load bilateral flows
    bilateral = pd.read_csv(PROCESSED_DATA_DIR / 'IRS_gravity.csv')
    
    # Load RUCC data
    rucc_2013 = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_RUCC_2013.csv')
    rucc_2023 = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_RUCC_2023_clean.csv')
    
    # Assign RUCC to years
    rucc_2013_panel = pd.concat([rucc_2013.assign(Year=y) for y in range(2011, 2020)], 
                                 ignore_index=True).rename(columns={'RUCC_2013': 'RUC_code'})
    rucc_2023_panel = pd.concat([rucc_2023.assign(Year=y) for y in range(2020, 2022)], 
                                 ignore_index=True).rename(columns={'RUCC_2023': 'RUC_code'})
    rucc_all = pd.concat([rucc_2013_panel, rucc_2023_panel], ignore_index=True)
    rucc_all['RUC_code'] = pd.to_numeric(rucc_all['RUC_code'], errors='coerce').astype('Int64')
    rucc_all['FIPS'] = rucc_all['FIPS'].astype(str)
    
    # Merge RUCC codes - standardize FIPS to string
    bilateral['dest_FIPS'] = bilateral['dest_FIPS'].astype(str).str.zfill(5)
    bilateral['origin_FIPS'] = bilateral['origin_FIPS'].astype(str).str.zfill(5)
    
    bilateral = bilateral.merge(
        rucc_all.rename(columns={'FIPS': 'dest_FIPS', 'RUC_code': 'RUC_code_dest'}),
        on=['dest_FIPS', 'Year'], how='left')
    bilateral = bilateral.merge(
        rucc_all.rename(columns={'FIPS': 'origin_FIPS', 'RUC_code': 'RUC_code_origin'}),
        on=['origin_FIPS', 'Year'], how='left')
    
    bilateral = bilateral.drop(columns=['movers_agi']).dropna(subset=['RUC_code_origin', 'RUC_code_dest'])
    
    # Create variables
    bilateral['log_movers'] = np.log(bilateral['movers'].astype(float) + 1)
    
    # Estimate
    X_grav = pd.get_dummies(bilateral[['RUC_code_origin', 'RUC_code_dest']], drop_first=True)
    X_grav = sm.add_constant(X_grav).astype(float)
    y_grav = bilateral['log_movers'].astype(float)
    
    model1 = sm.OLS(y_grav, X_grav).fit()
    
    # Save results
    results_df = pd.DataFrame({
        'Variable': model1.params.index,
        'Coefficient': model1.params.values,
        'Std_Error': model1.bse.values,
        'P_value': model1.pvalues.values})
    results_df['R2'] = model1.rsquared
    
    save_point(results_df, 'MODEL-1_gravity_results.csv', 'Gravity Model')
    
    print(f"R²: {model1.rsquared:.4f}")
    print(f"N: {model1.nobs}")
    print(" Model 1 complete\n")
    
    return model1

def model_2_panel_fe():
    """Model 2: Panel Fixed Effects."""
    print("\n" + "="*49)
    print("MODEL 2: PANEL FIXED EFFECTS")
    print("="*49 + "\n")
    
    panel = load_panel()
    model_2 = panel.copy().set_index(['FIPS', 'Year'])
    
    # Define variables
    panel_X_vars = [
        'total_population', 'median_age', 'under_18_in_hh',
        'median_hh_income', 'median_home_value', 'median_property_taxes',
        'commute_less_5min', 'commute_5_9min', 'commute_10_14min',
        'commute_15_19min', 'commute_20_24min', 'commute_25_29min',
        'commute_30_34min', 'commute_35_39min', 'commute_40_44min',
        'commute_45_59min', 'commute_60_89min', 'commute_90_plus_min',
        '%white_collar', '%owner_occupied',
        '%white', '%black', '%native', '%asian', '%pacific_islander',
        '%other_race', '%hispanic',
        '%never_married_male', '%never_married_female',
        '%now_married_male', '%now_married_female',
        '%divorced_male', '%divorced_female',
        'RUC_code', 'Farming', 'Mining', 'Mfging', 'Govt', 'Rec',
        'Low_Ed_cnty', 'Low_employ_cnty',
        'Retire_dest_cnty', 'Persistent_Pov_cnty', 'Pers_chld_pov_cnty',
        'Amenity_scale', 'BEA_PCI', 'BEA_GDP', 'RPP', 'unemploy_rate']
    
    # Time-varying only
    X_time_varying = model_2[[
        'total_population', 'median_hh_income', 'median_home_value',
        'median_property_taxes', 'BEA_PCI', 'BEA_GDP', 'RPP', 'unemploy_rate']]
    
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
    comparison = pd.DataFrame({
        'Model': ['Pooled_OLS', 'County_FE', 'TwoWay_FE'],
        'R2_Overall': [pooled_model.rsquared, fe_model.rsquared, twoway_model.rsquared],
        'R2_Within': [pooled_model.rsquared, fe_model.rsquared_within, twoway_model.rsquared_within],
        'N_Obs': [pooled_model.nobs, fe_model.nobs, twoway_model.nobs]})
    
    save_point(comparison, 'MODEL-2_panel_fe_comparison.csv', 'Panel FE specifications')
    
    print(comparison)
    print("\n Model 2 complete\n")
    
    return pooled_model, fe_model, twoway_model

def model_3_did():
    """Model 3: Difference-in-Differences."""
    print("\n" + "="*49)
    print("MODEL 3: DIFFERENCE-IN-DIFFERENCES")
    print("="*49 + "\n")
    
    panel = load_panel()
    model_3 = panel.copy()
    
    model_3['PULL'] = model_3['has_incentive']
    model_3['POST'] = (model_3['Year'] >= model_3['Incentive_CAT']).fillna(0).astype('Int64')
    model_3['pull_x_post'] = model_3['PULL'] * model_3['POST']
    
    panel_X_vars = [
        'total_population', 'median_hh_income', 'median_home_value',
        'median_property_taxes', 'BEA_PCI', 'BEA_GDP', 'RPP', 'unemploy_rate']
    
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
    
    # Save results
    results = pd.DataFrame({
        'Specification': ['Simple', 'Controls'],
        'Treatment_Effect': [
            model3a.params.get('pull_x_post', np.nan),
            model3b.params.get('pull_x_post', np.nan)],
        'Std_Error': [
            model3a.std_errors.get('pull_x_post', np.nan),
            model3b.std_errors.get('pull_x_post', np.nan)],
        'P_value': [
            model3a.pvalues.get('pull_x_post', np.nan),
            model3b.pvalues.get('pull_x_post', np.nan)],
        'R2_Within': [model3a.rsquared, model3b.rsquared]})
    
    save_point(results, 'MODEL-3_did_comparison.csv', 'DiD specifications')
    
    print(results)
    print("\n Model 3 complete\n")
    
    return model3a, model3b

def model_4_dynamic():
    """Model 4: Dynamic Panel."""
    print("\n" + "="*49)
    print("MODEL 4: DYNAMIC PANEL")
    print("="*49 + "\n")
    
    panel = load_panel()
    dynamic_panel = panel.copy().sort_values(['FIPS', 'Year'])
    dynamic_panel['move_net_lag1'] = dynamic_panel.groupby('FIPS')['move_net'].shift(1)
    
    panel_X_vars = [
        'total_population', 'median_hh_income', 'median_home_value',
        'median_property_taxes', 'BEA_PCI', 'BEA_GDP', 'RPP', 'unemploy_rate']
    
    dynamic_data = dynamic_panel.dropna(
        subset=['move_net', 'move_net_lag1'] + panel_X_vars).set_index(['FIPS', 'Year'])
    
    y_dynamic = dynamic_data['move_net']
    X_dynamic = dynamic_data[['move_net_lag1'] + panel_X_vars]
    
    model4 = PanelOLS(y_dynamic, X_dynamic, entity_effects=True,
                      drop_absorbed=True, time_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    
    persistence = model4.params['move_net_lag1']
    lr_multiplier = 1 / (1 - persistence) if abs(persistence) < 1 else np.inf
    
    # Save results
    results = pd.DataFrame({
        'Variable': model4.params.index,
        'Coefficient': model4.params.values,
        'Std_Error': model4.std_errors.values,
        'P_value': model4.pvalues.values})
    results['R2_Within'] = model4.rsquared
    results['Persistence'] = persistence
    results['LR_Multiplier'] = lr_multiplier
    
    save_point(results, 'MODEL-4_dynamic_results.csv', 'Dynamic panel')
    
    print(f"Persistence (ρ): {persistence:.4f}")
    print(f"Long-run multiplier: {lr_multiplier:.4f}")
    print(f"R²: {model4.rsquared:.4f}")
    print("\n Model 4 complete\n")
    
    return model4

def run():
    """Run all statistical models."""
    print("="*49)
    print("STATISTICAL MODELING")
    print("="*49)
    
    model1 = model_1_gravity()
    pooled, fe, twoway = model_2_panel_fe()
    model3a, model3b = model_3_did()
    model4 = model_4_dynamic()
    
    print("\n" + "="*49)
    print("ALL MODELS COMPLETE")
    print("="*49 + "\n")

if __name__ == '__main__':
    run()
