"""Statistical models for migration analysis."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels import PooledOLS
from config.settings import PROCESSED_DATA_DIR, OUTPUTS_DIR
from src.analysis.create_outputs import save_academic_results

def load_panel():
    """Load full panel dataset."""
    panel = pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')
    panel['FIPS'] = panel['FIPS'].astype(str).str.zfill(5)
    return panel

def load_validated_features():
    """Load features validated in Step 4 (Feature Selection)."""
    print("\n" + "= "*25)
    print("LOADING VALIDATED FEATURES FROM STEP 4")
    print("= "*25)
    
    selected = pd.read_csv(OUTPUTS_DIR / 'tables' / 'selected_features.csv')
    time_varying = pd.read_csv(OUTPUTS_DIR / 'tables' / 'time_varying_validation.csv')
    
    recommended_all = selected[selected['Both'] == True]['Feature'].tolist()
    recommended_tv = time_varying[
        (time_varying['Significant'] == True) & 
        (time_varying['R2_Within'] > 0.01)]['Feature'].tolist()
    
    print(f"\nRecommended (Stepwise + LASSO): {len(recommended_all)} features")
    print(f"Time-varying (for FE models): {len(recommended_tv)} features")
    
    return recommended_all, recommended_tv

def model_1_gravity():
    """Model 1: Gravity Model with geographic distance."""
    print("\n" + "-"*49)
    print("MODEL 1: GRAVITY MODEL")
    print("-"*49 + "\n")
    
    panel = load_panel()
    bilateral = pd.read_csv(PROCESSED_DATA_DIR / 'IRS_gravity.csv')
    centroids = pd.read_csv(PROCESSED_DATA_DIR / 'County_centroids.csv')
    
    # Standardize FIPS
    panel['FIPS'] = panel['FIPS'].astype(str).str.zfill(5)
    bilateral['origin_FIPS'] = bilateral['origin_FIPS'].astype(str).str.zfill(5)
    bilateral['dest_FIPS'] = bilateral['dest_FIPS'].astype(str).str.zfill(5)
    centroids['FIPS'] = centroids['FIPS'].astype(str).str.zfill(5)
    
    # Merge centroids for origin and destination
    bilateral = bilateral.merge(
        centroids.rename(columns={'FIPS': 'origin_FIPS', 'lat': 'origin_lat', 'lon': 'origin_lon'}),
        on='origin_FIPS', how='left')
    bilateral = bilateral.merge(
        centroids.rename(columns={'FIPS': 'dest_FIPS', 'lat': 'dest_lat', 'lon': 'dest_lon'}),
        on='dest_FIPS', how='left')
    
    # Get economic data from panel
    M1_vars = ['FIPS', 'Year', 'total_population']
    M1_data = panel[M1_vars].drop_duplicates(subset=['FIPS', 'Year'])
    
    # Merge origin economics
    bilateral = bilateral.merge(
        M1_data.rename(columns={
            'FIPS': 'origin_FIPS', 'total_population': 'origin_pop'}),
        on=['origin_FIPS', 'Year'], how='left')
    
    # Merge destination economics
    bilateral = bilateral.merge(
        M1_data.rename(columns={
            'FIPS': 'dest_FIPS', 'total_population': 'dest_pop'}),
        on=['dest_FIPS', 'Year'], how='left')
    
    # Calculate distance (Haversine formula - great circle distance)
    def haversine_vec(lat1, lon1, lat2, lon2):
        """Vectorized Haversine formula for great-circle distance in miles."""
        R = 3959  # Earth radius in miles
        lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    bilateral['distance'] = haversine_vec(
        bilateral['origin_lat'], bilateral['origin_lon'],
        bilateral['dest_lat'], bilateral['dest_lon'])
    
    # Create gravity variables
    bilateral['log_origin_pop'] = np.log(bilateral['origin_pop'] + 1)
    bilateral['log_dest_pop'] = np.log(bilateral['dest_pop'] + 1)
    bilateral['log_distance'] = np.log(bilateral['distance'] + 1)
    bilateral['log_movers'] = np.log(bilateral['movers'] + 1)
    bilateral['same_state'] = (bilateral['origin_FIPS'].str[:2] == 
                               bilateral['dest_FIPS'].str[:2]).astype(int)
    gravity_vars = ['log_origin_pop', 'log_dest_pop', 'log_distance', 'same_state']
    
    # Diagnostics
    print(f"Total bilateral flows: {len(bilateral):,}")
    print(f"Missing centroids: {bilateral['distance'].isna().sum():,}")
    print(f"Distance range: {bilateral['distance'].min():.1f} - {bilateral['distance'].max():.1f} miles")
    print(f"Median distance: {bilateral['distance'].median():.1f} miles")
    print(f"Same-state flows: {bilateral['same_state'].sum():,} ({bilateral['same_state'].mean()*100:.1f}%)\n")
    
    X = bilateral[gravity_vars].copy()
    X = sm.add_constant(X)
    y = bilateral['log_movers']
    
    data = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:].astype(float)
    
    model1 = sm.OLS(y_clean, X_clean).fit()
    
    # Save results
    model1_results = pd.DataFrame({
        'Statistic': ['R2', 'N_Obs', 'N_Features'],
        'Value': [model1.rsquared, model1.nobs, len(model1.params)]})
    model1_results.to_csv(OUTPUTS_DIR / 'tables' / 'MODEL-1_gravity_results.csv', index=False)
    
    save_academic_results(
        model=model1,
        model_name='MODEL-1_gravity',
        description='Gravity model: Population + Distance + Border effects',
        output_dir=OUTPUTS_DIR / 'tables')
    
    print(f"R² = {model1.rsquared:.4f}")
    print(f"N = {int(model1.nobs):,}")
    print(f"N_Features = {len(model1.params):,}")
    print(f"log_distance coef = {model1.params['log_distance']:.4f}")
    print(f"same_state coef = {model1.params['same_state']:.4f}")
    print("\nModel 1 complete\n")
    
    return model1

def model_2_panel_fe(recommended_all, recommended_tv):
    """Model 2: Panel Fixed Effects with validated features."""
    print("\n" + "-"*49)
    print("MODEL 2: PANEL FIXED EFFECTS")
    print("-"*49 + "\n")
    
    panel = load_panel()
    panel_indexed = panel.set_index(['FIPS', 'Year'])
    
    available_all = [f for f in recommended_all if f in panel_indexed.columns]
    available_tv = [f for f in recommended_tv if f in panel_indexed.columns]
    available_tv_w_commute = available_tv.copy()
    available_tv_no_commute = [f for f in available_tv if not f.startswith('commute_')]
    
    print(f"\nFeature sets from Step 4:")
    print(f"  Pooled OLS (all features): {len(available_all)} features")
    print(f"  Time-varying with commute: {len(available_tv_w_commute)} features")
    print(f"  Time-varying no commute: {len(available_tv_no_commute)} features")
    print(f"\nTime-varying features:")
    for feat in available_tv_no_commute:
        print(f"    - {feat}")
    
    y = panel_indexed['move_net']
    X_all = panel_indexed[available_all]
    X_tv_w = panel_indexed[available_tv_w_commute]
    X_tv_no = panel_indexed[available_tv_no_commute]
    
    # Estimate models
    M2_pooled = PooledOLS(y, X_all).fit(cov_type='clustered', cluster_entity=True)
    M2_fe_w = PanelOLS(y, X_tv_w, entity_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    M2_twoway_w = PanelOLS(y, X_tv_w, entity_effects=True, time_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    M2_fe_no = PanelOLS(y, X_tv_no, entity_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    M2_twoway_no = PanelOLS(y, X_tv_no, entity_effects=True, time_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    
    # Comparison table
    comparison = pd.DataFrame({
        'Model': ['Pooled_OLS', 'County_FE_w_commute', 'TwoWay_FE_w_commute', 
                  'County_FE_no_commute', 'TwoWay_FE_no_commute'],
        'N_Features': [len(available_all), 
                      len(available_tv_w_commute), len(available_tv_w_commute),
                      len(available_tv_no_commute), len(available_tv_no_commute)],
        'R2_Overall': [M2_pooled.rsquared, 
                      M2_fe_w.rsquared, M2_twoway_w.rsquared,
                      M2_fe_no.rsquared, M2_twoway_no.rsquared],
        'R2_Within': [M2_pooled.rsquared,
                     M2_fe_w.rsquared_within, M2_twoway_w.rsquared_within,
                     M2_fe_no.rsquared_within, M2_twoway_no.rsquared_within],
        'N_Obs': [M2_pooled.nobs, 
                 M2_fe_w.nobs, M2_twoway_w.nobs,
                 M2_fe_no.nobs, M2_twoway_no.nobs]})
    
    comparison.to_csv(OUTPUTS_DIR / 'tables' / 'MODEL-2_feature_comparison.csv', index=False)
    print("\nModel 2 Results:")
    print(comparison.to_string(index=False))
    
    # Save final feature set
    X_final = available_tv_no_commute
    pd.DataFrame({'Feature': X_final}).to_csv(OUTPUTS_DIR / 'tables' / 'X_final.csv', index=False)
    
    print(f"\nSelected: County FE with {len(X_final)} features (no commute)")
    print(f"R² Within: {M2_fe_no.rsquared_within:.4f}")
    
    save_academic_results(
        model=M2_fe_no,
        model_name='MODEL-2_county_fe_validated',
        description='County fixed effects with {len(X_final)} validated features',
        output_dir=OUTPUTS_DIR / 'tables')
    
    print("\n Model 2 complete\n")
    return M2_pooled, M2_fe_no, M2_twoway_no, X_final

def model_3_did(X_final):
    """Model 3: Difference-in-Differences."""
    print("\n" + "-"*49)
    print("MODEL 3: DIFFERENCE-IN-DIFFERENCES")
    print("-"*49 + "\n")
    
    panel = load_panel()
    panel_indexed = panel.set_index(['FIPS', 'Year'])
    
    available_features = [f for f in X_final if f in panel_indexed.columns]
    
    # Create program start year by finding first year has_incentive = 1
    panel['Program_Start_Year'] = panel[panel['has_incentive'] == 1].groupby('FIPS')['Year'].transform('min')
    panel['Program_Start_Year'] = panel.groupby('FIPS')['Program_Start_Year'].ffill().fillna(9999)

    # Create treatment indicators
    panel['POST'] = (panel['Year'] >= panel['Program_Start_Year']).astype(int)
    panel['pull_x_post'] = panel['has_incentive'] * panel['POST']

    # Categories for heterogeneous effects
    panel['CAT1_treat'] = ((panel['Incentive_CAT'] == 1) * panel['POST']).astype(int)
    panel['CAT2_treat'] = ((panel['Incentive_CAT'] == 2) * panel['POST']).astype(int)
    panel['CAT3_treat'] = ((panel['Incentive_CAT'] == 3) * panel['POST']).astype(int)
    
    panel_indexed = panel.set_index(['FIPS', 'Year'])
    y = panel_indexed['move_net']
    
    # Model 3a: Simple DiD
    X3a = panel_indexed[['pull_x_post']]
    model3a = PanelOLS(y, X3a, entity_effects=True, time_effects=True, 
                       drop_absorbed=True, check_rank=False).fit(
                           cov_type='clustered', cluster_entity=True)
    
    # Model 3b: DiD with controls
    X3b = panel_indexed[['pull_x_post'] + available_features]
    model3b = PanelOLS(y, X3b, entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False).fit(
                           cov_type='clustered', cluster_entity=True)
    
    # Model 3c: Heterogeneous effects by category
    X3c = panel_indexed[['CAT1_treat', 'CAT2_treat', 'CAT3_treat'] + available_features]
    model3c = PanelOLS(y, X3c, entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False).fit(
                           cov_type='clustered', cluster_entity=True)
    
    # Results table
    results = pd.DataFrame({
        'Specification': ['Simple', 'Controls', 'CAT1_Low', 'CAT2_Med', 'CAT3_High'],
        'N_Obs': [model3a.nobs, model3b.nobs] + [model3c.nobs] * 3,
        'N_Features': [0] + [len(available_features)] * 4,
        'Treatment_Effect': [
            model3a.params.get('pull_x_post', np.nan),
            model3b.params.get('pull_x_post', np.nan),
            model3c.params.get('CAT1_treat', np.nan),
            model3c.params.get('CAT2_treat', np.nan),
            model3c.params.get('CAT3_treat', np.nan)],
        'Std_Error': [
            model3a.std_errors.get('pull_x_post', np.nan),
            model3b.std_errors.get('pull_x_post', np.nan),
            model3c.std_errors.get('CAT1_treat', np.nan),
            model3c.std_errors.get('CAT2_treat', np.nan),
            model3c.std_errors.get('CAT3_treat', np.nan)],
        'P_value': [
            model3a.pvalues.get('pull_x_post', np.nan),
            model3b.pvalues.get('pull_x_post', np.nan),
            model3c.pvalues.get('CAT1_treat', np.nan),
            model3c.pvalues.get('CAT2_treat', np.nan),
            model3c.pvalues.get('CAT3_treat', np.nan)],
        'R2_Within': [model3a.rsquared, model3b.rsquared] + [model3c.rsquared] * 3})
    
    results.to_csv(OUTPUTS_DIR / 'tables' / 'MODEL-3_did_comparison.csv', index=False)
    print("\nDiD Results:")
    print(results.to_string(index=False))
    
    save_academic_results(
        model=model3b,
        model_name='MODEL-3_did_controls',
        description='Difference-in-differences with control variables',
        output_dir=OUTPUTS_DIR / 'tables')
    
    print("\n Model 3 complete\n")
    return model3a, model3b, model3c

def model_4_dynamic(X_final):
    """Model 4: Dynamic Panel Model."""
    print("\n" + "-"*49)
    print("MODEL 4: DYNAMIC PANEL")
    print("-"*49 + "\n")
    
    panel = load_panel()
    
    # Create lagged DV
    panel = panel.sort_values(['FIPS', 'Year'])
    panel['move_net_lag1'] = panel.groupby('FIPS')['move_net'].shift(1)
    
    panel_indexed = panel.set_index(['FIPS', 'Year'])
    available_features = [f for f in X_final if f in panel_indexed.columns]
    
    y = panel_indexed['move_net']
    X = panel_indexed[['move_net_lag1'] + available_features]
    
    model4 = PanelOLS(y, X, entity_effects=True, time_effects=True,
                      drop_absorbed=True, check_rank=False).fit(
                          cov_type='clustered', cluster_entity=True)
    
    persistence = model4.params['move_net_lag1']
    lr_multiplier = 1 / (1 - persistence) if abs(persistence) < 1 else np.inf
    
    results = pd.DataFrame({
        'Variable': model4.params.index,
        'Coefficient': model4.params.values,
        'Std_Error': model4.std_errors.values,
        'P_value': model4.pvalues.values})
    results['R2_Within'] = model4.rsquared
    results['Persistence'] = persistence
    results['LR_Multiplier'] = lr_multiplier
    results['N_Obs'] = model4.nobs
    results['N_Features'] = len(available_features)
    
    results.to_csv(OUTPUTS_DIR / 'tables' / 'MODEL-4_dynamic_results.csv', index=False)
    
    print(f"\nDynamic Panel Results:")
    print(f"Persistence (ρ): {persistence:.4f}")
    print(f"Long-run multiplier: {lr_multiplier:.4f}")
    print(f"R²: {model4.rsquared:.4f}")
    
    save_academic_results(
        model=model4,
        model_name='MODEL-4_dynamic_panel',
        description='Dynamic panel with lagged dependent variable',
        output_dir=OUTPUTS_DIR / 'tables')
    
    print("\n Model 4 complete\n")
    return model4

def run():
    """Run all statistical models."""
    print("-"*49)
    print("STATISTICAL MODELS")
    print("-"*49 + "\n")
    
    (OUTPUTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
    
    recommended_all, recommended_tv = load_validated_features()
    
    model1 = model_1_gravity()
    pooled, fe, twoway, X_final = model_2_panel_fe(recommended_all, recommended_tv)
    model3a, model3b, model3c = model_3_did(X_final)
    model4 = model_4_dynamic(X_final)
    
    print("\n" + "="*49)
    print("ALL MODELS COMPLETE")
    print("="*49)
    print("\nFiles saved in:", OUTPUTS_DIR / 'tables')
    print("  - MODEL-1_gravity_coefficients.csv")
    print("  - MODEL-2_fixed_coefficients.csv")
    print("  - MODEL-3_did_controls_coefficients.csv")
    print("  - MODEL-4_dynamic_panel_coefficients.csv")
    print("\nStep 5: Models complete\n")

if __name__ == '__main__':
    run()
    