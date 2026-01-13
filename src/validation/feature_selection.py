"""Feature selection validation: VIF → Stepwise → LASSO → Time-Varying."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS
from config.settings import PROCESSED_DATA_DIR, OUTPUTS_DIR

def calculate_vif():
    """Step 1: Calculate VIF to check multicollinearity."""
    print("\n" + "="*49)
    print("STEP 4.1: VIF ANALYSIS")
    print("="*49)
    
    panel = pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')
    
    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    exclude = ['FIPS', 'Year', 
               'move_net', 'move_in', 'move_out', 
               'agi_in', 'agi_out']
    features = [c for c in numeric_cols if c not in exclude]
    
    print(f"Analyzing {len(features)} features for multicollinearity...")
    
    X = panel[features].dropna()
    X_const = sm.add_constant(X).astype(float)
    
    vif_data = pd.DataFrame({
        'Feature': X_const.columns,
        'VIF': [variance_inflation_factor(X_const.values, i) 
                for i in range(X_const.shape[1])]})
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    vif_data['High_VIF'] = vif_data['VIF'] > 10
    vif_data['Low_VIF'] = vif_data['VIF'] < 11
    
    print(f"\nFeatures with VIF > 10: {vif_data['High_VIF'].sum()}")
    print(f"Features with VIF < 11: {vif_data['Low_VIF'].sum()}")
    
    output_path = OUTPUTS_DIR / 'tables'
    output_path.mkdir(parents=True, exist_ok=True)
    vif_data.to_csv(output_path / 'vif_analysis.csv', index=False)
    
    print("\nVIF analysis complete")
    return vif_data

def load_panel_for_selection():
    """Load and prepare panel data for feature selection."""
    panel = pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')
    
    candidate_features = [
        'total_population', 'median_age', 'under_18_in_hh',
        'median_hh_income', 'median_home_value', 'median_property_taxes',
        'commute_less_5min', 'commute_5_9min', 'commute_10_14min',
        'commute_15_19min', 'commute_20_24min', 'commute_25_29min',
        'commute_30_34min', 'commute_35_39min', 'commute_40_44min',
        'commute_45_59min', 'commute_60_89min', 'commute_90_plus_min',
        'pct_white_collar', 'pct_owner_occupied',
        'pct_white', 'pct_black', 'pct_native', 'pct_asian', 'pct_pacific_islander',
        'pct_other_race', 'pct_hispanic',
        'pct_never_married_male', 'pct_never_married_female',
        'pct_now_married_male', 'pct_now_married_female',
        'pct_divorced_male', 'pct_divorced_female',
        'RUCC_code', 'Farming', 'Mining', 'Mfging', 'Govt', 'Rec',
        'Low_Ed_cnty', 'Low_employ_cnty',
        'Retire_dest_cnty', 'Persistent_Pov_cnty', 'Pers_chld_pov_cnty',
        'Amenity_scale', 'BEA_PCI', 'BEA_GDP', 'RPP', 'unemploy_rate']
    
    available_features = [f for f in candidate_features if f in panel.columns]
    panel_clean = panel.dropna(subset=['move_net'] + available_features)
    
    X = panel_clean[available_features]
    y = panel_clean['move_net']
    return X, y, available_features

def _forward_step(X, y, included, excluded, threshold_in, verbose):
    """Forward selection step."""
    new_pval = pd.Series(index=excluded, dtype=float)
    for col in excluded:
        model = sm.OLS(y, sm.add_constant(X[included + [col]])).fit()
        new_pval[col] = model.pvalues[col]
    best_pval = new_pval.min()
    if best_pval < threshold_in:
        best_feature = new_pval.idxmin()
        if verbose:
            print(f"Add:    {best_feature:30s} (p={best_pval:.4f})")
        return best_feature, True
    return None, False

def _backward_step(X, y, included, threshold_out, verbose):
    """Backward elimination step."""
    if len(included) == 0:
        return None, False
    model = sm.OLS(y, sm.add_constant(X[included])).fit()
    pvalues = model.pvalues.iloc[1:]
    worst_pval = pvalues.max()
    if worst_pval > threshold_out:
        worst_feature = pvalues.idxmax()
        if verbose:
            print(f"Remove: {worst_feature:30s} (p={worst_pval:.4f})")
        return worst_feature, True
    return None, False

def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.10, verbose=True):
    """Step 2: Perform stepwise regression."""
    if verbose:
        print("\n" + "="*49)
        print("STEP 4.2: STEPWISE REGRESSION")
        print("="*49)
    
    included = []
    
    while True:
        excluded = list(set(X.columns) - set(included))
        add_feat, add_changed = _forward_step(X, y, included, excluded, threshold_in, verbose)
        if add_feat:
            included.append(add_feat)
        remove_feat, remove_changed = _backward_step(X, y, included, threshold_out, verbose)
        if remove_feat:
            included.remove(remove_feat)
        if not (add_changed or remove_changed):
            break
    
    final_model = sm.OLS(y, sm.add_constant(X[included])).fit()
    
    if verbose:
        print(f"\nSelected {len(included)} features")
        print(f"R²: {final_model.rsquared:.4f}")
    return included, final_model

def lasso_selection(X, y, verbose=True):
    """Step 3: Perform LASSO feature selection."""
    if verbose:
        print("\n" + "="*49)
        print("STEP 4.3: LASSO REGRESSION")
        print("="*49)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
    lasso.fit(X_scaled, y)
    
    selected_mask = lasso.coef_ != 0
    selected_features = X.columns[selected_mask].tolist()
    
    y_pred = lasso.predict(X_scaled)
    r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))

    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso.coef_,
        'Abs_Coefficient': np.abs(lasso.coef_)})
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
    if verbose:
        lasso_selected = coef_df[coef_df['Coefficient'] != 0].sort_values('Abs_Coefficient', ascending=False)
        for _, row in lasso_selected.iterrows():
            direction = "+" if row['Coefficient'] > 0 else "-"
            print(f"  {direction} {row['Feature']:35s} (coef={row['Coefficient']:8.4f})")
        print(f"\nOptimal alpha: {lasso.alpha_:.6f}")
        print(f"Selected {len(selected_features)} features")
        print(f"R²: {r2:.4f}")
    return selected_features, lasso, coef_df

def validate_time_varying_features():
    """Step 4: Validate time-varying features for Fixed Effects models."""
    print("\n" + "="*49)
    print("STEP 4.4: TIME-VARYING FEATURE VALIDATION")
    print("="*49)
    
    panel = pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')
    
    # Identify time-varying features
    print("\nIdentifying time-varying features...")
    time_varying = []
    exclude_cols = ['FIPS', 'Year', 'move_net', 'move_in', 'move_out', 'agi_in', 'agi_out']
    
    for col in panel.columns:
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(panel[col]):
            continue
        within_var = panel.groupby('FIPS')[col].std().mean()
        if pd.notna(within_var) and within_var >= 0.0001:
            time_varying.append(col)
    
    print(f"\nIdentified {len(time_varying)} time-varying features")
    
    # Test features in FE model
    print("\nTesting features in Fixed Effects model...")
    panel_indexed = panel[['FIPS', 'Year', 'move_net'] + time_varying].dropna().set_index(['FIPS', 'Year'])
    y = panel_indexed['move_net']
    
    results = []
    
    for i, feature in enumerate(time_varying, 1):
        if i % 10 == 0:
            print(f"  Testing feature {i}/{len(time_varying)}...")
        
        X_feature = panel_indexed[[feature]]
        model = PanelOLS(y, X_feature, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
        
        results.append({
            'Feature': feature,
            'Coefficient': model.params[feature],
            'Std_Error': model.std_errors[feature],
            'T_Stat': model.tstats[feature],
            'P_value': model.pvalues[feature],
            'R2_Within': model.rsquared_within,
            'N_Obs': model.nobs,
            'Significant': model.pvalues[feature] < 0.05})
    
    results_df = pd.DataFrame(results).sort_values('P_value')
    
    print(f"\nSuccessfully tested: {len(results_df)} features")
    print(f"Statistically significant (p<0.05): {(results_df['Significant']).sum()}")
    
    output_path = OUTPUTS_DIR / 'tables'
    results_df.to_csv(output_path / 'time_varying_validation.csv', index=False)
    
    print(f"\nTop {min(15, len(results_df))} most significant features:")
    display_cols = ['Feature', 'Coefficient', 'P_value', 'R2_Within', 'N_Obs']
    print(results_df.head(15)[display_cols].to_string(index=False))
    return results_df

def compare_methods():
    """Complete feature selection pipeline: VIF → Stepwise → LASSO → Time-Varying."""
    print("\n" + "="*49)
    print("STEP 4: FEATURE SELECTION")
    print("="*49)
    
    vif_results = calculate_vif()
    
    X, y, all_features = load_panel_for_selection()
    print(f"\nTotal observations: {len(X):,}")
    print(f"Candidate features: {len(all_features)}")
    
    stepwise_features, stepwise_model = stepwise_selection(X, y)
    lasso_features, lasso_model, lasso_coef = lasso_selection(X, y)
    time_varying_results = validate_time_varying_features()
    
    print("\n" + "="*49)
    print("COMPARISON RESULTS")
    print("="*49)
    
    both = set(stepwise_features) & set(lasso_features)
    
    print(f"\nStepwise selected: {len(stepwise_features)} features")
    print(f"LASSO selected: {len(lasso_features)} features")
    print(f"Both methods: {len(both)} features")
    
    output_path = OUTPUTS_DIR / 'tables'
    
    X_scaled = StandardScaler().fit_transform(X)
    lasso_r2 = lasso_model.score(X_scaled, y)
    
    comparison = pd.DataFrame({
        'Method': ['Stepwise', 'LASSO'],
        'N_Features': [len(stepwise_features), len(lasso_features)],
        'R2': [stepwise_model.rsquared, lasso_r2]})
    comparison.to_csv(output_path / 'feature_selection_comparison.csv', index=False)
    
    features_df = pd.DataFrame({
        'Feature': all_features,
        'Stepwise': [f in stepwise_features for f in all_features],
        'LASSO': [f in lasso_features for f in all_features],
        'Both': [f in both for f in all_features]})
    features_df.to_csv(output_path / 'selected_features.csv', index=False)
    
    lasso_coef.to_csv(output_path / 'lasso_coefficients.csv', index=False)
    
    with open(output_path / 'stepwise_summary.txt', 'w') as f:
        f.write(stepwise_model.summary().as_text())
    
    print("\nFiles saved:")
    print("  - vif_analysis.csv")
    print("  - feature_selection_comparison.csv")
    print("  - selected_features.csv")
    print("  - lasso_coefficients.csv")
    print("  - stepwise_summary.txt")
    print("  - time_varying_validation.csv")
    
    print("\nFeatures selected by BOTH methods:")
    for feature in sorted(both):
        print(f"  {feature}")

    recommended_tv = time_varying_results[
        (time_varying_results['Significant'] == True) & 
        (time_varying_results['R2_Within'] > 0.01)]
    
    print(f"\nRecommended time-varying features for FE models ({len(recommended_tv)}):")
    for _, row in recommended_tv.head(15).iterrows():
        print(f"  {row['Feature']:30s} (p={row['P_value']:.4f}, R²={row['R2_Within']:.4f})")
    
    print("\n" + "="*49)
    print("STEP 4 COMPLETE")
    print("="*49)
    
    return {
        'vif': vif_results,
        'stepwise_features': stepwise_features,
        'lasso_features': lasso_features,
        'both': both,
        'stepwise_model': stepwise_model,
        'lasso_model': lasso_model,
        'time_varying': time_varying_results}

if __name__ == '__main__':
    results = compare_methods()