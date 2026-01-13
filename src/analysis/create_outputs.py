"""Generate analysis outputs, tables, and reports for academic publication."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from config.settings import PROCESSED_DATA_DIR, OUTPUTS_DIR

# ================================================
# HELPER FUNCTIONS (Module-level)
# ================================================

def _sig_stars(p):
    """Convert p-value to significance stars (academic convention)."""
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    else:
        return ''

def _load_data_once():
    """Load all data files once to avoid repetition."""
    print("Loading data files...")
    
    data = {
        'panel': pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv'),
        'model1': pd.read_csv(OUTPUTS_DIR / 'tables' / 'MODEL-1_gravity_results.csv'),
        'model2': pd.read_csv(OUTPUTS_DIR / 'tables' / 'MODEL-2_feature_comparison.csv'),
        'model3': pd.read_csv(OUTPUTS_DIR / 'tables' / 'MODEL-3_did_comparison.csv'),
        'model4': pd.read_csv(OUTPUTS_DIR / 'tables' / 'MODEL-4_dynamic_results.csv'),
        'X_final': pd.read_csv(OUTPUTS_DIR / 'tables' / 'X_final.csv')['Feature'].tolist()}
    
    return data

# ================================================
# ACADEMIC STANDARD FUNCTIONS
# ================================================

def save_academic_results(model, model_name, description, output_dir):
    """Save model results in academic publication format.
    
    Call this from run_models.py after fitting each model.
    
    Parameters:
    -----------
    model : fitted model object (statsmodels or linearmodels)
    model_name : str (e.g., 'MODEL-1_gravity')
    description : str (model description)
    output_dir : Path (directory to save results)
    """
    
    f_stat_obj = getattr(model, 'f_statistic', None)
    if f_stat_obj is not None:
        if hasattr(f_stat_obj, 'stat'):
            f_stat = float(f_stat_obj.stat)
            f_pval = float(f_stat_obj.pval)
        else:
            f_stat = float(f_stat_obj)
            f_pval = np.nan
    else:
        f_stat = np.nan
        f_pval = np.nan
    
    # Model statistics
    stats = {
        'Model': model_name,
        'Description': description,
        'N_Observations': int(model.nobs),
        'R_squared': float(model.rsquared),
        'Adj_R_squared': float(getattr(model, 'rsquared_adj', model.rsquared)),
        'F_statistic': f_stat,
        'F_pvalue': f_pval}
    
    # Coefficient table with confidence intervals
    coef_table = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': (model.std_errors.values if hasattr(model, 'std_errors') 
                     else model.bse.values),
        'T_Statistic': (model.tstats.values if hasattr(model, 'tstats') 
                       else model.tvalues.values),
        'P_value': model.pvalues.values,
        'CI_Lower_95': model.conf_int().iloc[:, 0].values,
        'CI_Upper_95': model.conf_int().iloc[:, 1].values})
    
    # Add significance stars
    coef_table['Sig'] = coef_table['P_value'].apply(_sig_stars)
    
    # Round for publication
    coef_table['Coefficient'] = coef_table['Coefficient'].round(4)
    coef_table['Std_Error'] = coef_table['Std_Error'].round(4)
    coef_table['T_Statistic'] = coef_table['T_Statistic'].round(3)
    coef_table['P_value'] = coef_table['P_value'].round(4)
    coef_table['CI_Lower_95'] = coef_table['CI_Lower_95'].round(4)
    coef_table['CI_Upper_95'] = coef_table['CI_Upper_95'].round(4)
    
    # Save files
    coef_table.to_csv(output_dir / f'{model_name}_coefficients.csv', index=False)
    
    # Save model statistics as separate CSV (ADD THIS):
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / f'{model_name}_stats.csv', index=False)
    
    # Summary statistics
    with open(output_dir / f'{model_name}_summary.txt', 'w') as f:
        f.write(f"Model: {stats['Model']}\n")
        f.write(f"Description: {stats['Description']}\n")
        f.write(f"\nModel Statistics:\n")
        f.write(f"  N Observations: {stats['N_Observations']:,}\n")
        f.write(f"  R-squared: {stats['R_squared']:.4f}\n")
        f.write(f"  Adj. R-squared: {stats['Adj_R_squared']:.4f}\n")
        f.write(f"  F-statistic: {stats['F_statistic']:.2f}\n")
        f.write(f"  F p-value: {stats['F_pvalue']:.4f}\n")
        f.write(f"\nSignificance: *** p<0.01, ** p<0.05, * p<0.10\n")
    
    print(f"Saved academic results: {model_name}")
    return coef_table, stats

def create_descriptive_stats(data):
    """Create comprehensive descriptive statistics table (Table 1)."""
    print("Generating descriptive statistics...")
    
    panel = data['panel']
    
    # Key variables for descriptive stats
    desc_vars = [
        'move_net', 'total_population', 'median_hh_income', 
        'median_home_value', 'median_property_taxes',
        'unemploy_rate', 'BEA_PCI', 'RPP', 'Amenity_scale']
    
    # Overall statistics
    overall = panel[desc_vars].describe().T
    overall['N'] = panel[desc_vars].count()
    overall = overall[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'N']]
    overall.columns = ['Mean', 'SD', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'N']
    overall['Variable'] = overall.index
    overall = overall[['Variable', 'Mean', 'SD', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'N']]
    overall.to_csv(OUTPUTS_DIR / 'tables' / 'descriptive_stats_overall.csv', index=False)
    
    # By treatment/control
    by_incentive = panel.groupby('has_incentive')[desc_vars].describe().T
    
    # Flatten column names
    if isinstance(by_incentive.columns, pd.MultiIndex):
        by_incentive.columns = ['_'.join(map(str, col)) for col in by_incentive.columns]
    by_incentive.to_csv(OUTPUTS_DIR / 'tables' / 'descriptive_stats_by_treatment.csv')
    
    print("  Saved: descriptive_stats_overall.csv, descriptive_stats_by_treatment.csv")
    return overall, by_incentive

def create_correlation_matrix(data):
    """Create correlation matrix for key variables."""
    print("Generating correlation matrix...")
    
    panel = data['panel']
    X_final = data['X_final']
    
    # Compute correlation
    available = [f for f in X_final if f in panel.columns]
    corr = panel[available].corr()
    
    # Save
    corr.to_csv(OUTPUTS_DIR / 'tables' / 'correlation_matrix.csv')
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix - Final Features')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'figures' / 'correlation_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Saved: correlation_matrix.csv, correlation_matrix.png")
    return corr

def create_balance_table(data):
    """Create balance table comparing treatment vs control (pre-treatment)."""
    print("Generating balance table...")
    
    panel = data['panel']
    
    # Pre-treatment period (before any incentive starts)
    pre_treatment = panel[panel['Year'] < panel['Incentive_CAT'].min()]
    
    # Key variables
    balance_vars = [
        'total_population', 'median_hh_income', 'median_home_value',
        'unemploy_rate', 'BEA_PCI', 'RPP', 'RUCC_code']
    
    # Group by treatment status
    treatment = pre_treatment[pre_treatment['has_incentive'] == 1][balance_vars].mean()
    control = pre_treatment[pre_treatment['has_incentive'] == 0][balance_vars].mean()
    
    # T-tests for differences
    from scipy import stats
    p_values = []
    for var in balance_vars:
        t_stat, p_val = stats.ttest_ind(
            pre_treatment[pre_treatment['has_incentive'] == 1][var].dropna(),
            pre_treatment[pre_treatment['has_incentive'] == 0][var].dropna())
        p_values.append(p_val)
    
    # Balance table
    balance = pd.DataFrame({
        'Variable': balance_vars,
        'Treatment_Mean': treatment.values,
        'Control_Mean': control.values,
        'Difference': (treatment - control).values,
        'P_value': p_values})
    
    balance['Sig'] = balance['P_value'].apply(_sig_stars)
    balance.to_csv(OUTPUTS_DIR / 'tables' / 'balance_table_pretreatment.csv', index=False)
    
    print("  Saved: balance_table_pretreatment.csv")
    return balance

# ================================================
# ANALYSIS FUNCTIONS
# ================================================

def create_master_comparison(data):
    """Create master model comparison table."""
    print("Generating master model comparison...")
    
    model1 = data['model1']
    model1_stats = pd.read_csv(OUTPUTS_DIR / 'tables' / 'MODEL-1_gravity_stats.csv')
    model2 = data['model2']
    model3 = data['model3']
    model4 = data['model4']
    
    master = pd.DataFrame({
        'Model': [
            'Gravity_Flow',
            'Panel_Pooled_OLS',
            'Panel_County_FE',
            'Panel_TwoWay_FE',
            'DiD_Simple',
            'DiD_Controls',
            'Dynamic_Panel'],
        'R2': [
            model1.loc[model1['Statistic'] == 'R2', 'Value'].iloc[0],
            model2.loc[model2['Model'] == 'Pooled_OLS', 'R2_Overall'].iloc[0],
            model2.loc[model2['Model'] == 'County_FE_no_commute', 'R2_Overall'].iloc[0],  # ← CHANGED
            model2.loc[model2['Model'] == 'TwoWay_FE_no_commute', 'R2_Overall'].iloc[0],  # ← CHANGED
            model3.loc[model3['Specification'] == 'Simple', 'R2_Within'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'R2_Within'].iloc[0],
            model4['R2_Within'].iloc[0]],
        'R2_Within': [
            None,  
            model2.loc[model2['Model'] == 'Pooled_OLS', 'R2_Within'].iloc[0],
            model2.loc[model2['Model'] == 'County_FE_no_commute', 'R2_Within'].iloc[0],  # ← CHANGED
            model2.loc[model2['Model'] == 'TwoWay_FE_no_commute', 'R2_Within'].iloc[0],  # ← CHANGED
            model3.loc[model3['Specification'] == 'Simple', 'R2_Within'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'R2_Within'].iloc[0],
            model4['R2_Within'].iloc[0]],
        'N_Obs': [
            model1.loc[model1['Statistic'] == 'N_Obs', 'Value'].iloc[0],
            model2.loc[model2['Model'] == 'Pooled_OLS', 'N_Obs'].iloc[0],
            model2.loc[model2['Model'] == 'County_FE_no_commute', 'N_Obs'].iloc[0],  # ← CHANGED
            model2.loc[model2['Model'] == 'TwoWay_FE_no_commute', 'N_Obs'].iloc[0],  # ← CHANGED
            model3.loc[model3['Specification'] == 'Simple', 'N_Obs'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'N_Obs'].iloc[0],
            model4['N_Obs'].iloc[0]],
        'N_Features': [
            model1.loc[model1['Statistic'] == 'N_Features', 'Value'].iloc[0],
            model2.loc[model2['Model'] == 'Pooled_OLS', 'N_Features'].iloc[0],
            model2.loc[model2['Model'] == 'County_FE_no_commute', 'N_Features'].iloc[0],  # ← CHANGED
            model2.loc[model2['Model'] == 'TwoWay_FE_no_commute', 'N_Features'].iloc[0],  # ← CHANGED
            None,  
            None,
            None]})
    
    master.to_csv(OUTPUTS_DIR / 'tables' / 'MASTER_MODEL_COMPARISON.csv', index=False)
    print("  Saved: MASTER_MODEL_COMPARISON.csv")
    return master

def migration_flow_matrix(data):
    """Analyze bilateral flows by origin-destination type."""
    print("Generating migration flow matrix...")
    
    bilateral = pd.read_csv(PROCESSED_DATA_DIR / 'IRS_gravity.csv')
    rucc_2013 = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_RUCC_2013.csv')
    rucc_2023 = pd.read_csv(PROCESSED_DATA_DIR / 'USDA_RUCC_2023_clean.csv')
    
    # Assign RUCC
    rucc_2013_panel = pd.concat([rucc_2013.assign(Year=y) for y in range(2011, 2020)], 
                                ignore_index=True).rename(columns={'RUCC_2013': 'RUCC_code'})
    rucc_2023_panel = pd.concat([rucc_2023.assign(Year=y) for y in range(2020, 2022)], 
                                ignore_index=True).rename(columns={'RUCC_2023': 'RUCC_code'})
    rucc_all = pd.concat([rucc_2013_panel, rucc_2023_panel], ignore_index=True)
    
    # Standardize FIPS
    for df in [bilateral, rucc_all]:
        if 'FIPS' in df.columns:
            df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
        if 'dest_FIPS' in df.columns:
            df['dest_FIPS'] = df['dest_FIPS'].astype(str).str.zfill(5)
        if 'origin_FIPS' in df.columns:
            df['origin_FIPS'] = df['origin_FIPS'].astype(str).str.zfill(5)
    
    # Merge RUCC
    bilateral = bilateral.merge(
        rucc_all.rename(columns={'FIPS': 'origin_FIPS', 'RUCC_code': 'origin_rucc'}),
        on=['origin_FIPS', 'Year'], how='left')
    bilateral = bilateral.merge(
        rucc_all.rename(columns={'FIPS': 'dest_FIPS', 'RUCC_code': 'dest_rucc'}),
        on=['dest_FIPS', 'Year'], how='left')
    
    # Classify: Urban (1-3), Suburban (4-7), Rural (8-9)
    bilateral['origin_type'] = pd.cut(bilateral['origin_rucc'], 
                                     bins=[0, 3, 7, 9], 
                                     labels=['Urban', 'Suburban', 'Rural'])
    bilateral['dest_type'] = pd.cut(bilateral['dest_rucc'], 
                                   bins=[0, 3, 7, 9], 
                                   labels=['Urban', 'Suburban', 'Rural'])
    
    # Calculate flows
    flows = bilateral.groupby(['origin_type', 'dest_type'])['movers'].sum().reset_index()
    flows_pivot = flows.pivot(index='origin_type', columns='dest_type', values='movers')
    
    flows_pivot.to_csv(OUTPUTS_DIR / 'tables' / 'migration_flow_matrix.csv')
    print("  Saved: migration_flow_matrix.csv")
    return flows_pivot

def create_hypothesis_tables(data):
    """Create hypothesis testing tables."""
    print("Generating hypothesis tables...")
    
    panel = data['panel']
    
    # H1: Incentive vs Non-Incentive
    h1 = panel.groupby('has_incentive').agg({
        'move_net': ['mean', 'median', 'std', 'count'],
        'total_population': 'mean',
        'median_hh_income': 'mean',
        'FIPS': 'nunique'})
    
    h1.columns = ['_'.join(col) for col in h1.columns]
    h1.reset_index(inplace=True)
    h1['has_incentive'] = h1['has_incentive'].map({0: 'No_Incentive', 1: 'Has_Incentive'})
    h1.to_csv(OUTPUTS_DIR / 'tables' / 'H1_incentive_comparison.csv', index=False)
    
    print("  Saved: H1_incentive_comparison.csv")
    return h1

def program_timing_analysis(data):
    """Analyze treatment effects by program timing."""
    print("Generating program timing analysis...")
    
    panel = data['panel']
    X_final = data['X_final']
    
    # Split by timing
    panel['Pre_COVID'] = ((panel['Incentive_CAT'] < 2020) & 
                         (panel['has_incentive'] == 1)).astype(int)
    panel['COVID_Era'] = ((panel['Incentive_CAT'] >= 2020) & 
                         (panel['has_incentive'] == 1)).astype(int)
    panel['POST'] = (panel['Year'] >= panel['Incentive_CAT']).fillna(0).astype(int)
    
    # Create interactions
    panel['PreCOVID_treat'] = panel['Pre_COVID'] * panel['POST']
    panel['COVIDEra_treat'] = panel['COVID_Era'] * panel['POST']
    
    # Prepare data
    from linearmodels.panel import PanelOLS
    available_features = [f for f in X_final if f in panel.columns]
    panel_indexed = panel.dropna(subset=['move_net'] + available_features).set_index(['FIPS', 'Year'])
    
    y = panel_indexed['move_net']
    X = panel_indexed[['PreCOVID_treat', 'COVIDEra_treat'] + available_features]
    
    # Estimate
    model = PanelOLS(y, X, entity_effects=True, time_effects=True, 
                     drop_absorbed=True, check_rank=False).fit(
                         cov_type='clustered', cluster_entity=True)
    
    # Save results
    timing_results = pd.DataFrame({
        'Program_Timing': ['Pre-COVID Programs', 'COVID-Era Programs'],
        'Estimated_Effect': [
            model.params.get('PreCOVID_treat', np.nan),
            model.params.get('COVIDEra_treat', np.nan)],
        'Std_Error': [
            model.std_errors.get('PreCOVID_treat', np.nan),
            model.std_errors.get('COVIDEra_treat', np.nan)],
        'P_value': [
            model.pvalues.get('PreCOVID_treat', np.nan),
            model.pvalues.get('COVIDEra_treat', np.nan)]})
    
    timing_results['Sig'] = timing_results['P_value'].apply(_sig_stars)
    timing_results.to_csv(OUTPUTS_DIR / 'tables' / 'program_timing_results.csv', index=False)
    print("  Saved: program_timing_results.csv")
    return timing_results

def treatment_effects_table(data):
    """Create comprehensive treatment effects table."""
    print("Generating treatment effects table...")
    
    model3 = data['model3']
    timing = pd.read_csv(OUTPUTS_DIR / 'tables' / 'program_timing_results.csv')
    
    # Extract all treatment effects
    treatment = pd.DataFrame({
        'Specification': [
            'Overall (Simple)',
            'Overall (Controls)',
            'Low Value (CAT1)',
            'Medium Value (CAT2)',
            'High Value (CAT3)',
            'Pre-COVID Programs',
            'COVID-Era Programs'],
        'Effect': [
            model3.loc[model3['Specification'] == 'Simple', 'Treatment_Effect'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'Treatment_Effect'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT1_Low', 'Treatment_Effect'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT2_Med', 'Treatment_Effect'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT3_High', 'Treatment_Effect'].iloc[0],
            timing.loc[0, 'Estimated_Effect'],
            timing.loc[1, 'Estimated_Effect']],
        'Std_Error': [
            model3.loc[model3['Specification'] == 'Simple', 'Std_Error'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'Std_Error'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT1_Low', 'Std_Error'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT2_Med', 'Std_Error'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT3_High', 'Std_Error'].iloc[0],
            timing.loc[0, 'Std_Error'],
            timing.loc[1, 'Std_Error']],
        'P_value': [
            model3.loc[model3['Specification'] == 'Simple', 'P_value'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'P_value'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT1_Low', 'P_value'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT2_Med', 'P_value'].iloc[0],
            model3.loc[model3['Specification'] == 'CAT3_High', 'P_value'].iloc[0],
            timing.loc[0, 'P_value'],
            timing.loc[1, 'P_value']]})
    
    # Add significance and CIs
    treatment['Sig'] = treatment['P_value'].apply(_sig_stars)
    treatment['CI_Lower_95'] = treatment['Effect'] - 1.96 * treatment['Std_Error']
    treatment['CI_Upper_95'] = treatment['Effect'] + 1.96 * treatment['Std_Error']
    
    treatment.to_csv(OUTPUTS_DIR / 'tables' / 'treatment_effects_comprehensive.csv', index=False)
    print("  Saved: treatment_effects_comprehensive.csv")
    return treatment

def create_key_findings(data):
    """Create key findings summary report."""
    print("Generating key findings summary...")
    
    panel = data['panel']
    model2 = data['model2']
    model3 = data['model3']
    model4 = data['model4']
    X_final = data['X_final']
    timing = pd.read_csv(OUTPUTS_DIR / 'tables' / 'program_timing_results.csv')
    
    # Extract values
    persistence = model4.loc[model4['Variable'] == 'move_net_lag1', 'Coefficient'].iloc[0]
    lr_mult = model4['LR_Multiplier'].iloc[0]
    
    did_controls_effect = model3.loc[model3['Specification'] == 'Controls', 'Treatment_Effect'].iloc[0]
    did_controls_p = model3.loc[model3['Specification'] == 'Controls', 'P_value'].iloc[0]
    
    findings = f"""
KEY FINDINGS SUMMARY
====================

SAMPLE CHARACTERISTICS:
  • Total observations: {len(panel):,}
  • Counties: {panel['FIPS'].nunique():,}
  • Years: {panel['Year'].min()}-{panel['Year'].max()}
  • Incentive counties: {panel[panel['has_incentive']==1]['FIPS'].nunique():,}

TREATMENT EFFECTS (Difference-in-Differences):
  • Overall effect (with controls): {did_controls_effect:.2f} net migrants
    (p-value: {did_controls_p:.4f}) {_sig_stars(did_controls_p)}

MIGRATION PERSISTENCE (Dynamic Panel):
  • Coefficient (ρ): {persistence:.4f}
  • Long-run multiplier: {lr_mult:.4f}

FEATURE SELECTION:
  • Validated features: {len(X_final)}

Significance: *** p<0.01, ** p<0.05, * p<0.10
"""
    
    output_path = OUTPUTS_DIR / 'reports' / 'KEY_FINDINGS_SUMMARY.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(findings)
    print("  Saved: KEY_FINDINGS_SUMMARY.txt")
    return findings

# ================================================
# MAIN RUN FUNCTION
# ================================================

def run():
    """Run all analysis outputs."""
    print("="*49)
    print("ANALYSIS & OUTPUT GENERATION")
    print("="*49 + "\n")
    
    # Create directories
    (OUTPUTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    
    # Load data once
    data = _load_data_once()
    
    # Academic standard outputs
    print("\n--- ACADEMIC STANDARD OUTPUTS ---")
    create_descriptive_stats(data)
    create_correlation_matrix(data)
    create_balance_table(data)
    
    # Analysis outputs
    print("\n--- ANALYSIS OUTPUTS ---")
    create_master_comparison(data)
    migration_flow_matrix(data)
    create_hypothesis_tables(data)
    program_timing_analysis(data)
    treatment_effects_table(data)
    
    # Summary
    print("\n--- SUMMARY ---")
    findings = create_key_findings(data)
    print("\n" + findings)
    
    print("\nStep 6: Analysis complete\n")

if __name__ == '__main__':
    run()
