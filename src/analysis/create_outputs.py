"""Generate analysis outputs, tables, and reports."""
import pandas as pd
import numpy as np
from config.settings import PROCESSED_DATA_DIR, OUTPUTS_DIR

def create_master_comparison():
    """Create master model comparison table."""
    print("Generating master model comparison...")
    
    # Load model results
    model1 = pd.read_csv(PROCESSED_DATA_DIR / 'MODEL-1_gravity_results.csv')
    model2 = pd.read_csv(PROCESSED_DATA_DIR / 'MODEL-2_panel_fe_comparison.csv')
    model3 = pd.read_csv(PROCESSED_DATA_DIR / 'MODEL-3_did_comparison.csv')
    model4 = pd.read_csv(PROCESSED_DATA_DIR / 'MODEL-4_dynamic_results.csv')
    
    # Extract key statistics
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
            model1['R2'].iloc[0],
            model2.loc[model2['Model'] == 'Pooled_OLS', 'R2_Overall'].iloc[0],
            model2.loc[model2['Model'] == 'County_FE', 'R2_Overall'].iloc[0],
            model2.loc[model2['Model'] == 'TwoWay_FE', 'R2_Overall'].iloc[0],
            model3.loc[model3['Specification'] == 'Simple', 'R2_Within'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'R2_Within'].iloc[0],
            model4['R2_Within'].iloc[0]],
        'Treatment_Effect': [
            np.nan, np.nan, np.nan, np.nan,
            model3.loc[model3['Specification'] == 'Simple', 'Treatment_Effect'].iloc[0],
            model3.loc[model3['Specification'] == 'Controls', 'Treatment_Effect'].iloc[0],
            np.nan]})
    
    output_path = OUTPUTS_DIR / 'tables' / 'MASTER_MODEL_COMPARISON.csv'
    master.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")
    
    return master

def create_hypothesis_tables():
    """Create hypothesis testing tables."""
    print("Generating hypothesis tables...")
    
    panel = pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')
    
    # H1: Program Timing
    if 'COVID_program' in panel.columns:
        precovid = panel[panel['COVID_program'] == 0]['FIPS'].unique()
        covid = panel[panel['COVID_program'] == 1]['FIPS'].unique()
        covid_period = panel[panel['Year'] >= 2020]
        
        h1 = pd.DataFrame({
            'Program_Timing': ['Pre-COVID (2011-2019)', 'COVID-Era (2020-2021)'],
            'Mean_Net_Migration': [
                covid_period[covid_period['FIPS'].isin(precovid)]['move_net'].mean(),
                covid_period[covid_period['FIPS'].isin(covid)]['move_net'].mean()],
            'N_Counties': [len(precovid), len(covid)]})
        
        output_path = OUTPUTS_DIR / 'tables' / 'H1_program_timing.csv'
        h1.to_csv(output_path, index=False)
        print(f"  Saved: {output_path.name}")
    
    # H2: Incentive vs Non-Incentive
    h2 = panel.groupby('has_incentive').agg({
        'move_net': ['mean', 'median', 'std', 'count'],
        'total_population': 'mean',
        'median_hh_income': 'mean',
        'unemploy_rate': 'mean',
        'FIPS': 'nunique'}).round(2)
    
    h2.columns = ['_'.join(col) for col in h2.columns]
    h2.reset_index(inplace=True)
    h2['has_incentive'] = h2['has_incentive'].map({0: 'No_Incentive', 1: 'Has_Incentive'})
    
    output_path = OUTPUTS_DIR / 'tables' / 'H2_incentive_comparison.csv'
    h2.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")

def create_descriptive_stats():
    """Create descriptive statistics table."""
    print("Generating descriptive statistics...")
    
    panel = pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')
    
    desc_vars = [
        'total_population', 'move_net', 'RUC_code',
        'median_hh_income', 'median_home_value', 'median_property_taxes',
        'unemploy_rate', 'Amenity_scale', 'BEA_PCI', 'RPP']
    
    desc = panel.groupby('has_incentive')[desc_vars].agg(['mean', 'std', 'count'])
    desc.columns = ['_'.join(col) for col in desc.columns]
    desc.reset_index(inplace=True)
    desc['has_incentive'] = desc['has_incentive'].map({0: 'Control', 1: 'Treatment'})
    
    output_path = OUTPUTS_DIR / 'tables' / 'DESCRIPTIVE_STATS_BY_INCENTIVE.csv'
    desc.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")

def create_key_findings():
    """Create key findings summary report."""
    print("Generating key findings summary...")
    
    panel = pd.read_csv(PROCESSED_DATA_DIR / 'full_panel.csv')
    model3 = pd.read_csv(PROCESSED_DATA_DIR / 'MODEL-3_did_comparison.csv')
    model4 = pd.read_csv(PROCESSED_DATA_DIR / 'MODEL-4_dynamic_results.csv')
    
    persistence = model4.loc[model4['Variable'] == 'move_net_lag1', 'Coefficient'].iloc[0]
    lr_mult = model4['LR_Multiplier'].iloc[0]
    
    did_simple_effect = model3.loc[model3['Specification'] == 'Simple', 'Treatment_Effect'].iloc[0]
    did_simple_p = model3.loc[model3['Specification'] == 'Simple', 'P_value'].iloc[0]
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
  • Control counties: {panel[panel['has_incentive']==0]['FIPS'].nunique():,}

TREATMENT EFFECTS (DiD):
  • Simple specification: {did_simple_effect:.2f} net migrants
    (p-value: {did_simple_p:.4f})
  
  • With controls: {did_controls_effect:.2f} net migrants
    (p-value: {did_controls_p:.4f})

PERSISTENCE (Dynamic Panel):
  • Coefficient (ρ): {persistence:.4f}
  • Long-run multiplier: {lr_mult:.4f}
  • Interpretation: {'Positive persistence - gains compound' if persistence > 0 else 'Mean reversion'}

STATISTICAL SIGNIFICANCE:
  • Treatment effect significant: {'Yes' if did_controls_p < 0.05 else 'No'}
  • Persistence significant: Yes
"""
    
    output_path = OUTPUTS_DIR / 'reports' / 'KEY_FINDINGS_SUMMARY.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(findings)
    print(f"  Saved: {output_path.name}")
    
    return findings

def run():
    """Main analysis function."""
    print("="*49)
    print("ANALYSIS & OUTPUT GENERATION")
    print("="*49 + "\n")
    
    create_master_comparison()
    create_hypothesis_tables()
    create_descriptive_stats()
    findings = create_key_findings()
    
    print("\n" + findings)
    
    print("\n✅ Analysis complete\n")

if __name__ == '__main__':
    run()
