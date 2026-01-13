"""
Main pipeline controller for migration research.
Supports both full pipeline execution and step-by-step runs.

Usage:
    python main.py              # Run full pipeline
    python main.py --step 1     # Run only data acquisition
    python main.py --step 2     # Run only preprocessing
    python main.py --step 3     # Run only feature engineering
    python main.py --step 4     # Run only feature selection
    python main.py --step 5     # Run only modeling
    python main.py --step 6     # Run only analysis
"""
import argparse
import warnings
from datetime import datetime
from tests.test_data import run_all_tests
warnings.filterwarnings('ignore')

def run_step_1():
    """Step 1: Data Acquisition"""
    print("\n" + "="*49)
    print("STEP 1: DATA ACQUISITION")
    print("="*49 + "\n")
    from src.acquisition import bea, census, bls, irs, usda
    
    print("Running BEA data acquisition...")
    bea.run()
    print("\nRunning Census data acquisition...")
    census.run()
    print("\nRunning BLS data acquisition...")
    bls.run()
    print("\nRunning IRS data acquisition...")
    irs.run()
    print("\nRunning USDA data loading...")
    usda.run()
    print("\n Step 1 complete\n")

def run_step_2():
    """Step 2: Data Preprocessing"""
    print("\n" + "="*49)
    print("STEP 2: DATA PREPROCESSING")
    print("="*49 + "\n")
    from src.preprocessing import clean_data
    clean_data.run()
    print("\n Step 2 complete\n")

def run_step_3():
    """Step 3: Feature Engineering & Merge"""
    print("\n" + "="*49)
    print("STEP 3: FEATURE ENGINEERING AND MERGE")
    print("="*49 + "\n")
    from src.features import engineer_features
    engineer_features.run()
    print("\n Step 3 complete\n")

def run_step_4():
    """Step 4: Feature Selection (VIF, Stepwise, LASSO)."""
    print("\n" + "="*49)
    print("STEP 4: FEATURE SELECTION")
    print("="*49 + "\n")
    from src.validation import feature_selection
    feature_selection.compare_methods()
    print("\n Step 4 complete\n")

def run_step_5():
    """Step 5: Statistical Modeling."""
    print("\n" + "="*49)
    print("STEP 5: STATISTICAL MODELING")
    print("="*49 + "\n")
    import src.models.run_models as run_models
    run_models.run()
    print("\n Step 5 complete\n")

def run_step_6():
    """Step 6: Analysis & Outputs."""
    print("\n" + "="*49)
    print("STEP 6: ANALYSIS & VISUALIZATION")
    print("="*49 + "\n")
    import src.analysis.create_outputs as outputs
    outputs.run()
    print("\n Step 6 complete\n")

def run_full_pipeline():
    """Run all steps in sequence"""
    print("\n" + "="*49)
    print("RUNNING FULL PIPELINE")
    print("="*49 + "\n")
    
    run_step_1()
    run_step_2()
    run_step_3()
    run_step_4()
    run_step_5()
    run_step_6()
    
    print("\n" + "="*49)
    print(" FULL PIPELINE COMPLETE")
    print("="*49 + "\n")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migration Research Pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5, 6],
                       help='Run specific step (1-6) or omit to run all')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip validation tests')
    
    args = parser.parse_args()
    
    # Run validation tests unless skipped
    if not args.skip_tests:
        print("\n" + "="*49)
        print("RUNNING VALIDATION TESTS")
        print("="*49 + "\n")
        if not run_all_tests():
            print("\n❌ Validation failed. Fix issues before running pipeline.")
            exit(1)
        print("\n All validation tests passed\n")
    
    steps = {1: run_step_1, 
             2: run_step_2, 
             3: run_step_3, 
             4: run_step_4, 
             5: run_step_5, 
             6: run_step_6}
    if args.step:
        steps[args.step]()
    else:
        run_full_pipeline()
