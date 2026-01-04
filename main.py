"""
Main pipeline controller for migration research.
Supports both full pipeline execution and step-by-step runs.

Usage:
    python main.py                    # Run full pipeline
    python main.py --step 1           # Run only data acquisition
    python main.py --step 2           # Run only preprocessing
    python main.py --step 3           # Run only feature engineering
    python main.py --step 4           # Run only modeling
    python main.py --step 5           # Run only analysis
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

from tests.test_data import run_all_tests

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
    
    print("\n✅ Step 1 complete\n")

def run_step_2():
    """Step 2: Data Preprocessing"""
    print("\n" + "="*49)
    print("STEP 2: DATA PREPROCESSING")
    print("="*49 + "\n")
    
    from src.preprocessing import clean_data
    clean_data.run()
    
    print("\n✅ Step 2 complete\n")

def run_step_3():
    """Step 3: Feature Engineering"""
    print("\n" + "="*49)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*49 + "\n")
    
    from src.features import engineer_features
    engineer_features.run()
    
    print("\n✅ Step 3 complete\n")

def run_step_4():
    """Step 4: Statistical Modeling"""
    print("\n" + "="*49)
    print("STEP 4: STATISTICAL MODELING")
    print("="*49 + "\n")
    
    from src.models import run_models
    run_models.run()
    
    print("\n✅ Step 4 complete\n")

def run_step_5():
    """Step 5: Analysis & Visualization"""
    print("\n" + "="*49)
    print("STEP 5: ANALYSIS & VISUALIZATION")
    print("="*49 + "\n")
    
    from src.analysis import create_outputs
    create_outputs.run()
    
    print("\n✅ Step 5 complete\n")

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
    
    print("\n" + "="*49)
    print("✅ FULL PIPELINE COMPLETE")
    print("="*49 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migration Research Pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5],
                       help='Run specific step (1-5) or omit to run all')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip validation tests')
    
    args = parser.parse_args()
    
    # Run validation tests unless skipped
    if not args.skip_tests:
        if not run_all_tests():
            print("\n❌ Validation failed. Fix issues before running pipeline.")
            exit(1)
    
    # Run requested step(s)
    if args.step == 1:
        run_step_1()
    elif args.step == 2:
        run_step_2()
    elif args.step == 3:
        run_step_3()
    elif args.step == 4:
        run_step_4()
    elif args.step == 5:
        run_step_5()
    else:
        run_full_pipeline()
