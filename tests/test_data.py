"""Basic validation tests for data pipeline."""
from config.settings import RAW_DATA_DIR, REQUIRED_FILES

def test_required_files_exist():
    """Check all 16 required manual download files are present."""
    missing = []
    for filename in REQUIRED_FILES:
        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        print("\n❌ ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nExpected location: {RAW_DATA_DIR}")
        return False
    else:
        print(f"\n✅ All {len(REQUIRED_FILES)} required files found")
        return True

def test_api_keys_loaded():
    """Verify API keys are configured."""
    from config.settings import API_KEY_BEA, API_KEY_CENSUS, API_KEY_BLS
    
    keys_ok = True
    if not API_KEY_BEA or API_KEY_BEA == 'your_key_here':
        print("❌ BEA_API_KEY not configured")
        keys_ok = False
    if not API_KEY_CENSUS or API_KEY_CENSUS == 'your_key_here':
        print("❌ CENSUS_API_KEY not configured")
        keys_ok = False
    if not API_KEY_BLS or API_KEY_BLS == 'your_key_here':
        print("❌ BLS_API_KEY not configured")
        keys_ok = False
    
    if keys_ok:
        print("✅ All API keys configured")
    return keys_ok

def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*49)
    print("RUNNING VALIDATION TESTS")
    print("="*49)
    
    files_ok = test_required_files_exist()
    keys_ok = test_api_keys_loaded()
    
    if files_ok and keys_ok:
        print("\n✅ All tests passed! Ready to run pipeline.")
        return True
    else:
        print("\n❌ Some tests failed. Fix issues before running pipeline.")
        return False

if __name__ == '__main__':
    run_all_tests()
