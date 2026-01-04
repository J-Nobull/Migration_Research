"""Configuration settings for migration research pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# API Keys
API_KEY_BEA = os.getenv('BEA_API_KEY', 'your_key_here')
API_KEY_CENSUS = os.getenv('CENSUS_API_KEY', 'your_key_here')
API_KEY_BLS = os.getenv('BLS_API_KEY', 'your_key_here')

# Validate API keys
if API_KEY_BEA == 'your_key_here':
    raise ValueError("BEA_API_KEY not set")
if API_KEY_CENSUS == 'your_key_here':
    raise ValueError("CENSUS_API_KEY not set")
if API_KEY_BLS == 'your_key_here':
    raise ValueError("BLS_API_KEY not set")

# Study Parameters
YEARS = list(range(2011, 2022))

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'reports').mkdir(parents=True, exist_ok=True)

# Required manual download files
REQUIRED_FILES = [
    'Incentives.csv',
    'countyinflow1112.csv', 'countyinflow1213.csv', 'countyinflow1314.csv',
    'countyinflow1415.csv', 'countyinflow1516.csv', 'countyinflow1617.csv',
    'countyinflow1718.csv', 'countyinflow1819.csv', 'countyinflow1920.csv',
    'countyinflow2021.csv', 'countyinflow2122.csv',
    'ruralurbancodes2013.xls',
    'Ruralurbancontinuumcodes2023.csv',
    'erscountytypology2015edition.csv',
    'natamenf_1_.xls']

# IRS files
IRS_FILES = [f for f in REQUIRED_FILES if f.startswith('countyinflow')]

print(f"\n{'='*49}")
print("CONFIGURATION LOADED")
print(f"{'='*49}")
print(f"Study period: {YEARS[0]}-{YEARS[-1]}")
print(f"Working directory: {PROJECT_ROOT}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*49}\n")
