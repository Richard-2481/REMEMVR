#!/usr/bin/env python3
"""
Step 03: Extract Self-Report Scores (DASS, VR Experience, Sleep)
RQ 7.1.4: Unique REMEMVR variance unexplained by all predictors

Extract REAL DASS subscales, VR experience, and sleep data from dfnonvr.csv
Using exact column names from DATA_DICTIONARY.md
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
RQ_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = RQ_DIR.parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Set up logging
LOG_FILE = RQ_DIR / "logs" / "step03_extract_self_report.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    """Log to both console and file."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()

def main():
    """Main execution."""
    log("[START] Step 03: Extract self-report scores - USING REAL DATA")
    log("[INFO] All column names from DATA_DICTIONARY.md")
    
    # Load participant data
    log("[LOAD] Reading dfnonvr.csv...")
    df = pd.read_csv(PROJ_ROOT / "data" / "dfnonvr.csv")
    log(f"[INFO] Loaded {len(df)} participants with {len(df.columns)} columns")
    
    # Initialize output dataframe
    self_report = pd.DataFrame()
    self_report['uid'] = df['UID'].astype(str)
    
    # Extract DASS columns using EXACT names from DATA_DICTIONARY.md
    log("[EXTRACT] Extracting DASS subscales with correct column names...")
    
    # DASS Anxiety - exact column name: total-dass-anxiety-items
    if 'total-dass-anxiety-items' in df.columns:
        self_report['DASS_Anx'] = df['total-dass-anxiety-items']
        log(f"[SUCCESS] DASS Anxiety extracted from 'total-dass-anxiety-items'")
        log(f"  - Non-null values: {self_report['DASS_Anx'].notna().sum()}")
        log(f"  - Range: [{self_report['DASS_Anx'].min():.1f}, {self_report['DASS_Anx'].max():.1f}]")
    else:
        log("[ERROR] Column 'total-dass-anxiety-items' not found in dfnonvr.csv")
        log("[CRITICAL] Cannot proceed without real data - stopping")
        sys.exit(1)
    
    # DASS Stress - exact column name: total-dass-stress-items
    if 'total-dass-stress-items' in df.columns:
        self_report['DASS_Str'] = df['total-dass-stress-items']
        log(f"[SUCCESS] DASS Stress extracted from 'total-dass-stress-items'")
        log(f"  - Non-null values: {self_report['DASS_Str'].notna().sum()}")
        log(f"  - Range: [{self_report['DASS_Str'].min():.1f}, {self_report['DASS_Str'].max():.1f}]")
    else:
        log("[ERROR] Column 'total-dass-stress-items' not found in dfnonvr.csv")
        log("[CRITICAL] Cannot proceed without real data - stopping")
        sys.exit(1)
    
    # DASS Depression - exact column name: total-dass-depression-items
    if 'total-dass-depression-items' in df.columns:
        self_report['DASS_Dep'] = df['total-dass-depression-items']
        log(f"[SUCCESS] DASS Depression extracted from 'total-dass-depression-items'")
        log(f"  - Non-null values: {self_report['DASS_Dep'].notna().sum()}")
        log(f"  - Range: [{self_report['DASS_Dep'].min():.1f}, {self_report['DASS_Dep'].max():.1f}]")
    else:
        log("[ERROR] Column 'total-dass-depression-items' not found in dfnonvr.csv")
        log("[CRITICAL] Cannot proceed without real data - stopping")
        sys.exit(1)
    
    # VR Experience - exact column name: vr-exposure
    log("[EXTRACT] Extracting VR experience...")
    if 'vr-exposure' in df.columns:
        self_report['VR_Exp'] = df['vr-exposure']
        log(f"[SUCCESS] VR Experience extracted from 'vr-exposure'")
        log(f"  - Non-null values: {self_report['VR_Exp'].notna().sum()}")
        log(f"  - Range: [{self_report['VR_Exp'].min():.1f}, {self_report['VR_Exp'].max():.1f}]")
        log(f"  - Scale: 0=Never, 1=<1hr, 2=1-10hrs, 3=10-50hrs, 4=>50hrs")
    else:
        log("[ERROR] Column 'vr-exposure' not found in dfnonvr.csv")
        log("[CRITICAL] Cannot proceed without real data - stopping")
        sys.exit(1)
    
    # Sleep - exact column name: typical-sleep-hours
    log("[EXTRACT] Extracting sleep data...")
    if 'typical-sleep-hours' in df.columns:
        self_report['Sleep'] = df['typical-sleep-hours']
        log(f"[SUCCESS] Sleep hours extracted from 'typical-sleep-hours'")
        log(f"  - Non-null values: {self_report['Sleep'].notna().sum()}")
        log(f"  - Range: [{self_report['Sleep'].min():.1f}, {self_report['Sleep'].max():.1f}]")
    else:
        log("[ERROR] Column 'typical-sleep-hours' not found in dfnonvr.csv")
        log("[CRITICAL] Cannot proceed without real data - stopping")
        sys.exit(1)
    
    # Report summary statistics
    log("\n[SUMMARY] Self-report variables (ALL REAL DATA):")
    for col in ['DASS_Dep', 'DASS_Anx', 'DASS_Str', 'VR_Exp', 'Sleep']:
        if col in self_report.columns:
            mean_val = self_report[col].mean()
            std_val = self_report[col].std()
            min_val = self_report[col].min()
            max_val = self_report[col].max()
            n_missing = self_report[col].isna().sum()
            log(f"  - {col}: M={mean_val:.2f}, SD={std_val:.2f}, Range=[{min_val:.1f}, {max_val:.1f}], Missing={n_missing}")
    
    # Check for any missing data
    total_missing = self_report[['DASS_Dep', 'DASS_Anx', 'DASS_Str', 'VR_Exp', 'Sleep']].isna().sum().sum()
    if total_missing > 0:
        log(f"[WARNING] Total missing values across all self-report variables: {total_missing}")
        log("[INFO] Missing data will be handled in subsequent analysis steps")
    else:
        log("[SUCCESS] No missing data in self-report variables")
    
    # Save output
    output_path = RQ_DIR / "data" / "step03_self_report.csv"
    output_path.parent.mkdir(exist_ok=True)
    self_report.to_csv(output_path, index=False)
    log(f"[SAVE] Saved self-report data to {output_path}")
    log(f"[INFO] Shape: {self_report.shape}")
    
    log("\n[SUCCESS] Step 03 complete - ALL DATA IS REAL from dfnonvr.csv")
    log("[INFO] No simulated/fake data used")
    return 0

if __name__ == "__main__":
    sys.exit(main())