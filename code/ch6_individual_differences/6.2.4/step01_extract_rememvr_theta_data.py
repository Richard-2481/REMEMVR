#!/usr/bin/env python3
"""
Step 1: Extract and Prepare REMEMVR Data
RQ 7.2.4 - VR Scaffolding Validation

Purpose: Extract REMEMVR theta_all scores from Ch5 5.1.1 outputs and prepare for correlation analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_extract_rememvr.log"

def log(msg):
    """Log to both file and stdout"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def main():
    log("=" * 60)
    log("Step 1: Extract REMEMVR Theta Data")
    log("=" * 60)
    
    # Read Ch5 5.1.1 theta scores
    ch5_path = Path("results/ch5/5.1.1/data/step03_theta_scores.csv")
    log(f"Reading Ch5 theta scores from: {ch5_path}")
    
    df_theta = pd.read_csv(ch5_path)
    log(f"Loaded {len(df_theta)} rows with columns: {list(df_theta.columns)}")
    
    # Rename Theta_All to theta_all for consistency
    if 'Theta_All' in df_theta.columns:
        df_theta = df_theta.rename(columns={'Theta_All': 'theta_all'})
        log("Renamed Theta_All to theta_all")
    
    # Aggregate theta scores across tests (mean per participant)
    log("Aggregating theta scores by participant (mean across 4 tests)")
    df_rememvr = df_theta.groupby('UID').agg({'theta_all': 'mean'}).reset_index()
    
    # Ensure UID is string type
    df_rememvr['UID'] = df_rememvr['UID'].astype(str)
    
    # Standardize theta_all scores (z-score transformation)
    theta_mean = df_rememvr['theta_all'].mean()
    theta_std = df_rememvr['theta_all'].std()
    df_rememvr['theta_all_z'] = (df_rememvr['theta_all'] - theta_mean) / theta_std
    
    # Quality checks
    n_participants = len(df_rememvr)
    extreme_count = np.sum(np.abs(df_rememvr['theta_all_z']) > 2.0)
    extreme_pct = (extreme_count / n_participants) * 100
    
    log(f"\nRESULTS:")
    log(f"REMEMVR theta extracted: {n_participants} participants")
    log(f"Theta descriptives: mean={theta_mean:.3f}, sd={theta_std:.3f}")
    log(f"Range: min={df_rememvr['theta_all'].min():.3f}, max={df_rememvr['theta_all'].max():.3f}")
    log(f"Extreme values (|z|>2): {extreme_count} ({extreme_pct:.1f}%)")
    
    # Flag if too many extreme values
    if extreme_pct > 5:
        log(f"WARNING: Range restriction concern - {extreme_pct:.1f}% of values are extreme")
    else:
        log("Range restriction check: PASS")
    
    # Verify standardization
    z_mean = df_rememvr['theta_all_z'].mean()
    z_std = df_rememvr['theta_all_z'].std()
    log(f"\nStandardization verification: z_mean={z_mean:.6f}, z_std={z_std:.6f}")
    
    # Save output
    output_path = RQ_DIR / "data" / "step01_rememvr_theta_data.csv"
    df_rememvr.to_csv(output_path, index=False)
    log(f"\nSaved to: {output_path}")
    log(f"Output shape: {df_rememvr.shape}")
    
    # Final verification
    log("\nFinal data check:")
    log(f"  - Participants: {n_participants}")
    log(f"  - Columns: {list(df_rememvr.columns)}")
    log(f"  - No missing values: {df_rememvr.isnull().sum().sum() == 0}")
    log(f"  - All finite values: {np.isfinite(df_rememvr.select_dtypes(include=[np.number])).all().all()}")
    
    log("\nStep 1 completed successfully")

if __name__ == "__main__":
    main()