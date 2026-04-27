#!/usr/bin/env python3
"""
Step 04: Extract Ch5 Theta Scores
RQ 7.1.4: Unique REMEMVR variance unexplained by all predictors

Extract overall theta scores from Ch5 5.1.1 results
These represent REMEMVR memory performance (outcome variable)
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
LOG_FILE = RQ_DIR / "logs" / "step04_extract_theta_scores.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    """Log to both console and file."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()

def main():
    """Main execution."""
    log("[START] Step 04: Extract Ch5 theta scores")
    
    # Load Ch5 5.1.1 theta scores
    ch5_path = PROJ_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
    log(f"[LOAD] Reading Ch5 5.1.1 theta scores from {ch5_path}...")
    
    if not ch5_path.exists():
        log(f"[ERROR] Ch5 theta file not found: {ch5_path}")
        return 1
    
    df_theta = pd.read_csv(ch5_path)
    log(f"[INFO] Loaded {len(df_theta)} theta scores")
    
    # Check structure
    log("[CHECK] Ch5 theta data structure:")
    log(f"  - Columns: {df_theta.columns.tolist()}")
    log(f"  - Shape: {df_theta.shape}")
    
    # Ch5 5.1.1 has 400 rows (100 participants × 4 tests)
    # We need to aggregate to get mean theta per participant
    log("[AGGREGATE] Computing mean theta per participant...")
    
    # Group by UID and calculate mean theta
    # Use the actual column name from Ch5
    theta_col = 'Theta_All' if 'Theta_All' in df_theta.columns else 'theta'
    
    theta_summary = df_theta.groupby('UID').agg({
        theta_col: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    theta_summary.columns = ['uid', 'theta_mean', 'theta_std', 'n_tests']
    
    # Ensure uid is string
    theta_summary['uid'] = theta_summary['uid'].astype(str)
    
    log(f"[INFO] Aggregated to {len(theta_summary)} participants")
    
    # Report summary statistics
    log("[SUMMARY] Theta scores (mean across 4 tests):")
    log(f"  - Mean: {theta_summary['theta_mean'].mean():.3f}")
    log(f"  - SD: {theta_summary['theta_mean'].std():.3f}")
    log(f"  - Range: [{theta_summary['theta_mean'].min():.3f}, {theta_summary['theta_mean'].max():.3f}]")
    log(f"  - All participants have {theta_summary['n_tests'].iloc[0]:.0f} tests")
    
    # Check for outliers
    z_scores = np.abs((theta_summary['theta_mean'] - theta_summary['theta_mean'].mean()) / theta_summary['theta_mean'].std())
    n_outliers = (z_scores > 3).sum()
    if n_outliers > 0:
        log(f"[WARNING] {n_outliers} participants with extreme theta scores (|z| > 3)")
    
    # Rename columns for consistency with other steps
    output_df = pd.DataFrame()
    output_df['uid'] = theta_summary['uid']
    output_df['theta'] = theta_summary['theta_mean']
    output_df['theta_sd'] = theta_summary['theta_std']  # Within-person SD across tests
    
    # Save output
    output_path = RQ_DIR / "data" / "step04_theta_scores.csv"
    output_path.parent.mkdir(exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"[SAVE] Saved theta scores to {output_path}")
    log(f"[INFO] Shape: {output_df.shape}")
    
    log("[SUCCESS] Step 04 complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())