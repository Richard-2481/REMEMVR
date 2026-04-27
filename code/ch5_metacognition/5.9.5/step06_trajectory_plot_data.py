#!/usr/bin/env python3
"""Step 6: Create trajectory plot data - aggregate by measure/location/time"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_trajectories.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

try:
    log("[START] Step 6: Trajectory Plot Data")
    
    # Load merged data
    df = pd.read_csv(RQ_DIR / "data" / "step03_merged_data_long.csv")
    log(f"[LOADED] {len(df)} rows")
    
    # Aggregate by measure, location, TSVR_hours
    df_plot = df.groupby(['measure', 'location', 'TSVR_hours'])['theta'].agg([
        ('mean_theta', 'mean'),
        ('se_mean', 'sem'),
        ('n', 'count')
    ]).reset_index()
    
    # Compute 95% CI
    df_plot['ci_lower'] = df_plot['mean_theta'] - 1.96 * df_plot['se_mean']
    df_plot['ci_upper'] = df_plot['mean_theta'] + 1.96 * df_plot['se_mean']
    
    log(f"[AGGREGATED] {len(df_plot)} rows (should be 16: 2x2x4)")
    
    # Save
    output_path = RQ_DIR / "data" / "step06_trajectory_plot_data.csv"
    df_plot.to_csv(output_path, index=False, encoding='utf-8')
    log(f"[SAVE] {output_path.name}")
    
    log("[SUCCESS] Step 6 complete")
    sys.exit(0)
except Exception as e:
    log(f"[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
