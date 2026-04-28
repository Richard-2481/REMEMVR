#!/usr/bin/env python3
"""
Step ID: step11
Step Name: accuracy_merge
RQ: results/ch7/7.3.1

PURPOSE:
Merge accuracy theta scores (from RQ 7.1.1) with the same predictor set used
for the confidence model (step04), producing a directly comparable analysis dataset.

This enables apples-to-apples comparison between accuracy and confidence prediction
using identical predictors (age, sex, education, RAVLT_T, BVMT_T, RPM_T) and N=100.

EXPECTED INPUTS:
  - results/ch7/7.1.1/data/step02_theta_means.csv  (UID, theta_mean)
  - results/ch7/7.3.1/data/step02_cognitive_tests.csv  (UID, RAVLT_T, BVMT_T, RPM_T, age, sex, education)

EXPECTED OUTPUTS:
  - results/ch7/7.3.1/data/step11_accuracy_analysis_dataset.csv
    Columns: UID, accuracy_theta, RAVLT_T, BVMT_T, RPM_T, age, sex, education
    Expected rows: 100
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step11_accuracy_merge.log"

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        # Clear log
        LOG_FILE.write_text("")

        log("Step 11: Merge accuracy theta with predictors")

        # Load accuracy theta from RQ 7.1.1
        theta_path = PROJECT_ROOT / "results" / "ch7" / "7.1.1" / "data" / "step02_theta_means.csv"
        theta_df = pd.read_csv(theta_path)
        log(f"{theta_path.name}: {len(theta_df)} rows, columns={list(theta_df.columns)}")
        theta_df = theta_df.rename(columns={'theta_mean': 'accuracy_theta'})

        # Load predictors from RQ 7.3.1 step02
        pred_path = RQ_DIR / "data" / "step02_cognitive_tests.csv"
        pred_df = pd.read_csv(pred_path)
        log(f"{pred_path.name}: {len(pred_df)} rows, columns={list(pred_df.columns)}")

        # Inner merge on UID
        merged = pd.merge(theta_df, pred_df, on='UID', how='inner')
        log(f"N={len(merged)} participants (inner join on UID)")

        # Verify completeness
        n_missing = merged.isna().sum().sum()
        log(f"Missing values: {n_missing}")

        if len(merged) != 100:
            log(f"Expected N=100, got N={len(merged)}")

        # Save
        out_path = RQ_DIR / "data" / "step11_accuracy_analysis_dataset.csv"
        merged.to_csv(out_path, index=False)
        log(f"{out_path.name} ({len(merged)} rows, {len(merged.columns)} cols)")
        log(f"{list(merged.columns)}")
        log(f"accuracy_theta: mean={merged['accuracy_theta'].mean():.4f}, sd={merged['accuracy_theta'].std():.4f}")

        log("Step 11: Accuracy merge complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
