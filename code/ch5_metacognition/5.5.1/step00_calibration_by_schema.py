#!/usr/bin/env python3
"""
RQ 6.5.1: Create Calibration Input by Schema Congruence
========================================================

Merges accuracy theta (pass2) and confidence theta into long format
for SEM calibration computation.

Input:
  - step03_pass2_theta.csv (accuracy theta, wide: composite_ID × 3 schema levels)
  - step03_theta_confidence.csv (confidence theta, wide: composite_ID × 3 schema levels)
  - step00_tsvr_mapping.csv (composite_ID → UID, test, TSVR_hours)

Output:
  - step00_calibration_by_schema.csv (1200 rows: 100 UID × 4 tests × 3 schema levels)

Author: Claude Code
Date: 2026-04-07
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"

# Load data
df_acc = pd.read_csv(DATA_DIR / "step03_pass2_theta.csv")
df_conf = pd.read_csv(DATA_DIR / "step03_theta_confidence.csv")
df_tsvr = pd.read_csv(DATA_DIR / "step00_tsvr_mapping.csv")

print(f"Accuracy theta: {df_acc.shape}")
print(f"Confidence theta: {df_conf.shape}")
print(f"TSVR mapping: {df_tsvr.shape}")

SCHEMA_LEVELS = ['Common', 'Congruent', 'Incongruent']

# Melt accuracy to long format
df_acc_long = df_acc.melt(
    id_vars='composite_ID',
    value_vars=[f'theta_{s}' for s in SCHEMA_LEVELS],
    var_name='Schema',
    value_name='theta_accuracy'
)
df_acc_long['Schema'] = df_acc_long['Schema'].str.replace('theta_', '')

# Melt confidence to long format
df_conf_long = df_conf.melt(
    id_vars='composite_ID',
    value_vars=[f'theta_{s}' for s in SCHEMA_LEVELS],
    var_name='Schema',
    value_name='theta_confidence'
)
df_conf_long['Schema'] = df_conf_long['Schema'].str.replace('theta_', '')

# Merge accuracy + confidence
df = df_acc_long.merge(df_conf_long, on=['composite_ID', 'Schema'])

# Merge TSVR mapping
df = df.merge(df_tsvr, on='composite_ID')

# Rename columns to match standard format
df = df.rename(columns={'test': 'TEST'})

# Z-standardize WITHIN each schema level (matching 6.3.2 pattern)
for schema in SCHEMA_LEVELS:
    mask = df['Schema'] == schema
    for col, zcol in [('theta_accuracy', 'theta_accuracy_z'),
                      ('theta_confidence', 'theta_confidence_z')]:
        vals = df.loc[mask, col]
        df.loc[mask, zcol] = (vals - vals.mean()) / vals.std()

# Simple calibration (z-difference, pre-SEM)
df['calibration'] = df['theta_confidence_z'] - df['theta_accuracy_z']

# Select and order columns
df_out = df[['UID', 'TEST', 'Schema', 'TSVR_hours',
             'theta_accuracy', 'theta_confidence',
             'theta_accuracy_z', 'theta_confidence_z',
             'calibration']].copy()

df_out = df_out.sort_values(['UID', 'Schema', 'TEST']).reset_index(drop=True)

print(f"\nOutput shape: {df_out.shape}")
print(f"Schema levels: {sorted(df_out['Schema'].unique())}")
print(f"UIDs: {df_out['UID'].nunique()}")
print(f"Tests: {sorted(df_out['TEST'].unique())}")

# Validate
assert len(df_out) == 1200, f"Expected 1200 rows, got {len(df_out)}"
assert df_out['theta_accuracy'].isna().sum() == 0, "Missing accuracy values"
assert df_out['theta_confidence'].isna().sum() == 0, "Missing confidence values"

# Check z-standardization
for schema in SCHEMA_LEVELS:
    mask = df_out['Schema'] == schema
    acc_mean = df_out.loc[mask, 'theta_accuracy_z'].mean()
    acc_std = df_out.loc[mask, 'theta_accuracy_z'].std()
    conf_mean = df_out.loc[mask, 'theta_confidence_z'].mean()
    conf_std = df_out.loc[mask, 'theta_confidence_z'].std()
    print(f"{schema}: acc_z mean={acc_mean:.4f} std={acc_std:.4f}, conf_z mean={conf_mean:.4f} std={conf_std:.4f}")

output_file = DATA_DIR / "step00_calibration_by_schema.csv"
df_out.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")
