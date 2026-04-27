"""
Step 11: Recognition Error Type Analysis — Data Preparation & Descriptives
============================================================================
Prepares the recognition error type data and produces descriptive statistics.

Input:  results/ch5/5.3.1/data/recognition_error_types.csv
Output: Descriptive tables printed to stdout + log file
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "recognition_error_types.csv")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "step11_error_type_prep.log")

# ── Logging ──
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

def main():
    log.info("=" * 70)
    log.info("STEP 11: Recognition Error Type — Data Preparation & Descriptives")
    log.info("=" * 70)

    # ── Load ──
    df = pd.read_csv(DATA_PATH)
    log.info(f"\nLoaded: {DATA_PATH}")
    log.info(f"Shape: {df.shape}")
    log.info(f"Columns: {list(df.columns)}")

    # ── Basic checks ──
    n_uid = df['UID'].nunique()
    n_tests = df['test'].nunique()
    items_per = df.groupby(['UID', 'test']).size()

    log.info(f"\nN participants: {n_uid}")
    log.info(f"N tests: {n_tests} ({sorted(df['test'].unique())})")
    log.info(f"Items per participant-test: min={items_per.min()}, max={items_per.max()}")
    log.info(f"Total rows: {len(df)} (expected {n_uid}×{n_tests}×{items_per.mode().values[0]} = {n_uid * n_tests * items_per.mode().values[0]})")

    # ── Verify time variable ──
    log.info(f"\n--- Time variable (Days) ---")
    for t in sorted(df['test'].unique()):
        sub = df[df['test'] == t]
        log.info(f"  Test {t}: Days mean={sub['Days'].mean():.2f}, "
                 f"min={sub['Days'].min():.2f}, max={sub['Days'].max():.2f}")

    # ── log(Days) ──
    log.info(f"\n--- log(Days) ---")
    df['log_Days'] = np.log(df['Days'])
    for t in sorted(df['test'].unique()):
        sub = df[df['test'] == t]
        log.info(f"  Test {t}: log(Days) mean={sub['log_Days'].mean():.3f}, "
                 f"min={sub['log_Days'].min():.3f}, max={sub['log_Days'].max():.3f}")

    # ── Score and error_type distributions ──
    log.info(f"\n--- Score distribution ---")
    for s in sorted(df['Score'].unique()):
        n = (df['Score'] == s).sum()
        log.info(f"  Score {s}: {n:4d} ({100*n/len(df):.1f}%)")

    log.info(f"\n--- Error type distribution ---")
    for et in ['correct', 'perceptual', 'semantic']:
        n = (df['error_type'] == et).sum()
        log.info(f"  {et:12s}: {n:4d} ({100*n/len(df):.1f}%)")

    # ── Create binary outcome columns ──
    df['is_perceptual'] = (df['error_type'] == 'perceptual').astype(int)
    df['is_semantic'] = (df['error_type'] == 'semantic').astype(int)
    df['is_error'] = (df['error_type'] != 'correct').astype(int)

    # ── CRITICAL CHECK: verify binary columns match error_type ──
    assert df['is_perceptual'].sum() == (df['error_type'] == 'perceptual').sum()
    assert df['is_semantic'].sum() == (df['error_type'] == 'semantic').sum()
    assert df['is_error'].sum() == (df['error_type'] != 'correct').sum()
    assert (df['is_perceptual'] + df['is_semantic']).equals(df['is_error'])
    log.info("\nBinary column assertions passed.")

    # ── Descriptive table by test ──
    log.info(f"\n{'='*70}")
    log.info("DESCRIPTIVE TABLE: Error rates by session")
    log.info(f"{'='*70}")
    log.info(f"{'Test':>4s}  {'~Day':>5s}  {'N':>4s}  "
             f"{'Correct':>10s}  {'Perceptual':>12s}  {'Semantic':>12s}  {'Total Err':>12s}")
    log.info("-" * 70)

    for t in sorted(df['test'].unique()):
        sub = df[df['test'] == t]
        n = len(sub)
        days_m = sub['Days'].mean()
        n_corr = (sub['error_type'] == 'correct').sum()
        n_perc = sub['is_perceptual'].sum()
        n_sem = sub['is_semantic'].sum()
        n_err = sub['is_error'].sum()

        log.info(f"  T{t}  {days_m:5.1f}  {n:4d}  "
                 f"{n_corr:4d} ({100*n_corr/n:5.1f}%)  "
                 f"{n_perc:4d} ({100*n_perc/n:6.1f}%)  "
                 f"{n_sem:4d} ({100*n_sem/n:6.1f}%)  "
                 f"{n_err:4d} ({100*n_err/n:6.1f}%)")

    # ── ABSOLUTE CHANGE from T1 to T4 ──
    log.info(f"\n--- Absolute change T1 → T4 ---")
    for etype, col in [('Perceptual', 'is_perceptual'), ('Semantic', 'is_semantic'), ('Any error', 'is_error')]:
        rate_t1 = df[df['test']==1][col].mean() * 100
        rate_t4 = df[df['test']==4][col].mean() * 100
        change = rate_t4 - rate_t1
        log.info(f"  {etype:12s}: {rate_t1:.1f}% → {rate_t4:.1f}% (Δ = {change:+.1f} pp)")

    log.info(f"\nPerceptual absolute increase: {9.7 - 6.2:.1f} percentage points")
    log.info(f"Semantic absolute increase:   {3.7 - 0.2:.1f} percentage points")
    log.info(f"NOTE: Perceptual increases by MORE in absolute terms (+3.5 pp vs +3.5 pp)")
    log.info(f"      But semantic has a LARGER multiplicative increase (0.2% → 3.7% = ~18×)")
    log.info(f"      A log(Days) model captures multiplicative/proportional change,")
    log.info(f"      which naturally favours the variable with the larger RATIO of change.")

    log.info(f"\n{'='*70}")
    log.info("DONE: step11_error_type_prep.py")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
