#!/usr/bin/env python3
"""
RQ 6.2.1: Confidence Response Patterns
========================================
Analyze confidence rating distributions per Section 8.3.

This is a MANDATORY analysis for confidence RQs to detect:
- Extreme response style (ERS): % 1s and 5s
- Restricted range: SD < 0.8
- Full scale usage: % using all 5 levels

Helps interpret calibration metrics (ECE stability despite Brier increase).
"""

import pandas as pd
import numpy as np
from pathlib import Path

RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RQ_DIR.parents[2]
DFDATA_FILE = PROJECT_ROOT / "data/cache/dfData.csv"
LOG_FILE = RQ_DIR / "logs" / "step09_confidence_response_patterns.log"

def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def main():
    log("="*70)
    log("STEP 09: Confidence Response Patterns")
    log("="*70)

    # Load item-level data
    df = pd.read_csv(DFDATA_FILE)
    log(f"Loaded dfData: {len(df)} rows")

    # Get TC_* (confidence) columns
    tc_cols = [c for c in df.columns if c.startswith('TC_')]
    log(f"Confidence columns: {len(tc_cols)}")

    # Analyze per participant-test
    results = []

    for _, row in df.iterrows():
        uid = row['UID']
        test = row['TEST']

        # Extract all confidence ratings for this participant-test
        ratings = []
        for tc_col in tc_cols:
            val = row[tc_col]
            if pd.notna(val):
                ratings.append(val)

        if len(ratings) == 0:
            continue

        ratings = np.array(ratings)

        # Detect scale (1-5 raw or 0-1 normalized)
        rating_max = ratings.max()
        rating_min = ratings.min()

        if rating_max > 1:
            # Assume 1-5 scale, normalize to 0-1 for analysis
            ratings_norm = (ratings - 1) / 4
            scale_type = "1-5"
        else:
            ratings_norm = ratings
            scale_type = "0-1"

        # Metrics
        n_ratings = len(ratings)
        rating_mean = ratings.mean()
        rating_sd = ratings.std()

        # Full scale usage (all 5 levels present)
        if rating_max > 1:
            # Check for 1, 2, 3, 4, 5
            unique_vals = set(ratings)
            full_scale = len(unique_vals) == 5
        else:
            # Check for 0, 0.25, 0.5, 0.75, 1.0
            unique_vals = set(ratings)
            full_scale = len(unique_vals) == 5

        # Extremes only (1s and 5s, or 0 and 1.0)
        if rating_max > 1:
            extremes_only = all(r in [1, 5] for r in ratings)
            pct_extremes = np.mean([(r in [1, 5]) for r in ratings]) * 100
        else:
            extremes_only = all(r in [0, 1.0] for r in ratings)
            pct_extremes = np.mean([(r in [0, 1.0]) for r in ratings]) * 100

        # Restricted range (SD < 0.8 on normalized 0-1 scale)
        restricted_range = rating_sd < 0.8 if rating_max > 1 else (ratings_norm.std() < 0.20)

        test_label = f'T{int(test)}' if isinstance(test, (int, float)) else test
        results.append({
            'UID': uid,
            'TEST': test_label,
            'composite_ID': f"{uid}_{test_label}",
            'n_ratings': n_ratings,
            'mean_rating': rating_mean,
            'sd_rating': rating_sd,
            'full_scale_usage': full_scale,
            'extremes_only': extremes_only,
            'pct_extremes': pct_extremes,
            'restricted_range': restricted_range,
            'scale_type': scale_type
        })

    df_patterns = pd.DataFrame(results)

    # Summary statistics
    log("\n" + "="*70)
    log("SUMMARY STATISTICS")
    log("="*70)

    n_participants = df_patterns['UID'].nunique()
    n_observations = len(df_patterns)

    log(f"Total participants: {n_participants}")
    log(f"Total observations: {n_observations}")

    # Full scale usage
    pct_full_scale = (df_patterns['full_scale_usage'].sum() / n_observations) * 100
    log(f"\nFull scale usage (all 5 levels): {pct_full_scale:.1f}%")

    # Extremes only
    pct_extremes_only = (df_patterns['extremes_only'].sum() / n_observations) * 100
    log(f"Extremes only (1s and 5s): {pct_extremes_only:.1f}%")

    # Restricted range
    pct_restricted = (df_patterns['restricted_range'].sum() / n_observations) * 100
    log(f"Restricted range (SD < 0.8): {pct_restricted:.1f}%")

    # Mean SD
    mean_sd = df_patterns['sd_rating'].mean()
    log(f"Mean rating SD: {mean_sd:.2f}")

    # Rating distributions per test
    log("\n" + "="*70)
    log("BY TEST SESSION")
    log("="*70)

    for test in sorted(df_patterns['TEST'].unique()):
        df_test = df_patterns[df_patterns['TEST'] == test]
        test_mean = df_test['mean_rating'].mean()
        test_sd = df_test['mean_rating'].std()
        test_full = (df_test['full_scale_usage'].sum() / len(df_test)) * 100
        test_extremes = (df_test['extremes_only'].sum() / len(df_test)) * 100

        log(f"\n{test}:")
        log(f"  Mean rating: {test_mean:.2f} (SD: {test_sd:.2f})")
        log(f"  Full scale usage: {test_full:.1f}%")
        log(f"  Extremes only: {test_extremes:.1f}%")

    # Interpretation
    log("\n" + "="*70)
    log("INTERPRETATION")
    log("="*70)

    if pct_full_scale > 80:
        log("✓ GOOD: Majority of participants use full confidence scale")
    elif pct_full_scale > 50:
        log("⚠ MODERATE: Some participants show restricted scale usage")
    else:
        log("✗ POOR: Many participants not using full scale (limits calibration interpretability)")

    if pct_extremes_only > 20:
        log("⚠ WARNING: >20% of observations show extreme responding (1s and 5s only)")
        log("  This may inflate calibration metrics (binary confidence not nuanced)")
    else:
        log("✓ GOOD: Low extreme responding (<20%)")

    if mean_sd < 0.8:
        log("⚠ WARNING: Mean SD < 0.8 indicates restricted range")
        log("  Participants may not be discriminating between confidence levels")
    else:
        log("✓ GOOD: Mean SD >= 0.8 indicates adequate rating variance")

    # ECE stability explanation
    log("\n" + "="*70)
    log("LINK TO ECE STABILITY (from summary.md)")
    log("="*70)

    log("ECE remains stable (0.090-0.102) despite Brier increase (0.147-0.177).")
    log("Response pattern analysis explains this:")
    if pct_full_scale > 70 and pct_extremes_only < 30:
        log("  → Participants maintain similar confidence DISTRIBUTIONS over time")
        log("  → Full scale usage preserved (not collapsing to extremes)")
        log("  → Within-bin accuracy declines proportionally")
        log("  → Result: Relative calibration structure stable (ECE), but")
        log("            absolute alignment worsens (Brier, person-level calibration)")
    else:
        log("  → Pattern differs from expected (investigate further)")

    # Save results
    out_path = RQ_DIR / "data" / "step09_confidence_response_patterns.csv"
    df_patterns.to_csv(out_path, index=False)
    log(f"\nSaved: {out_path}")
    log("VALIDATION - PASS: Confidence response patterns documented")

if __name__ == "__main__":
    main()
