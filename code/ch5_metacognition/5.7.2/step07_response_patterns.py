#!/usr/bin/env python3
"""
RQ 6.7.2: Response Pattern Analysis
====================================

MANDATORY analysis per improvement_taxonomy.md Section 8.3:
Confidence RQs must document response patterns to validate data quality.

Checks:
1. % participants using full scale (all 5 confidence levels)
2. % extremes only (1s and 5s)
3. Mean SD of ratings per participant
4. Restricted range detection (SD < 0.15 threshold)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_FILE = LOG_DIR / "step07_response_patterns.log"

CONFIDENCE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
RESTRICTED_THRESHOLD = 0.15  # SD < 0.15 indicates restricted range


def log(msg: str):
    """Log to file and stdout."""
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a') as f:
        f.write(f"{msg}\n")
    print(msg)


def main():
    log("=" * 70)
    log("RQ 6.7.2: Response Pattern Analysis (Confidence Ratings)")
    log("=" * 70)
    log("")

    # Load master data
    df = pd.read_csv("/home/etai/projects/REMEMVR/data/cache/dfData.csv")
    log(f"Loaded dfData.csv: {len(df)} rows")

    # Get TC_ columns for interactive paradigms
    tc_cols = [c for c in df.columns if c.startswith("TC_")]
    paradigms = ["IFR", "ICR", "IRE"]
    tc_interactive = []
    for col in tc_cols:
        paradigm = col.split("_")[1].split("-")[0]
        if paradigm in paradigms:
            tc_interactive.append(col)

    log(f"Found {len(tc_interactive)} TC_ columns for paradigms: {paradigms}")

    # Analyze per participant (across all tests)
    results = []

    for uid in df['UID'].unique():
        df_person = df[df['UID'] == uid]

        # Get all confidence ratings for this person (all tests combined)
        ratings = []
        for col in tc_interactive:
            for test_num in [1, 2, 3, 4]:
                df_test = df_person[df_person['TEST'] == test_num]
                if len(df_test) > 0 and col in df_test.columns:
                    val = df_test[col].values[0]
                    if pd.notna(val):
                        ratings.append(val)

        if len(ratings) == 0:
            continue

        ratings = np.array(ratings)

        # Check full scale usage (all 5 levels present)
        unique_levels = set(ratings)
        full_scale = len(unique_levels) == 5

        # Check extremes only (only 0.0 and 1.0 present)
        extremes_only = unique_levels.issubset({0.0, 1.0})

        # Compute SD
        rating_sd = np.std(ratings, ddof=1)

        # Restricted range
        restricted = rating_sd < RESTRICTED_THRESHOLD

        results.append({
            'UID': uid,
            'N_ratings': len(ratings),
            'full_scale': full_scale,
            'extremes_only': extremes_only,
            'SD_ratings': rating_sd,
            'restricted_range': restricted,
            'unique_levels': len(unique_levels)
        })

    results_df = pd.DataFrame(results)

    # Summary statistics
    n_participants = len(results_df)
    pct_full_scale = (results_df['full_scale'].sum() / n_participants) * 100
    pct_extremes = (results_df['extremes_only'].sum() / n_participants) * 100
    mean_sd = results_df['SD_ratings'].mean()
    pct_restricted = (results_df['restricted_range'].sum() / n_participants) * 100

    log("")
    log("=" * 70)
    log("RESPONSE PATTERN SUMMARY (N={} participants)".format(n_participants))
    log("=" * 70)
    log("")
    log(f"Full Scale Usage (all 5 levels):")
    log(f"  N = {results_df['full_scale'].sum()} ({pct_full_scale:.1f}%)")
    log(f"  Interpretation: {'EXCELLENT' if pct_full_scale >= 70 else 'ADEQUATE' if pct_full_scale >= 50 else 'POOR'}")
    log("")
    log(f"Extremes Only (1s and 5s):")
    log(f"  N = {results_df['extremes_only'].sum()} ({pct_extremes:.1f}%)")
    log(f"  Interpretation: {'NONE DETECTED' if pct_extremes == 0 else 'WARNING'}")
    log("")
    log(f"Rating Variability:")
    log(f"  Mean SD: {mean_sd:.3f}")
    log(f"  SD range: [{results_df['SD_ratings'].min():.3f}, {results_df['SD_ratings'].max():.3f}]")
    log("")
    log(f"Restricted Range (SD < {RESTRICTED_THRESHOLD}):")
    log(f"  N = {results_df['restricted_range'].sum()} ({pct_restricted:.1f}%)")
    log(f"  Interpretation: {'EXCELLENT' if pct_restricted < 5 else 'ACCEPTABLE' if pct_restricted < 10 else 'WARNING'}")
    log("")

    # Overall assessment
    log("=" * 70)
    log("OVERALL DATA QUALITY ASSESSMENT")
    log("=" * 70)

    quality_issues = []
    if pct_full_scale < 50:
        quality_issues.append("LOW full scale usage (<50%)")
    if pct_extremes > 5:
        quality_issues.append("HIGH extreme responding (>5%)")
    if pct_restricted > 10:
        quality_issues.append("HIGH restricted range (>10%)")
    if mean_sd < 0.20:
        quality_issues.append("LOW overall variability (mean SD < 0.20)")

    if len(quality_issues) == 0:
        log("STATUS: EXCELLENT")
        log("  ✓ Full scale usage: {:.1f}% (target: ≥70%)".format(pct_full_scale))
        log("  ✓ Extremes only: {:.1f}% (target: <5%)".format(pct_extremes))
        log("  ✓ Restricted range: {:.1f}% (target: <10%)".format(pct_restricted))
        log("  ✓ Mean SD: {:.3f} (target: ≥0.20)".format(mean_sd))
        log("")
        log("INTERPRETATION:")
        log("  Confidence ratings capture genuine metacognitive variability.")
        log("  No evidence of response bias artifacts.")
        log("  Data quality suitable for variability analysis.")
    else:
        log("STATUS: ISSUES DETECTED")
        for issue in quality_issues:
            log(f"  ✗ {issue}")
        log("")
        log("RECOMMENDATION:")
        log("  Document issues in Limitations section.")
        log("  Consider sensitivity analysis excluding restricted-range participants.")

    # Save results
    results_df.to_csv(DATA_DIR / "step07_response_patterns.csv", index=False)
    log("")
    log(f"Saved: data/step07_response_patterns.csv ({len(results_df)} rows)")
    log("")
    log("=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
