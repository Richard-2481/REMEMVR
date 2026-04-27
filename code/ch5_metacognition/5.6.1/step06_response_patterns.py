#!/usr/bin/env python3
"""
Step 06: Confidence Rating Response Pattern Analysis for RQ 6.6.1

Purpose: Document confidence scale usage patterns (Section 1.4 MANDATORY requirement)
         - % participants using full scale (all 5 levels: 0.2, 0.4, 0.6, 0.8, 1.0)
         - % using extremes only (0.2 and 1.0)
         - Mean SD of ratings per participant
         - Flag restricted range (limits calibration validity)

Input: data/step00_item_level.csv (~28,800 item-responses)
Output: data/step06_response_patterns.csv (100 rows, 1 per participant)
Log: logs/step06_response_patterns.log

Author: rq_platinum agent (Section 1.4 compliance)
Date: 2025-12-27
"""

import pandas as pd
import numpy as np
import os

# Paths
DATA_FILE = '/home/etai/projects/REMEMVR/results/ch6/6.6.1/data/step00_item_level.csv'
OUTPUT_FILE = '/home/etai/projects/REMEMVR/results/ch6/6.6.1/data/step06_response_patterns.csv'
LOG_FILE = '/home/etai/projects/REMEMVR/results/ch6/6.6.1/logs/step06_response_patterns.log'

# Redirect output to log
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
log = open(LOG_FILE, 'w')

def logprint(msg):
    """Print to console and log file"""
    print(msg)
    log.write(msg + '\n')

logprint("="*80)
logprint("STEP 06: CONFIDENCE RATING RESPONSE PATTERN ANALYSIS")
logprint("="*80)
logprint(f"Input: {DATA_FILE}")
logprint(f"Output: {OUTPUT_FILE}")
logprint("")

# Load item-level data
logprint("Loading item-level data...")
data = pd.read_csv(DATA_FILE)
logprint(f"✓ Loaded {len(data):,} item-responses from {data['UID'].nunique()} participants")
logprint("")

# Expected confidence levels (5-level Likert scale)
EXPECTED_LEVELS = {0.2, 0.4, 0.6, 0.8, 1.0}
EXTREME_LEVELS = {0.2, 1.0}

# Initialize results list
results = []

logprint("Analyzing response patterns per participant...")
logprint("-" * 80)

for uid in sorted(data['UID'].unique()):
    # Filter to this participant
    uid_data = data[data['UID'] == uid]

    # Extract confidence ratings (drop NaN)
    ratings = uid_data['confidence'].dropna()

    if len(ratings) == 0:
        logprint(f"WARNING: {uid} has no valid confidence ratings (all NaN)")
        continue

    # Unique levels used
    levels_used = set(ratings.unique())
    n_levels_used = len(levels_used)

    # Classification
    full_scale_user = (levels_used == EXPECTED_LEVELS)  # Uses all 5 levels
    extremes_only = levels_used.issubset(EXTREME_LEVELS)  # Only uses 0.2 and 1.0

    # Rating variability (SD)
    rating_sd = ratings.std()

    # Store results
    results.append({
        'UID': uid,
        'n_ratings': len(ratings),
        'n_levels_used': n_levels_used,
        'full_scale_user': full_scale_user,
        'extremes_only': extremes_only,
        'rating_mean': ratings.mean(),
        'rating_sd': rating_sd,
        'restricted_range': (rating_sd < 0.2)  # Flag if SD < 0.2 (very limited variance)
    })

logprint(f"✓ Analyzed {len(results)} participants")
logprint("")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Aggregate statistics
n_participants = len(results_df)
pct_full_scale = (results_df['full_scale_user'].sum() / n_participants) * 100
pct_extremes = (results_df['extremes_only'].sum() / n_participants) * 100
pct_restricted = (results_df['restricted_range'].sum() / n_participants) * 100
mean_sd = results_df['rating_sd'].mean()
median_levels = results_df['n_levels_used'].median()

logprint("="*80)
logprint("AGGREGATE RESPONSE PATTERN STATISTICS")
logprint("="*80)
logprint(f"Total Participants: {n_participants}")
logprint(f"Full-scale users (all 5 levels): {results_df['full_scale_user'].sum()} ({pct_full_scale:.1f}%)")
logprint(f"Extremes-only users (0.2 and 1.0): {results_df['extremes_only'].sum()} ({pct_extremes:.1f}%)")
logprint(f"Restricted range (SD < 0.2): {results_df['restricted_range'].sum()} ({pct_restricted:.1f}%)")
logprint(f"Mean rating SD: {mean_sd:.3f}")
logprint(f"Median levels used: {median_levels:.0f}")
logprint("")

# Interpretation
logprint("="*80)
logprint("INTERPRETATION")
logprint("="*80)

if pct_full_scale > 70:
    logprint("✓ GOOD: Majority of participants use full confidence scale")
    logprint("  Interpretation: Confidence ratings reflect nuanced metacognitive judgments")
elif pct_full_scale > 40:
    logprint("⚠ MODERATE: Partial full-scale usage")
    logprint("  Interpretation: Some participants exhibit restricted response patterns")
else:
    logprint("✗ CONCERN: Low full-scale usage")
    logprint("  Interpretation: Confidence scale may not capture fine-grained monitoring")

if pct_extremes > 10:
    logprint(f"⚠ WARNING: {pct_extremes:.1f}% participants use extremes only")
    logprint("  Interpretation: Binary all-or-nothing confidence (not graded)")
    logprint("  Impact: HCE threshold (>= 0.75) may conflate true high confidence with response style")
else:
    logprint(f"✓ GOOD: Low extremes-only usage ({pct_extremes:.1f}%)")

if mean_sd < 0.3:
    logprint(f"⚠ WARNING: Mean SD ({mean_sd:.3f}) suggests restricted variability")
    logprint("  Interpretation: Participants not differentiating confidence levels well")
else:
    logprint(f"✓ GOOD: Adequate rating variability (Mean SD = {mean_sd:.3f})")

logprint("")

# Save results
logprint("="*80)
logprint("SAVING RESULTS")
logprint("="*80)

results_df.to_csv(OUTPUT_FILE, index=False)
logprint(f"✓ Results saved: {OUTPUT_FILE}")
logprint(f"  Rows: {len(results_df)}")
logprint(f"  Columns: {len(results_df.columns)}")
logprint("")

# Validation
logprint("="*80)
logprint("VALIDATION")
logprint("="*80)

assert len(results_df) == 100, f"Expected 100 participants, found {len(results_df)}"
assert all(results_df['n_ratings'] > 0), "Some participants have zero ratings"
assert all(results_df['n_levels_used'] >= 1), "Some participants have zero levels used"
assert all(results_df['n_levels_used'] <= 5), "Some participants use >5 levels (impossible)"
assert all(results_df['rating_sd'] >= 0), "Negative SD detected"

logprint("✓ All validation checks passed")
logprint("")

logprint("="*80)
logprint("STEP 06 COMPLETE - Response patterns documented per Section 1.4")
logprint("="*80)

log.close()
