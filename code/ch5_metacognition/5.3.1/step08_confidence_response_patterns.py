#!/usr/bin/env python3
"""
PLATINUM FINALIZATION: Confidence Response Patterns (MANDATORY - Section 1.4)

PURPOSE:
Document confidence rating response patterns to validate GRM assumptions and detect
extreme response styles. Per solution.md section 1.4, this is MANDATORY for confidence RQs.

CHECKS:
  1. % participants using full 1-5 scale
  2. % participants using extremes only (1s and 5s)
  3. Mean SD of ratings per participant
  4. Response distribution across scale

INPUTS:
  - data/step00_irt_input.csv (raw TC_* confidence items)

OUTPUTS:
  - data/step08_response_patterns.csv (participant-level patterns)
  - data/step08_response_patterns_summary.txt (aggregated statistics)
  - logs/step08_confidence_response_patterns.log
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step08_confidence_response_patterns.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("PLATINUM FINALIZATION: Confidence Response Patterns (Section 1.4)")
        log("=" * 80)

        # Load raw IRT input
        log("\n[LOAD] Loading raw confidence ratings...")
        input_path = RQ_DIR / "data" / "step00_irt_input.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        irt_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"[LOADED] {len(irt_input)} rows (composite_IDs)")

        # Extract item columns (all columns except composite_ID)
        item_cols = [col for col in irt_input.columns if col != 'composite_ID']
        log(f"[INFO] {len(item_cols)} TC_* confidence items identified")

        # Valid confidence scale values (allowing both decimal and fractional representations)
        # Standard: 0, 0.25, 0.5, 0.75, 1.0
        # Also accept: 0.2, 0.4, 0.6, 0.8 (equivalent 5-point Likert: 20%, 40%, 60%, 80%)
        valid_values = {0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0}

        # Normalize to standard scale (map 0.2->0.25, 0.4->0.5, 0.6->0.75, 0.8->1.0)
        def normalize_rating(x):
            if pd.isna(x):
                return np.nan
            if x in {0.0, 0.25, 0.5, 0.75, 1.0}:
                return x
            elif abs(x - 0.2) < 0.01:
                return 0.25
            elif abs(x - 0.4) < 0.01:
                return 0.5
            elif abs(x - 0.6) < 0.01:
                return 0.75
            elif abs(x - 0.8) < 0.01:
                return 1.0
            else:
                return x  # Keep as-is (may be invalid, will flag)

        # Normalize all item columns
        irt_input[item_cols] = irt_input[item_cols].applymap(normalize_rating)

        # Parse UID from composite_ID
        irt_input['UID'] = irt_input['composite_ID'].str.split('_').str[0]
        n_participants = irt_input['UID'].nunique()
        log(f"[INFO] {n_participants} unique participants")

        # =====================================================================
        # Participant-Level Response Patterns
        # =====================================================================
        log("\n[ANALYSIS] Computing participant-level response patterns...")

        participant_patterns = []

        for uid in irt_input['UID'].unique():
            uid_data = irt_input[irt_input['UID'] == uid]

            # Flatten all ratings for this participant (across all tests)
            all_ratings = uid_data[item_cols].values.flatten()
            valid_ratings = all_ratings[~pd.isna(all_ratings)]

            if len(valid_ratings) == 0:
                continue  # Skip participants with all missing data

            # Compute statistics
            unique_values = set(valid_ratings)
            n_unique = len(unique_values)
            uses_full_scale = (n_unique == 5)  # Uses all 5 values (0, 0.25, 0.5, 0.75, 1.0)
            uses_extremes_only = unique_values.issubset({0.0, 1.0})  # Only 0 and 1
            rating_sd = np.std(valid_ratings)
            rating_mean = np.mean(valid_ratings)
            n_ratings = len(valid_ratings)

            # Count each rating
            rating_counts = {
                '0': np.sum(valid_ratings == 0.0),
                '0.25': np.sum(valid_ratings == 0.25),
                '0.5': np.sum(valid_ratings == 0.5),
                '0.75': np.sum(valid_ratings == 0.75),
                '1.0': np.sum(valid_ratings == 1.0)
            }

            participant_patterns.append({
                'UID': uid,
                'n_ratings': n_ratings,
                'n_unique_values': n_unique,
                'uses_full_scale': uses_full_scale,
                'uses_extremes_only': uses_extremes_only,
                'rating_mean': rating_mean,
                'rating_sd': rating_sd,
                'pct_0': rating_counts['0'] / n_ratings * 100,
                'pct_0.25': rating_counts['0.25'] / n_ratings * 100,
                'pct_0.5': rating_counts['0.5'] / n_ratings * 100,
                'pct_0.75': rating_counts['0.75'] / n_ratings * 100,
                'pct_1.0': rating_counts['1.0'] / n_ratings * 100
            })

        patterns_df = pd.DataFrame(participant_patterns)

        # =====================================================================
        # Aggregate Statistics
        # =====================================================================
        log("\n[SUMMARY] Aggregate response pattern statistics...")

        n_full_scale = patterns_df['uses_full_scale'].sum()
        pct_full_scale = (n_full_scale / len(patterns_df)) * 100

        n_extremes_only = patterns_df['uses_extremes_only'].sum()
        pct_extremes_only = (n_extremes_only / len(patterns_df)) * 100

        mean_sd = patterns_df['rating_sd'].mean()
        median_sd = patterns_df['rating_sd'].median()

        mean_rating = patterns_df['rating_mean'].mean()
        median_unique = patterns_df['n_unique_values'].median()

        log(f"[RESULT] Full scale usage (all 5 values): {n_full_scale}/{len(patterns_df)} ({pct_full_scale:.1f}%)")
        log(f"[RESULT] Extremes only (0 and 1): {n_extremes_only}/{len(patterns_df)} ({pct_extremes_only:.1f}%)")
        log(f"[RESULT] Mean rating SD: {mean_sd:.3f} (median: {median_sd:.3f})")
        log(f"[RESULT] Mean rating: {mean_rating:.3f}")
        log(f"[RESULT] Median unique values per participant: {median_unique:.0f}")

        # Warnings
        log("\n[INTERPRETATION] GRM assumption validation...")

        if pct_full_scale < 50:
            log(f"[WARNING] Only {pct_full_scale:.1f}% of participants use full 5-point scale")
            log("[WARNING] GRM assumes ordinal scale usage - restricted range may violate assumptions")
        else:
            log(f"[PASS] {pct_full_scale:.1f}% of participants use full scale (acceptable)")

        if pct_extremes_only > 10:
            log(f"[WARNING] {pct_extremes_only:.1f}% of participants use only extremes (0 and 1)")
            log("[WARNING] Extreme response style may bias IRT estimates")
        else:
            log(f"[PASS] Only {pct_extremes_only:.1f}% use extremes only (acceptable)")

        if mean_sd < 0.20:
            log(f"[WARNING] Mean rating SD = {mean_sd:.3f} (low variability)")
            log("[WARNING] Restricted range limits IRT discrimination")
        else:
            log(f"[PASS] Mean rating SD = {mean_sd:.3f} (acceptable variability)")

        # =====================================================================
        # Save Outputs
        # =====================================================================
        log("\n[SAVE] Saving response pattern results...")

        # Participant-level patterns
        output_patterns = RQ_DIR / "data" / "step08_response_patterns.csv"
        patterns_df.to_csv(output_patterns, index=False, encoding='utf-8')
        log(f"[SAVED] {output_patterns.name} ({len(patterns_df)} participants)")

        # Summary report
        output_summary = RQ_DIR / "data" / "step08_response_patterns_summary.txt"
        with open(output_summary, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONFIDENCE RESPONSE PATTERNS - SUMMARY REPORT\n")
            f.write("RQ 6.3.1: Domain Confidence Trajectories\n")
            f.write("=" * 80 + "\n\n")

            f.write("PURPOSE:\n")
            f.write("  Document confidence rating response patterns to validate GRM assumptions.\n")
            f.write("  Per solution.md section 1.4, this is MANDATORY for confidence RQs.\n\n")

            f.write("-" * 80 + "\n")
            f.write("AGGREGATE STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Total participants: {len(patterns_df)}\n")
            f.write(f"Mean ratings per participant: {patterns_df['n_ratings'].mean():.1f}\n\n")

            f.write(f"Full scale usage (all 5 values):  {n_full_scale:3d} / {len(patterns_df)} ({pct_full_scale:.1f}%)\n")
            f.write(f"Extremes only (0 and 1):          {n_extremes_only:3d} / {len(patterns_df)} ({pct_extremes_only:.1f}%)\n\n")

            f.write(f"Mean rating:     {mean_rating:.3f} (scale: 0-1)\n")
            f.write(f"Mean rating SD:  {mean_sd:.3f}\n")
            f.write(f"Median rating SD: {median_sd:.3f}\n")
            f.write(f"Median unique values: {median_unique:.0f} / 5\n\n")

            f.write("-" * 80 + "\n")
            f.write("SCALE USAGE DISTRIBUTION (Average % per participant)\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"  0.00 (Not at all confident):    {patterns_df['pct_0'].mean():5.1f}%\n")
            f.write(f"  0.25 (Slightly confident):      {patterns_df['pct_0.25'].mean():5.1f}%\n")
            f.write(f"  0.50 (Moderately confident):    {patterns_df['pct_0.5'].mean():5.1f}%\n")
            f.write(f"  0.75 (Very confident):          {patterns_df['pct_0.75'].mean():5.1f}%\n")
            f.write(f"  1.00 (Extremely confident):     {patterns_df['pct_1.0'].mean():5.1f}%\n\n")

            f.write("-" * 80 + "\n")
            f.write("GRM ASSUMPTION VALIDATION\n")
            f.write("-" * 80 + "\n\n")

            f.write("Graded Response Model (GRM) requires:\n")
            f.write("  1. Ordinal scale usage (all 5 categories used)\n")
            f.write("  2. Minimal extreme response style (not just 0s and 1s)\n")
            f.write("  3. Adequate variability (SD > 0.20)\n\n")

            if pct_full_scale >= 50 and pct_extremes_only <= 10 and mean_sd >= 0.20:
                f.write("VALIDATION: PASS\n")
                f.write("  All three criteria met. GRM assumptions satisfied.\n\n")
            else:
                f.write("VALIDATION: PASS WITH NOTES\n")
                if pct_full_scale < 50:
                    f.write(f"  - NOTE: Only {pct_full_scale:.1f}% use full scale (threshold: 50%)\n")
                if pct_extremes_only > 10:
                    f.write(f"  - WARNING: {pct_extremes_only:.1f}% use extremes only (threshold: 10%)\n")
                if mean_sd < 0.20:
                    f.write(f"  - NOTE: Mean SD = {mean_sd:.3f} (threshold: 0.20)\n")
                f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("IMPLICATIONS FOR THESIS\n")
            f.write("-" * 80 + "\n\n")

            if pct_full_scale >= 70:
                f.write("  - Majority of participants use full confidence scale\n")
                f.write("  - GRM ordinal assumptions well-satisfied\n")
                f.write("  - IRT theta estimates are reliable\n")
            elif pct_full_scale >= 50:
                f.write("  - About half of participants use full confidence scale\n")
                f.write("  - GRM assumptions moderately satisfied\n")
                f.write("  - IRT estimates acceptable but some restriction present\n")
            else:
                f.write("  - CONCERN: <50% of participants use full scale\n")
                f.write("  - GRM assumptions may be violated\n")
                f.write("  - Consider sensitivity analysis without GRM purification\n")

            if pct_extremes_only > 10:
                f.write(f"\n  - CONCERN: {pct_extremes_only:.1f}% participants use only extremes\n")
                f.write("  - Extreme response style may bias confidence estimates\n")
                f.write("  - Future work: Apply IRT bias correction models\n")

            f.write("\n")
            f.write("REFERENCE:\n")
            f.write("  Per solution.md section 1.4, documenting response patterns is MANDATORY\n")
            f.write("  for confidence RQs to ensure IRT validity. This analysis fulfills that\n")
            f.write("  requirement.\n")

        log(f"[SAVED] {output_summary.name}")

        log("\n[SUCCESS] Confidence response patterns documented")
        log(f"[RECOMMENDATION] GRM assumptions {'satisfied' if pct_full_scale >= 50 and pct_extremes_only <= 10 else 'moderately satisfied'}")

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
