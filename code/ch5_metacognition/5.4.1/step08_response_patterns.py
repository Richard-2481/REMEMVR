"""
RQ 6.4.1 - Step 08: Confidence Response Patterns (MANDATORY Section 1.4)

PURPOSE:
Document confidence rating scale usage patterns to:
1. Validate GRM assumptions (do participants use full ordinal scale?)
2. Explain 100% item retention anomaly (exceptional quality vs restricted range?)
3. Assess extreme response style (ERS) or midpoint bias

ANALYSIS:
For each participant:
- Full-range usage: Uses all 5 values (0, 0.25, 0.5, 0.75, 1.0)?
- Extremes-only: Uses ONLY 0 and 1.0 (no midpoints)?
- Rating SD: Standard deviation of confidence ratings

AGGREGATES:
- % participants using full scale
- % participants using extremes only
- Mean rating SD across participants

INTERPRETATION:
- High full-range (>60%) → GRM appropriate, 100% retention reflects item quality
- High extremes-only (>30%) → GRM assumptions violated, consider dichotomous model
- Low rating SD (<0.8) → Restricted range, limited sensitivity

INPUT:
- data/step00_irt_input.csv (400 rows × 72 TC_* items, values in {0, 0.25, 0.5, 0.75, 1.0, NaN})

OUTPUT:
- data/step08_response_patterns.csv (participant-level statistics)
- data/step08_response_patterns_summary.txt (aggregate statistics + interpretation)

Date: 2025-12-28
RQ: ch6/6.4.1
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step08_response_patterns.log"
DATA_DIR = RQ_DIR / "data"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Step 08: Confidence Response Patterns")
        log("=" * 80)

        log("Loading raw confidence ratings...")
        irt_input = pd.read_csv(DATA_DIR / "step00_irt_input.csv", encoding='utf-8')

        # Extract composite_ID and TC_* items
        composite_id_col = 'composite_ID'
        tc_items = [col for col in irt_input.columns if col.startswith('TC_')]
        log(f"  ✓ Loaded {len(irt_input)} observations (UID × test combinations)")
        log(f"  ✓ Found {len(tc_items)} TC_* confidence items")

        # Extract UID from composite_ID (format: UID_test)
        irt_input['UID'] = irt_input[composite_id_col].str.split('_').str[0]
        unique_uids = irt_input['UID'].unique()
        log(f"  ✓ Unique participants: {len(unique_uids)}")

        # ANALYSIS: Per-participant response patterns
        log("\nComputing per-participant response patterns...")

        results = []
        for uid in unique_uids:
            # Get all confidence ratings for this participant (across all tests and items)
            uid_data = irt_input[irt_input['UID'] == uid][tc_items]
            all_ratings = uid_data.values.flatten()
            all_ratings = all_ratings[~pd.isna(all_ratings)]  # Remove NaN

            if len(all_ratings) == 0:
                log(f"  WARNING: {uid} has NO valid confidence ratings (all NaN)")
                continue

            # Valid confidence values
            valid_values = {0.0, 0.25, 0.5, 0.75, 1.0}
            unique_used = set(all_ratings)

            # Full-range usage: Uses all 5 values?
            full_range = unique_used == valid_values

            # Extremes-only: Uses ONLY 0 and 1.0 (no midpoints 0.25, 0.5, 0.75)?
            extremes_only = unique_used.issubset({0.0, 1.0}) and len(unique_used) > 0

            # Rating SD
            rating_sd = np.std(all_ratings, ddof=1)

            # Rating mean
            rating_mean = np.mean(all_ratings)

            # N ratings
            n_ratings = len(all_ratings)

            results.append({
                'UID': uid,
                'n_ratings': n_ratings,
                'unique_values_used': len(unique_used),
                'full_range_usage': full_range,
                'extremes_only': extremes_only,
                'rating_mean': rating_mean,
                'rating_sd': rating_sd,
                'uses_0': 0.0 in unique_used,
                'uses_025': 0.25 in unique_used,
                'uses_050': 0.5 in unique_used,
                'uses_075': 0.75 in unique_used,
                'uses_100': 1.0 in unique_used,
            })

        patterns_df = pd.DataFrame(results)
        log(f"  ✓ Computed patterns for {len(patterns_df)} participants")

        # AGGREGATES
        log("\nSummary statistics:")

        n_participants = len(patterns_df)
        pct_full_range = (patterns_df['full_range_usage'].sum() / n_participants) * 100
        pct_extremes_only = (patterns_df['extremes_only'].sum() / n_participants) * 100
        mean_rating_sd = patterns_df['rating_sd'].mean()
        median_rating_sd = patterns_df['rating_sd'].median()
        mean_n_unique = patterns_df['unique_values_used'].mean()

        log(f"  N participants analyzed: {n_participants}")
        log(f"  Full-range users (all 5 values): {patterns_df['full_range_usage'].sum()} ({pct_full_range:.1f}%)")
        log(f"  Extremes-only users (0 and 1.0 only): {patterns_df['extremes_only'].sum()} ({pct_extremes_only:.1f}%)")
        log(f"  Mean unique values used: {mean_n_unique:.2f}")
        log(f"  Mean rating SD: {mean_rating_sd:.3f}")
        log(f"  Median rating SD: {median_rating_sd:.3f}")

        
        # Value-specific usage
        log("\n[VALUE USAGE] Percentage of participants using each value:")
        val_cols = {'0.0': 'uses_0', '0.25': 'uses_025', '0.5': 'uses_050', '0.75': 'uses_075', '1.0': 'uses_100'}
        for val_str, col in val_cols.items():
            pct = (patterns_df[col].sum() / n_participants) * 100
            log(f"  Value {val_str}: {patterns_df[col].sum()} participants ({pct:.1f}%)")

        # INTERPRETATION
        log("\nResponse pattern quality:")

        # Flag 1: Full-range usage
        if pct_full_range > 60:
            full_range_flag = "GOOD"
            full_range_msg = f"Majority ({pct_full_range:.1f}%) use full scale → GRM appropriate"
        elif pct_full_range > 30:
            full_range_flag = "MODERATE"
            full_range_msg = f"Some ({pct_full_range:.1f}%) use full scale → GRM acceptable but not ideal"
        else:
            full_range_flag = "CONCERN"
            full_range_msg = f"Few ({pct_full_range:.1f}%) use full scale → GRM assumptions may be violated"

        log(f"  Full-range usage: {full_range_flag} - {full_range_msg}")

        # Flag 2: Extremes-only usage
        if pct_extremes_only > 30:
            extremes_flag = "CONCERN"
            extremes_msg = f"High extremes-only ({pct_extremes_only:.1f}%) → Consider dichotomous model (2PL instead of GRM)"
        elif pct_extremes_only > 10:
            extremes_flag = "MODERATE"
            extremes_msg = f"Some extremes-only ({pct_extremes_only:.1f}%) → Acceptable for GRM but note limitation"
        else:
            extremes_flag = "GOOD"
            extremes_msg = f"Low extremes-only ({pct_extremes_only:.1f}%) → GRM assumptions met"

        log(f"  Extremes-only: {extremes_flag} - {extremes_msg}")

        # Flag 3: Rating SD
        if mean_rating_sd > 0.25:
            sd_flag = "GOOD"
            sd_msg = f"Adequate variability (mean SD = {mean_rating_sd:.3f}) → Confidence scale sensitive"
        elif mean_rating_sd > 0.15:
            sd_flag = "MODERATE"
            sd_msg = f"Moderate variability (mean SD = {mean_rating_sd:.3f}) → Some restriction"
        else:
            sd_flag = "CONCERN"
            sd_msg = f"Low variability (mean SD = {mean_rating_sd:.3f}) → Restricted range, limited sensitivity"

        log(f"  Rating SD: {sd_flag} - {sd_msg}")

        # CONCLUSION
        log("\nOverall assessment:")

        if pct_full_range > 60 and pct_extremes_only < 10 and mean_rating_sd > 0.25:
            overall = "EXCELLENT"
            conclusion = "Response patterns support GRM validity. 100% item retention reflects genuine item quality."
        elif pct_full_range > 30 and pct_extremes_only < 30 and mean_rating_sd > 0.15:
            overall = "ACCEPTABLE"
            conclusion = "Response patterns acceptable for GRM. 100% item retention likely valid, document as unusual."
        else:
            overall = "CONCERNS"
            conclusion = "Response patterns raise concerns. 100% item retention may reflect lenient thresholds or GRM misfit."

        log(f"  Overall: {overall}")
        log(f"  {conclusion}")

        # Save participant-level data
        patterns_df.to_csv(DATA_DIR / "step08_response_patterns.csv", index=False, encoding='utf-8')
        log(f"\n  ✓ Saved participant-level patterns to step08_response_patterns.csv")

        # Save summary report
        with open(DATA_DIR / "step08_response_patterns_summary.txt", 'w', encoding='utf-8') as f:
            f.write("RQ 6.4.1 - Confidence Response Patterns Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"N participants: {n_participants}\n")
            f.write(f"N confidence items: {len(tc_items)}\n\n")
            f.write("AGGREGATE STATISTICS:\n")
            f.write(f"  Full-range users (all 5 values): {patterns_df['full_range_usage'].sum()} ({pct_full_range:.1f}%)\n")
            f.write(f"  Extremes-only users (0 and 1.0): {patterns_df['extremes_only'].sum()} ({pct_extremes_only:.1f}%)\n")
            f.write(f"  Mean unique values used: {mean_n_unique:.2f}\n")
            f.write(f"  Mean rating SD: {mean_rating_sd:.3f}\n")
            f.write("VALUE USAGE:\n")
            f.write(f"  Median rating SD: {median_rating_sd:.3f}\n\n")
            val_cols = {'0.0': 'uses_0', '0.25': 'uses_025', '0.5': 'uses_050', '0.75': 'uses_075', '1.0': 'uses_100'}
            for val_str, col in val_cols.items():
                pct = (patterns_df[col].sum() / n_participants) * 100
                f.write(f"  {val_str}: {patterns_df[col].sum()} participants ({pct:.1f}%)\n")
            f.write("\nINTERPRETATION:\n")
            f.write(f"  Full-range usage: {full_range_flag} - {full_range_msg}\n")
            f.write(f"  Extremes-only: {extremes_flag} - {extremes_msg}\n")
            f.write(f"  Rating SD: {sd_flag} - {sd_msg}\n\n")
            f.write(f"OVERALL: {overall}\n")
            f.write(f"{conclusion}\n\n")
            f.write("RECOMMENDATION FOR summary.md:\n")
            f.write(f"Add to Section 4 (Limitations):\n")
            f.write(f'"Confidence response patterns: {pct_full_range:.0f}% full-range, {pct_extremes_only:.0f}% extremes-only, mean SD={mean_rating_sd:.2f}. ')
            f.write(f"{conclusion}"+"\n")

        log(f"  ✓ Saved summary report to step08_response_patterns_summary.txt")

        log("\n" + "=" * 80)
        log("Step 08: Response Patterns Complete")
        log("=" * 80)

    except Exception as e:
        log(f"{e}")
        import traceback
        log(traceback.format_exc())
        raise
