#!/usr/bin/env python3
"""Extract Segment-Specific Slopes: Extract 6 segment-congruence slopes (Early/Late x Common/Congruent/Incongruent)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLMResults
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq6
LOG_FILE = RQ_DIR / "logs" / "step03_extract_slopes.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Slope Computation Functions (INLINE IMPLEMENTATION - to extract to tools/ later)

def compute_segment_congruence_slopes(lmm_model: MixedLMResults) -> pd.DataFrame:
    """
    Compute 6 segment-congruence slopes via linear combinations with delta method SEs.

    Piecewise LMM uses treatment coding:
    - Segment reference: 'Early'
    - Congruence reference: 'Common'

    Slope formulas:
      Early-Common: beta[Days_within]
      Early-Congruent: beta[Days_within] + beta[Days_within:Congruence[T.Congruent]]
      Early-Incongruent: beta[Days_within] + beta[Days_within:Congruence[T.Incongruent]]
      Late-Common: beta[Days_within] + beta[Days_within:Segment[T.Late]]
      Late-Congruent: beta[Days_within] + beta[Days_within:Segment[T.Late]] +
                      beta[Days_within:Congruence[T.Congruent]] +
                      beta[Days_within:Segment[T.Late]:Congruence[T.Congruent]]
      Late-Incongruent: beta[Days_within] + beta[Days_within:Segment[T.Late]] +
                        beta[Days_within:Congruence[T.Incongruent]] +
                        beta[Days_within:Segment[T.Late]:Congruence[T.Incongruent]]

    Args:
        lmm_model: Fitted MixedLMResults object

    Returns:
        DataFrame with columns: Segment, Congruence, Slope, SE, CI_lower, CI_upper, Interpretation
    """
    params = lmm_model.params
    cov_matrix = lmm_model.cov_params()
    coef_names = list(params.index)

    log(f"Found {len(coef_names)} coefficients:")
    for name in coef_names:
        log(f"  {name}: {params[name]:.4f}")

    # Find coefficient names (treatment coding creates verbose names)
    def find_coef_list(patterns: list) -> list:
        """Find coefficient names matching ALL patterns."""
        matches = []
        for name in coef_names:
            if all(p in name for p in patterns):
                matches.append(name)
        return matches

    # Find key coefficients
    days_within = 'Days_within'

    # 2-way interactions
    days_seg_late = find_coef_list(['Days_within', 'Segment', 'Late'])
    days_seg_late = [n for n in days_seg_late if 'Congruence' not in n]  # Exclude 3-way

    days_cong_congruent = find_coef_list(['Days_within', 'Congruence', 'Congruent'])
    days_cong_congruent = [n for n in days_cong_congruent if 'Segment' not in n]  # Exclude 3-way

    days_cong_incongruent = find_coef_list(['Days_within', 'Congruence', 'Incongruent'])
    days_cong_incongruent = [n for n in days_cong_incongruent if 'Segment' not in n]  # Exclude 3-way

    # 3-way interactions
    days_seg_cong_congruent = find_coef_list(['Days_within', 'Segment', 'Late', 'Congruent'])
    days_seg_cong_incongruent = find_coef_list(['Days_within', 'Segment', 'Late', 'Incongruent'])

    # Extract single names (should be exactly 1 match each)
    days_seg_late = days_seg_late[0] if days_seg_late else None
    days_cong_congruent = days_cong_congruent[0] if days_cong_congruent else None
    days_cong_incongruent = days_cong_incongruent[0] if days_cong_incongruent else None
    days_seg_cong_congruent = days_seg_cong_congruent[0] if days_seg_cong_congruent else None
    days_seg_cong_incongruent = days_seg_cong_incongruent[0] if days_seg_cong_incongruent else None

    log(f"Identified coefficients:")
    log(f"  days_within: {days_within}")
    log(f"  days_seg_late: {days_seg_late}")
    log(f"  days_cong_congruent: {days_cong_congruent}")
    log(f"  days_cong_incongruent: {days_cong_incongruent}")
    log(f"  days_seg_cong_congruent: {days_seg_cong_congruent}")
    log(f"  days_seg_cong_incongruent: {days_seg_cong_incongruent}")

    # Get coefficient values
    b_days = params[days_within]
    b_seg_late = params[days_seg_late] if days_seg_late else 0
    b_cong_congruent = params[days_cong_congruent] if days_cong_congruent else 0
    b_cong_incongruent = params[days_cong_incongruent] if days_cong_incongruent else 0
    b_seg_cong_congruent = params[days_seg_cong_congruent] if days_seg_cong_congruent else 0
    b_seg_cong_incongruent = params[days_seg_cong_incongruent] if days_seg_cong_incongruent else 0

    # Compute slopes
    slopes = []

    # 1. Early-Common: beta[Days_within]
    slope_early_common = b_days
    se_early_common = np.sqrt(cov_matrix.loc[days_within, days_within])
    slopes.append({
        'Segment': 'Early',
        'Congruence': 'Common',
        'Slope': slope_early_common,
        'SE': se_early_common
    })
    log(f"Early-Common: {slope_early_common:.4f} +/- {se_early_common:.4f}")

    # 2. Early-Congruent: beta[Days_within] + beta[Days_within:Congruence[T.Congruent]]
    slope_early_congruent = b_days + b_cong_congruent
    if days_cong_congruent:
        terms = [days_within, days_cong_congruent]
        var_sum = sum(cov_matrix.loc[t1, t2] for t1 in terms for t2 in terms)
        se_early_congruent = np.sqrt(var_sum)
    else:
        se_early_congruent = se_early_common
    slopes.append({
        'Segment': 'Early',
        'Congruence': 'Congruent',
        'Slope': slope_early_congruent,
        'SE': se_early_congruent
    })
    log(f"Early-Congruent: {slope_early_congruent:.4f} +/- {se_early_congruent:.4f}")

    # 3. Early-Incongruent: beta[Days_within] + beta[Days_within:Congruence[T.Incongruent]]
    slope_early_incongruent = b_days + b_cong_incongruent
    if days_cong_incongruent:
        terms = [days_within, days_cong_incongruent]
        var_sum = sum(cov_matrix.loc[t1, t2] for t1 in terms for t2 in terms)
        se_early_incongruent = np.sqrt(var_sum)
    else:
        se_early_incongruent = se_early_common
    slopes.append({
        'Segment': 'Early',
        'Congruence': 'Incongruent',
        'Slope': slope_early_incongruent,
        'SE': se_early_incongruent
    })
    log(f"Early-Incongruent: {slope_early_incongruent:.4f} +/- {se_early_incongruent:.4f}")

    # 4. Late-Common: beta[Days_within] + beta[Days_within:Segment[T.Late]]
    slope_late_common = b_days + b_seg_late
    if days_seg_late:
        terms = [days_within, days_seg_late]
        var_sum = sum(cov_matrix.loc[t1, t2] for t1 in terms for t2 in terms)
        se_late_common = np.sqrt(var_sum)
    else:
        se_late_common = se_early_common
    slopes.append({
        'Segment': 'Late',
        'Congruence': 'Common',
        'Slope': slope_late_common,
        'SE': se_late_common
    })
    log(f"Late-Common: {slope_late_common:.4f} +/- {se_late_common:.4f}")

    # 5. Late-Congruent: All 4 terms
    slope_late_congruent = b_days + b_seg_late + b_cong_congruent + b_seg_cong_congruent
    terms = [days_within]
    if days_seg_late:
        terms.append(days_seg_late)
    if days_cong_congruent:
        terms.append(days_cong_congruent)
    if days_seg_cong_congruent:
        terms.append(days_seg_cong_congruent)
    var_sum = sum(cov_matrix.loc[t1, t2] for t1 in terms for t2 in terms)
    se_late_congruent = np.sqrt(var_sum)
    slopes.append({
        'Segment': 'Late',
        'Congruence': 'Congruent',
        'Slope': slope_late_congruent,
        'SE': se_late_congruent
    })
    log(f"Late-Congruent: {slope_late_congruent:.4f} +/- {se_late_congruent:.4f}")

    # 6. Late-Incongruent: All 4 terms
    slope_late_incongruent = b_days + b_seg_late + b_cong_incongruent + b_seg_cong_incongruent
    terms = [days_within]
    if days_seg_late:
        terms.append(days_seg_late)
    if days_cong_incongruent:
        terms.append(days_cong_incongruent)
    if days_seg_cong_incongruent:
        terms.append(days_seg_cong_incongruent)
    var_sum = sum(cov_matrix.loc[t1, t2] for t1 in terms for t2 in terms)
    se_late_incongruent = np.sqrt(var_sum)
    slopes.append({
        'Segment': 'Late',
        'Congruence': 'Incongruent',
        'Slope': slope_late_incongruent,
        'SE': se_late_incongruent
    })
    log(f"Late-Incongruent: {slope_late_incongruent:.4f} +/- {se_late_incongruent:.4f}")

    # Create DataFrame
    df_slopes = pd.DataFrame(slopes)

    # Add 95% CI
    df_slopes['CI_lower'] = df_slopes['Slope'] - 1.96 * df_slopes['SE']
    df_slopes['CI_upper'] = df_slopes['Slope'] + 1.96 * df_slopes['SE']

    # Add interpretation
    df_slopes['Interpretation'] = df_slopes['Slope'].apply(
        lambda x: 'decline' if x < 0 else 'improvement'
    )

    return df_slopes

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: Extract Segment-Specific Slopes")
        # Load Fitted LMM Model

        log("Loading fitted piecewise LMM model from Step 2...")

        model_path = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
        lmm_model = MixedLMResults.load(str(model_path))

        log(f"{model_path.name}")
        # Compute Segment-Congruence Slopes
        # Output: 6 slopes with SEs and CIs

        log("Computing 6 segment-congruence slopes via delta method...")

        df_slopes = compute_segment_congruence_slopes(lmm_model)

        log(f"Computed {len(df_slopes)} slopes")
        # Validate and Save

        log("Validating slopes...")

        # Check slope count
        if len(df_slopes) != 6:
            raise ValueError(
                f"Slope count incorrect: expected 6, found {len(df_slopes)}"
            )
        log(f"Slope count correct (6)")

        # Check all combinations present
        expected_combos = [
            ('Early', 'Common'), ('Early', 'Congruent'), ('Early', 'Incongruent'),
            ('Late', 'Common'), ('Late', 'Congruent'), ('Late', 'Incongruent')
        ]
        actual_combos = set(zip(df_slopes['Segment'], df_slopes['Congruence']))
        if actual_combos != set(expected_combos):
            raise ValueError(
                f"Segment-congruence combinations incomplete: "
                f"expected {expected_combos}, found {actual_combos}"
            )
        log(f"All segment-congruence combinations present")

        # Check all SEs are positive
        if (df_slopes['SE'] <= 0).any():
            raise ValueError(
                f"Delta method failed: Found non-positive SE values"
            )
        log(f"All SE values positive (delta method successful)")

        # Check no NaN
        if df_slopes[['Slope', 'SE', 'CI_lower', 'CI_upper']].isna().any().any():
            raise ValueError(
                f"Missing values detected in slopes/SEs/CIs"
            )
        log(f"No missing values")

        # Check CI ordering
        if not ((df_slopes['CI_lower'] < df_slopes['Slope']).all() and
                (df_slopes['Slope'] < df_slopes['CI_upper']).all()):
            raise ValueError(
                f"CI ordering incorrect: Expected CI_lower < Slope < CI_upper for all rows"
            )
        log(f"CI ordering correct")

        # Check slope range plausibility (warning only)
        if (df_slopes['Slope'].abs() > 2).any():
            log(f"Some slopes exceed plausible range [-2, 2] theta/day")
            for _, row in df_slopes[df_slopes['Slope'].abs() > 2].iterrows():
                log(f"  {row['Segment']}-{row['Congruence']}: {row['Slope']:.4f}")

        # Save output
        output_path = RQ_DIR / "results" / "step03_segment_slopes.csv"
        df_slopes.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(df_slopes)} rows)")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
