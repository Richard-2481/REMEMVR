#!/usr/bin/env python3
"""
==============================================================================
Step 07: Prepare Dual-Scale Plot Data
==============================================================================
RQ: 5.5.2 - Source-Destination Consolidation (Two-Phase)
Generated: 2025-12-05

Purpose:
    Create plot source data for trajectory visualization with both theta
    (latent ability) and probability (observable performance) scales.
    Decision D069: Dual-scale trajectory plots required.

Input:
    - data/step02_lmm_input_long.csv (800 rows - observed data)
    - data/step04_segment_location_slopes.csv (4 rows - segment-location slopes)

Output:
    - data/step07_piecewise_theta_data.csv (theta scale)
    - data/step07_piecewise_probability_data.csv (probability scale)
==============================================================================
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy import stats

# ==============================================================================
# PATH SETUP
# ==============================================================================

RQ_DIR = Path(__file__).resolve().parent.parent  # results/ch5/5.5.2

# Folder conventions
DATA_DIR = RQ_DIR / "data"
LOGS_DIR = RQ_DIR / "logs"

# Input
INPUT_DATA = DATA_DIR / "step02_lmm_input_long.csv"
INPUT_SLOPES = DATA_DIR / "step04_segment_location_slopes.csv"

# Output
OUTPUT_THETA = DATA_DIR / "step07_piecewise_theta_data.csv"
OUTPUT_PROB = DATA_DIR / "step07_piecewise_probability_data.csv"
LOG_FILE = LOGS_DIR / "step07_prepare_plot_data.log"

# Create directories
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOGGING
# ==============================================================================

def log(msg: str) -> None:
    """Log message to console and file."""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log file
LOG_FILE.write_text("", encoding="utf-8")


def logistic(theta):
    """Convert theta to probability: p = 1 / (1 + exp(-theta))"""
    return 1.0 / (1.0 + np.exp(-theta))


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 7: Prepare Dual-Scale Plot Data")

        # ==============================================================
        # STEP 1: Load Input Data
        # ==============================================================

        log("[LOAD] Loading LMM input data...")
        df = pd.read_csv(INPUT_DATA)
        log(f"[LOADED] {INPUT_DATA.name} ({len(df)} rows)")

        log("[LOAD] Loading segment-location slopes...")
        df_slopes = pd.read_csv(INPUT_SLOPES)
        log(f"[LOADED] {INPUT_SLOPES.name} ({len(df_slopes)} rows)")

        # ==============================================================
        # STEP 2: Compute Observed Means per Segment x LocationType
        # ==============================================================

        log("[AGGREGATE] Computing observed means per Segment x LocationType...")

        observed_list = []

        for segment in ['Early', 'Late']:
            for location in ['Source', 'Destination']:
                mask = (df['Segment'] == segment) & (df['LocationType'] == location)
                subset = df[mask]

                n_obs = len(subset)
                mean_theta = subset['theta'].mean()
                sd_theta = subset['theta'].std()
                se_theta = sd_theta / np.sqrt(n_obs)

                # 95% CI
                t_crit = stats.t.ppf(0.975, df=n_obs-1)
                ci_lower = mean_theta - t_crit * se_theta
                ci_upper = mean_theta + t_crit * se_theta

                # Mean Days_within for x-axis position
                mean_days = subset['Days_within'].mean()

                observed_list.append({
                    'Segment': segment,
                    'LocationType': location,
                    'Days_within': mean_days,
                    'theta': mean_theta,
                    'CI_lower': ci_lower,
                    'CI_upper': ci_upper,
                    'n_obs': n_obs,
                    'Data_Type': 'observed'
                })

                log(f"  {segment}-{location}: n={n_obs}, theta={mean_theta:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

        df_observed = pd.DataFrame(observed_list)
        log(f"[AGGREGATED] 4 observed means computed")

        # ==============================================================
        # STEP 3: Generate Model Predictions
        # ==============================================================

        log("[PREDICT] Generating model-predicted trajectories...")

        # Create prediction grid
        # Early: 0 to 2 days
        early_days = np.linspace(0, 2, 20)
        # Late: 0 to 8 days (recentered)
        late_days = np.linspace(0, 8, 60)

        # Get slopes from Step 4
        slopes_dict = {}
        for _, row in df_slopes.iterrows():
            key = (row['Segment'], row['LocationType'])
            slopes_dict[key] = {
                'slope': row['slope'],
                'SE': row['SE'],
                'CI_lower': row['CI_lower'],
                'CI_upper': row['CI_upper']
            }

        # Get intercepts from observed means at Days_within closest to 0
        # For Early: use observed mean at early timepoint
        # For Late: use observed mean at late segment start

        # Early segment intercepts (from observed Early means)
        early_source = df_observed[(df_observed['Segment'] == 'Early') &
                                   (df_observed['LocationType'] == 'Source')].iloc[0]
        early_dest = df_observed[(df_observed['Segment'] == 'Early') &
                                 (df_observed['LocationType'] == 'Destination')].iloc[0]

        # Late segment intercepts (need to compute from early end-point)
        # Late segment starts at Days_within=0 (which is TSVR=48h)
        # Use early end point + transition
        late_source = df_observed[(df_observed['Segment'] == 'Late') &
                                  (df_observed['LocationType'] == 'Source')].iloc[0]
        late_dest = df_observed[(df_observed['Segment'] == 'Late') &
                                (df_observed['LocationType'] == 'Destination')].iloc[0]

        prediction_list = []

        # Early predictions
        for location in ['Source', 'Destination']:
            slope_info = slopes_dict[('Early', location)]
            slope = slope_info['slope']

            # Intercept from observed data (extrapolate to Days_within=0)
            if location == 'Source':
                obs_mean = early_source['theta']
                obs_days = early_source['Days_within']
            else:
                obs_mean = early_dest['theta']
                obs_days = early_dest['Days_within']

            # Intercept at Days_within=0
            intercept = obs_mean - slope * obs_days

            for d in early_days:
                theta_pred = intercept + slope * d
                prediction_list.append({
                    'Segment': 'Early',
                    'LocationType': location,
                    'Days_within': d,
                    'theta': theta_pred,
                    'CI_lower': np.nan,  # Predictions don't have simple CIs
                    'CI_upper': np.nan,
                    'n_obs': 0,
                    'Data_Type': 'predicted'
                })

        # Late predictions
        for location in ['Source', 'Destination']:
            slope_info = slopes_dict[('Late', location)]
            slope = slope_info['slope']

            # Intercept from observed data
            if location == 'Source':
                obs_mean = late_source['theta']
                obs_days = late_source['Days_within']
            else:
                obs_mean = late_dest['theta']
                obs_days = late_dest['Days_within']

            # Intercept at Days_within=0 (start of Late segment)
            intercept = obs_mean - slope * obs_days

            for d in late_days:
                theta_pred = intercept + slope * d
                prediction_list.append({
                    'Segment': 'Late',
                    'LocationType': location,
                    'Days_within': d,
                    'theta': theta_pred,
                    'CI_lower': np.nan,
                    'CI_upper': np.nan,
                    'n_obs': 0,
                    'Data_Type': 'predicted'
                })

        df_predicted = pd.DataFrame(prediction_list)
        log(f"[PREDICTED] {len(df_predicted)} prediction points generated")

        # ==============================================================
        # STEP 4: Combine and Save Theta Scale Data
        # ==============================================================

        log("[COMBINE] Combining observed and predicted data (theta scale)...")
        df_theta = pd.concat([df_observed, df_predicted], ignore_index=True)
        log(f"[COMBINED] {len(df_theta)} total rows")

        log(f"[SAVE] Saving to {OUTPUT_THETA.name}...")
        df_theta.to_csv(OUTPUT_THETA, index=False)
        log(f"[SAVED] {OUTPUT_THETA.name}")

        # ==============================================================
        # STEP 5: Convert to Probability Scale
        # ==============================================================

        log("[TRANSFORM] Converting theta to probability scale...")

        df_prob = df_theta.copy()
        df_prob['probability'] = logistic(df_prob['theta'])
        df_prob['CI_lower_prob'] = logistic(df_prob['CI_lower'])
        df_prob['CI_upper_prob'] = logistic(df_prob['CI_upper'])

        # Rename columns for probability scale
        df_prob = df_prob.rename(columns={
            'theta': 'theta_original',
            'probability': 'theta',  # Use theta column for consistency
            'CI_lower': 'CI_lower_theta',
            'CI_upper': 'CI_upper_theta',
            'CI_lower_prob': 'CI_lower',
            'CI_upper_prob': 'CI_upper'
        })

        # Keep only relevant columns
        df_prob = df_prob[['Segment', 'LocationType', 'Days_within', 'theta',
                          'CI_lower', 'CI_upper', 'n_obs', 'Data_Type']]

        log(f"[SAVE] Saving to {OUTPUT_PROB.name}...")
        df_prob.to_csv(OUTPUT_PROB, index=False)
        log(f"[SAVED] {OUTPUT_PROB.name}")

        # ==============================================================
        # STEP 6: Validation
        # ==============================================================

        log("[VALIDATION] Checking output data quality...")

        # Check theta scale data
        if len(df_theta) < 4:
            raise ValueError(f"Expected at least 4 rows in theta data, got {len(df_theta)}")
        log(f"[PASS] Theta scale: {len(df_theta)} rows")

        # Check all Segment x LocationType combinations present
        observed_theta = df_theta[df_theta['Data_Type'] == 'observed']
        expected_combos = {('Early', 'Source'), ('Early', 'Destination'),
                         ('Late', 'Source'), ('Late', 'Destination')}
        actual_combos = set(zip(observed_theta['Segment'], observed_theta['LocationType']))
        if actual_combos != expected_combos:
            raise ValueError(f"Missing segment-location combinations: {expected_combos - actual_combos}")
        log("[PASS] All 4 Segment x LocationType combinations present")

        # Check probability scale data
        if len(df_prob) != len(df_theta):
            raise ValueError("Probability scale row count mismatch")
        log(f"[PASS] Probability scale: {len(df_prob)} rows")

        # Check probability values in [0, 1]
        observed_prob = df_prob[df_prob['Data_Type'] == 'observed']
        if observed_prob['theta'].min() < 0 or observed_prob['theta'].max() > 1:
            raise ValueError(f"Probability values out of range [0, 1]")
        log(f"[PASS] Probability values in valid range [0, 1]")

        # Check CI ordering for observed values
        valid_ci = observed_theta['CI_lower'] <= observed_theta['CI_upper']
        if not valid_ci.all():
            raise ValueError("CI_lower > CI_upper detected")
        log("[PASS] CI ordering valid (lower <= upper)")

        # ==============================================================
        # SUCCESS
        # ==============================================================

        log("[SUCCESS] Step 07 complete")
        log(f"[SUCCESS] Theta scale data: {OUTPUT_THETA}")
        log(f"[SUCCESS] Probability scale data: {OUTPUT_PROB}")
        log("[SUCCESS] Decision D069 compliance: Dual-scale plot data prepared")

        # Summary statistics
        log("[SUMMARY] Observed means (theta scale):")
        for _, row in observed_theta.iterrows():
            log(f"  {row['Segment']}-{row['LocationType']}: "
                f"theta={row['theta']:.3f} [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]")

        log("[SUMMARY] Observed means (probability scale):")
        for _, row in observed_prob[observed_prob['Data_Type'] == 'observed'].iterrows():
            log(f"  {row['Segment']}-{row['LocationType']}: "
                f"p={row['theta']:.3f} [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]")

        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
