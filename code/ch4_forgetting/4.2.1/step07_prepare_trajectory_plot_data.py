#!/usr/bin/env python3
"""Prepare Trajectory Plot Data (Decision D069): Aggregate analysis outputs for dual-scale trajectory visualization (Decision D069)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback
import pickle
import statsmodels.api as sm

# parents[4] = REMEMVR/ (code -> 5.2.1 -> ch5 -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import plotting tools
from tools.plotting import convert_theta_to_probability

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.1
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_trajectory_plot_data.log"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
(RQ_DIR / "plots").mkdir(parents=True, exist_ok=True)

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Clear log file for fresh run
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        log("Step 07: Prepare Trajectory Plot Data (Decision D069)")
        log("=" * 60)
        # Load Input Data

        log("\nLoading input data...")

        # Load LMM input (step04 output)
        lmm_input_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        if not lmm_input_path.exists():
            raise FileNotFoundError(f"Input file not found: {lmm_input_path}")
        df_lmm = pd.read_csv(lmm_input_path, encoding='utf-8')
        log(f"step04_lmm_input.csv ({len(df_lmm)} rows)")

        # Load item parameters (step03 output)
        item_params_path = RQ_DIR / "data" / "step03_item_parameters.csv"
        if not item_params_path.exists():
            raise FileNotFoundError(f"Input file not found: {item_params_path}")
        df_items = pd.read_csv(item_params_path, encoding='utf-8')
        log(f"step03_item_parameters.csv ({len(df_items)} items)")

        # Map test values from {1,2,3,4} to nominal days {0,1,3,6}
        # Test 1 = Day 0 (immediate), Test 2 = Day 1, Test 3 = Day 3, Test 4 = Day 6
        TEST_TO_DAYS = {1: 0, 2: 1, 3: 3, 4: 6}
        if df_lmm['test'].isin([1, 2, 3, 4]).all():
            df_lmm['test'] = df_lmm['test'].map(TEST_TO_DAYS)
            log(f"Test values mapped: {{1,2,3,4}} -> {{0,1,3,6}}")
        # Compute Domain-Specific IRT Parameters
        # For probability conversion, we need average discrimination and difficulty
        # per domain from the item parameters

        log("\nComputing domain-specific IRT parameters...")

        # Identify domain column (may be 'factor' or 'domain')
        factor_col = 'factor' if 'factor' in df_items.columns else 'domain'

        # Check which difficulty column exists
        diff_col = None
        for col in ['Difficulty_1', 'Difficulty', 'b', 'difficulty']:
            if col in df_items.columns:
                diff_col = col
                break
        if diff_col is None:
            raise ValueError(f"No difficulty column found. Available: {df_items.columns.tolist()}")

        # Check which discrimination column exists
        disc_col = None
        for col in ['Discrimination', 'a', 'discrimination']:
            if col in df_items.columns:
                disc_col = col
                break
        if disc_col is None:
            raise ValueError(f"No discrimination column found. Available: {df_items.columns.tolist()}")

        log(f"  Using columns: factor={factor_col}, disc={disc_col}, diff={diff_col}")

        # Compute mean a and b per domain
        domain_params = df_items.groupby(factor_col).agg({
            disc_col: 'mean',
            diff_col: 'mean'
        }).reset_index()
        domain_params.columns = ['domain', 'mean_a', 'mean_b']

        log(f"Domain parameters computed:")
        for _, row in domain_params.iterrows():
            log(f"  {row['domain']}: a={row['mean_a']:.3f}, b={row['mean_b']:.3f}")
        # Fit LMM Model for Predictions

        log("\nFitting LMM model for predictions...")

        # Prepare data for LMM (need log_Days and domain coding)
        df_plot = df_lmm.copy()
        df_plot['Days'] = df_plot['TSVR_hours'] / 24.0
        df_plot['log_Days'] = np.log(df_plot['Days'] + 1)

        # Get UID from composite_ID if not present
        if 'UID' not in df_plot.columns:
            df_plot['UID'] = df_plot['composite_ID'].str.split('_').str[0]

        # Re-fit the Log model (best model from step05)
        import statsmodels.formula.api as smf

        # Fit Log model: Ability ~ log(Days) * Domain with random intercept + slope per UID
        log_model = smf.mixedlm(
            "theta ~ log_Days * C(domain, Treatment('what'))",
            data=df_plot,
            groups=df_plot['UID'],
            re_formula='~log_Days'
        )
        log_result = log_model.fit(method='powell', reml=False)
        log(f"  Model fitted: AIC = {log_result.aic:.2f}")
        # Generate Individual-Level Predictions

        log("\nGenerating predictions for each observation...")

        # Get fixed effects
        fe = log_result.fe_params

        # Generate predictions for each row
        predicted_theta = []
        for _, row in df_plot.iterrows():
            domain = row['domain']
            log_days = row['log_Days']

            # Build prediction from fixed effects only (marginal prediction)
            pred = fe['Intercept'] + fe['log_Days'] * log_days

            # Add domain effects (treatment coding with 'what' as reference)
            if domain == 'when':
                pred += fe["C(domain, Treatment('what'))[T.when]"]
                pred += fe["log_Days:C(domain, Treatment('what'))[T.when]"] * log_days
            elif domain == 'where':
                pred += fe["C(domain, Treatment('what'))[T.where]"]
                pred += fe["log_Days:C(domain, Treatment('what'))[T.where]"] * log_days
            # 'what' is reference, no additional terms

            predicted_theta.append(float(pred))

        df_plot['predicted_theta'] = predicted_theta
        log(f"Predictions generated for {len(df_plot)} observations")
        # Prepare Theta Scale Output (Individual-Level)

        log("\nCreating theta scale plot data (individual-level)...")

        # Select columns for theta output
        df_theta_plot = df_plot[['TSVR_hours', 'domain', 'theta', 'predicted_theta', 'UID']].copy()

        # Sort for consistent output
        df_theta_plot = df_theta_plot.sort_values(['domain', 'TSVR_hours']).reset_index(drop=True)

        log(f"Theta plot data prepared: {len(df_theta_plot)} rows")
        log(f"  Domains: {sorted(df_theta_plot['domain'].unique().tolist())}")
        log(f"  TSVR range: {df_theta_plot['TSVR_hours'].min():.1f} - {df_theta_plot['TSVR_hours'].max():.1f} hours")
        # Convert to Probability Scale (Decision D069)

        log("\nTransforming to probability scale (Decision D069)...")

        # Create probability dataframe
        df_prob_plot = df_theta_plot[['TSVR_hours', 'domain', 'UID']].copy()

        # Convert each observation's theta values to probability using domain-specific params
        probs = []
        predicted_probs = []

        for _, row in df_theta_plot.iterrows():
            domain = row['domain']

            # Get domain-specific parameters
            domain_row = domain_params[domain_params['domain'] == domain]
            if len(domain_row) == 0:
                log(f"  Warning: No parameters for domain '{domain}', using defaults (a=1, b=0)")
                a, b = 1.0, 0.0
            else:
                a = domain_row['mean_a'].values[0]
                b = domain_row['mean_b'].values[0]

            # Convert theta to probability
            prob = convert_theta_to_probability(row['theta'], discrimination=a, difficulty=b)
            predicted_prob = convert_theta_to_probability(row['predicted_theta'], discrimination=a, difficulty=b)

            probs.append(float(prob))
            predicted_probs.append(float(predicted_prob))

        df_prob_plot['probability'] = probs
        df_prob_plot['predicted_probability'] = predicted_probs

        # Reorder columns
        df_prob_plot = df_prob_plot[['TSVR_hours', 'domain', 'probability', 'predicted_probability', 'UID']]

        log(f"Probability conversion complete")
        log(f"  Probability range: {df_prob_plot['probability'].min():.3f} - {df_prob_plot['probability'].max():.3f}")
        # Save Outputs

        log("\nSaving plot data files...")

        # Save theta scale data
        theta_output_path = RQ_DIR / "plots" / "step07_trajectory_theta_data.csv"
        df_theta_plot.to_csv(theta_output_path, index=False, encoding='utf-8')
        log(f"{theta_output_path}")
        log(f"  Rows: {len(df_theta_plot)}, Columns: {list(df_theta_plot.columns)}")

        # Save probability scale data
        prob_output_path = RQ_DIR / "plots" / "step07_trajectory_probability_data.csv"
        df_prob_plot.to_csv(prob_output_path, index=False, encoding='utf-8')
        log(f"{prob_output_path}")
        log(f"  Rows: {len(df_prob_plot)}, Columns: {list(df_prob_plot.columns)}")
        # Validation

        log("\nRunning validation checks...")

        validation_errors = []

        # Check 1: Both output files exist
        if not theta_output_path.exists():
            validation_errors.append("Theta output file not created")
        if not prob_output_path.exists():
            validation_errors.append("Probability output file not created")
        if not validation_errors:
            log("  Both output files exist")

        # Check 2: Row count (individual-level: ~1200 = 100 participants x 4 tests x 3 domains)
        if len(df_theta_plot) < 1000:
            validation_errors.append(f"Theta data has {len(df_theta_plot)} rows, expected ~1200")
        else:
            log(f"  Individual-level data: {len(df_theta_plot)} rows")

        # Check 3: Domain coverage
        expected_domains = {'what', 'where', 'when'}
        actual_domains = set(df_theta_plot['domain'].unique())
        if actual_domains != expected_domains:
            validation_errors.append(f"Missing domains: {expected_domains - actual_domains}")
        else:
            log("  All 3 domains present")

        # Check 4: TSVR range reasonable (0-300 hours = ~12 days max, study is ~6 days nominal)
        if df_theta_plot['TSVR_hours'].min() < 0 or df_theta_plot['TSVR_hours'].max() > 300:
            validation_errors.append(f"TSVR_hours out of expected range: {df_theta_plot['TSVR_hours'].min():.1f} - {df_theta_plot['TSVR_hours'].max():.1f}")
        else:
            log(f"  TSVR range valid: {df_theta_plot['TSVR_hours'].min():.1f} - {df_theta_plot['TSVR_hours'].max():.1f} hours")

        # Check 5: No NaN values in critical columns
        critical_cols_theta = ['TSVR_hours', 'domain', 'theta', 'predicted_theta']
        critical_cols_prob = ['TSVR_hours', 'domain', 'probability', 'predicted_probability']
        nan_theta = df_theta_plot[critical_cols_theta].isna().sum().sum()
        nan_prob = df_prob_plot[critical_cols_prob].isna().sum().sum()
        if nan_theta > 0:
            validation_errors.append(f"NaN values in theta data: {nan_theta}")
        if nan_prob > 0:
            validation_errors.append(f"NaN values in probability data: {nan_prob}")
        if nan_theta == 0 and nan_prob == 0:
            log("  No NaN values in critical columns")

        # Check 6: Probability bounds
        if df_prob_plot['probability'].min() < 0 or df_prob_plot['probability'].max() > 1:
            validation_errors.append("probability out of [0, 1] range")
        else:
            log("  Probability values in [0, 1] range")

        # Check 7: Predictions present
        if 'predicted_theta' not in df_theta_plot.columns:
            validation_errors.append("predicted_theta column missing")
        if 'predicted_probability' not in df_prob_plot.columns:
            validation_errors.append("predicted_probability column missing")
        if 'predicted_theta' in df_theta_plot.columns and 'predicted_probability' in df_prob_plot.columns:
            log("  Prediction columns present")

        # Report validation result
        if validation_errors:
            log("\nValidation errors detected:")
            for err in validation_errors:
                log(f"  - {err}")
            raise ValueError(f"Validation failed: {validation_errors}")
        else:
            log("\nAll validation checks passed")
        # Summary

        log("\n" + "=" * 60)
        log("Step 07 complete: Trajectory plot data prepared")
        log("=" * 60)
        log("\nOutputs created:")
        log(f"  1. {theta_output_path}")
        log(f"  2. {prob_output_path}")
        log("\nDecision D069 implemented: Dual-scale trajectory data")
        log("  - Theta scale: IRT ability units for technical audience")
        log("  - Probability scale: 0-1 range for general audience")
        log("\nContinuous TSVR: Individual-level data preserves real time variability")
        log(f"  - {len(df_theta_plot)} individual observations")
        log(f"  - TSVR range: {df_theta_plot['TSVR_hours'].min():.1f} - {df_theta_plot['TSVR_hours'].max():.1f} hours")

        # Print summary statistics per domain
        log("\nTheta by domain:")
        for domain in sorted(df_theta_plot['domain'].unique()):
            domain_data = df_theta_plot[df_theta_plot['domain'] == domain]
            log(f"  {domain}: n={len(domain_data)}, theta mean={domain_data['theta'].mean():.3f}, std={domain_data['theta'].std():.3f}")

        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("\nFull error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
