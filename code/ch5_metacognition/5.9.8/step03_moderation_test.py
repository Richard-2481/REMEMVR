#!/usr/bin/env python3
"""Test Moderation - Mixed-Effects Regression: Test whether schema condition moderates the confidence-accuracy relationship using"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.formula.api as smf

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.8
LOG_FILE = RQ_DIR / "logs" / "step03_moderation_test.log"
INPUT_FILE = RQ_DIR / "data" / "step01_reshaped_long.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step03_moderation_test.csv"
DIAGNOSTICS_FILE = RQ_DIR / "data" / "step03_moderation_diagnostics.txt"

# Key timepoints for analysis
KEY_TIMEPOINTS = ['T1', 'T2', 'T4']

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Test Moderation with Mixed-Effects Model")
        # Load and Prepare Data
        log("Loading long-format data...")
        df = pd.read_csv(INPUT_FILE)
        log(f"{len(df)} rows")

        log("Filtering to key timepoints...")
        df_key = df[df['test'].isin(KEY_TIMEPOINTS)].copy()
        log(f"{len(df_key)} rows retained")

        # Verify expected row count
        expected_rows = 100 * 3 * 3  # 100 UIDs x 3 timepoints x 3 schema conditions
        if len(df_key) != expected_rows:
            log(f"Expected {expected_rows} rows, got {len(df_key)}")

        # Standardize confidence for interpretability
        log("Standardizing confidence scores...")
        df_key['confidence_z'] = zscore(df_key['theta_confidence'])
        log(f"Standardized confidence: mean = {df_key['confidence_z'].mean():.6f}, SD = {df_key['confidence_z'].std():.6f}")

        # Create dummy variables (Common as reference category)
        log("Creating dummy variables (Common as reference)...")
        df_key['schema_Congruent'] = (df_key['schema_condition'] == 'Congruent').astype(int)
        df_key['schema_Incongruent'] = (df_key['schema_condition'] == 'Incongruent').astype(int)

        log(f"Congruent: {df_key['schema_Congruent'].sum()} rows")
        log(f"Incongruent: {df_key['schema_Incongruent'].sum()} rows")
        log(f"Common (reference): {(df_key['schema_condition'] == 'Common').sum()} rows")
        # Fit Mixed-Effects Model
        log("Fitting mixed-effects model...")
        log("Formula: theta_accuracy ~ confidence_z + schema_Congruent + schema_Incongruent + confidence_z:schema_Congruent + confidence_z:schema_Incongruent")
        log("Random effects: random intercepts by UID (1|UID)")

        formula = '''theta_accuracy ~ confidence_z + schema_Congruent + schema_Incongruent +
                     confidence_z:schema_Congruent + confidence_z:schema_Incongruent'''

        model = smf.mixedlm(
            formula=formula,
            data=df_key,
            groups=df_key['UID']
        )

        log("Running optimization (REML method)...")
        result = model.fit(reml=True)  # Restricted Maximum Likelihood
        log("Model fitting complete")
        # Extract Results
        log("Extracting fixed effects table...")

        # Fixed effects
        fe_results = pd.DataFrame({
            'term': result.params.index.tolist(),
            'beta': result.params.values,
            'SE': result.bse.values,
            't_value': result.tvalues.values,
            'p_value': result.pvalues.values,
            'CI_lower': result.conf_int()[0].values,
            'CI_upper': result.conf_int()[1].values
        })

        log(f"Fixed effects: {len(fe_results)} terms")
        for _, row in fe_results.iterrows():
            log(f"  {row['term']:30s}: beta = {row['beta']:7.4f}, SE = {row['SE']:.4f}, t = {row['t_value']:6.2f}, p = {row['p_value']:.4f}")

        # Model diagnostics
        log("Extracting model diagnostics...")
        diagnostics = []
        diagnostics.append("=" * 80)
        diagnostics.append("MIXED-EFFECTS MODEL DIAGNOSTICS")
        diagnostics.append("=" * 80)
        diagnostics.append("")
        # Get number of groups from model object (not result)
        n_groups = len(df_key['UID'].unique())

        diagnostics.append(f"Converged: {result.converged}")
        diagnostics.append(f"Observations: {result.nobs}")
        diagnostics.append(f"Groups (UIDs): {n_groups}")
        diagnostics.append(f"AIC: {result.aic:.2f}")
        diagnostics.append(f"BIC: {result.bic:.2f}")
        diagnostics.append(f"Log-Likelihood: {result.llf:.2f}")
        diagnostics.append("")
        diagnostics.append("Random Effects Variance:")
        diagnostics.append(f"  Group (UID): {result.cov_re.values[0, 0]:.4f}")
        diagnostics.append(f"  Residual: {result.scale:.4f}")
        diagnostics.append("")
        diagnostics.append("Fixed Effects Summary:")
        diagnostics.append(str(result.summary().tables[1]))
        diagnostics.append("")
        diagnostics.append("=" * 80)

        log(f"Model converged: {result.converged}")
        log(f"Observations: {result.nobs}, Groups: {n_groups}")
        log(f"AIC = {result.aic:.2f}, BIC = {result.bic:.2f}")
        # Validate Results
        log("Checking model results...")

        errors = []

        # Check convergence
        if not result.converged:
            errors.append("Model did not converge")
        else:
            log("Model converged successfully")

        # Check expected row count (may include Group Var from random effects)
        if len(fe_results) < 6:
            errors.append(f"Expected at least 6 fixed effects terms, got {len(fe_results)}")
        elif len(fe_results) == 6:
            log("Fixed effects: 6 terms (Intercept + 5 predictors)")
        else:
            log(f"Fixed effects table has {len(fe_results)} rows (includes Group Var)")

        # Check observations and groups
        if result.nobs != 900:
            log(f"Expected 900 observations, got {result.nobs}")
        else:
            log("Observations: 900")

        if n_groups != 100:
            log(f"Expected 100 groups (UIDs), got {n_groups}")
        else:
            log("Groups: 100 (UIDs)")

        # Check SE values positive
        if (fe_results['SE'] <= 0).any():
            errors.append("Found non-positive SE values")
        else:
            log("All SE values positive")

        # Check main effect of confidence_z
        conf_main = fe_results[fe_results['term'] == 'confidence_z']
        if not conf_main.empty:
            if conf_main['beta'].values[0] > 0 and conf_main['p_value'].values[0] < 0.05:
                log("Main effect of confidence_z positive and significant (expected)")
            else:
                log(f"Main effect of confidence_z: beta = {conf_main['beta'].values[0]:.4f}, p = {conf_main['p_value'].values[0]:.4f}")
        else:
            errors.append("confidence_z term not found in results")

        # Check interaction terms
        interaction_terms = fe_results[fe_results['term'].str.contains(':', regex=False)]
        if len(interaction_terms) == 2:
            log(f"Interaction terms: {len(interaction_terms)}")
            for _, row in interaction_terms.iterrows():
                if row['p_value'] > 0.05:
                    log(f"{row['term']}: p = {row['p_value']:.4f} > 0.05 (non-significant, supports NULL moderation)")
                else:
                    log(f"{row['term']}: p = {row['p_value']:.4f} < 0.05 (significant, contradicts NULL hypothesis)")

            if (interaction_terms['p_value'] > 0.05).all():
                log("All interaction terms non-significant -> Schema does NOT moderate relationship")
            else:
                log("Some interaction terms significant -> Schema MAY moderate relationship")
        else:
            errors.append(f"Expected 2 interaction terms, found {len(interaction_terms)}")
        # Save Outputs
        if errors:
            log("FAIL - Errors detected:")
            for error in errors:
                log(f"  - {error}")
            raise ValueError(f"Validation failed with {len(errors)} error(s)")

        log("PASS - All checks passed")

        fe_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"Fixed effects table: {OUTPUT_FILE}")

        with open(DIAGNOSTICS_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(diagnostics))
        log(f"Model diagnostics: {DIAGNOSTICS_FILE}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
