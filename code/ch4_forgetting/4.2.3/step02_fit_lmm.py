#!/usr/bin/env python3
"""step02_fit_lmm: Fit Linear Mixed Model testing whether age effects on forgetting rate vary by"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_fit_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Fit LMM with 3-Way Age x Domain x Time Interaction")
        # Load Input Data

        log("Loading LMM input data...")
        input_file = RQ_DIR / "data/step01_lmm_input.csv"
        lmm_input = pd.read_csv(input_file)
        log(f"{input_file.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

        # Log data structure for transparency
        log(f"Columns: {list(lmm_input.columns)}")
        log(f"Unique UIDs: {lmm_input['UID'].nunique()}")
        log(f"Domains: {sorted(lmm_input['domain'].unique())}")
        log(f"Tests: {sorted(lmm_input['test'].unique())}")
        log(f"Age range: {lmm_input['age'].min():.1f} - {lmm_input['age'].max():.1f}")
        log(f"Age_c range: {lmm_input['Age_c'].min():.2f} - {lmm_input['Age_c'].max():.2f}")
        # Run Analysis Tool

        log("Fitting LMM with 3-way Age x Domain x Time interaction...")
        log("Formula: theta ~ TSVR_hours + log_TSVR + Age_c + domain + TSVR_hours:Age_c + log_TSVR:Age_c + TSVR_hours:domain + log_TSVR:domain + Age_c:domain + TSVR_hours:Age_c:domain + log_TSVR:Age_c:domain")
        log("Random effects: Random intercepts only (convergence fix)")
        log("REML=False (ML estimation for model comparison)")
        log("Random slopes caused convergence issues with 2-domain data")

        # Define formula (complex 3-way interaction structure)
        # Main effects: TSVR_hours, log_TSVR, Age_c, domain
        # 2-way interactions: TSVR:Age, log_TSVR:Age, TSVR:domain, log_TSVR:domain, Age:domain
        # 3-way interactions: TSVR:Age:domain, log_TSVR:Age:domain
        formula = "theta ~ TSVR_hours + log_TSVR + Age_c + domain + TSVR_hours:Age_c + log_TSVR:Age_c + TSVR_hours:domain + log_TSVR:domain + Age_c:domain + TSVR_hours:Age_c:domain + log_TSVR:Age_c:domain"

        # Call analysis tool
        # Note: Using fit_lmm_trajectory (not fit_lmm_trajectory_tsvr) because
        # our data is already merged and in the correct format
        # CONVERGENCE FIX: Use random intercepts only (re_formula=None) instead of
        # random slopes. The complex fixed effects structure with reduced sample size
        # (800 vs 1200 rows due to When exclusion) caused convergence failures.
        lmm_model = fit_lmm_trajectory(
            data=lmm_input,           # Merged LMM input with all variables
            formula=formula,
            groups="UID",             # Grouping variable for random effects
            re_formula=None,          # Random intercepts only (convergence fix)
            reml=False                # ML estimation (required for LRT in step02c)
        )

        log("LMM fitting complete")
        log(f"Model converged: {lmm_model.converged}")
        log(f"Number of observations: {lmm_model.nobs}")
        log(f"Number of groups: {len(lmm_model.model.group_labels)}")
        log(f"Log-likelihood: {lmm_model.llf:.2f}")
        log(f"AIC: {lmm_model.aic:.2f}")
        log(f"BIC: {lmm_model.bic:.2f}")
        # Save Analysis Outputs
        # These outputs will be used by: step02b (assumption validation), step02c (model selection), step03 (interaction extraction)

        # Output 1: Model object (pickle)
        model_file = RQ_DIR / "data/step02_lmm_model.pkl"
        log(f"Saving model object to {model_file.name}...")
        # IMPORTANT: Use statsmodels.save() method, NOT pickle.dump()
        # Reason: pickle.dump() causes patsy/eval errors on load
        lmm_model.save(str(model_file))
        log(f"{model_file.name} (statsmodels MixedLMResults object)")

        # Output 2: Model summary (text)
        summary_file = RQ_DIR / "data/step02_lmm_summary.txt"
        log(f"Saving model summary to {summary_file.name}...")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(str(lmm_model.summary()))
        log(f"{summary_file.name} (full model summary with fixed effects table)")

        # Output 3: Fixed effects table (CSV)
        fixed_effects_file = RQ_DIR / "data/step02_fixed_effects.csv"
        log(f"Saving fixed effects table to {fixed_effects_file.name}...")

        # Extract fixed effects from model
        fe_summary = lmm_model.summary().tables[1]  # Fixed effects table
        fe_df = pd.DataFrame({
            'term': fe_summary.index,
            'estimate': lmm_model.params,
            'se': lmm_model.bse,
            'z': lmm_model.tvalues,
            'p': lmm_model.pvalues,
            'CI_lower': lmm_model.conf_int()[0],
            'CI_upper': lmm_model.conf_int()[1]
        })

        fe_df.to_csv(fixed_effects_file, index=False, encoding='utf-8')
        log(f"{fixed_effects_file.name} ({len(fe_df)} rows, {len(fe_df.columns)} cols)")
        log(f"Fixed effects include {len(fe_df)} terms")

        # Log key interaction terms for verification
        interaction_terms = fe_df[fe_df['term'].str.contains(':Age_c:domain', na=False)]
        log(f"3-way interaction terms found: {len(interaction_terms)}")
        for _, row in interaction_terms.iterrows():
            log(f"  - {row['term']}: beta={row['estimate']:.4f}, p={row['p']:.4f}")
        # Run Validation Tool
        # Validates: Model convergence status and warnings
        # Threshold: Model must have converged successfully

        log("Running validate_lmm_convergence...")
        validation_result = validate_lmm_convergence(
            lmm_result=lmm_model
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check validation passed
        if isinstance(validation_result, dict) and not validation_result.get('converged', False):
            log("Model did not converge")
            log(f"Message: {validation_result.get('message', 'Unknown error')}")
            sys.exit(1)

        log("Model converged successfully")

        # Additional manual checks per validation criteria
        log("Checking additional criteria...")

        # Check 1: No singular fit (random effects variance > 0)
        re_variance = lmm_model.cov_re.iloc[0, 0]  # Random intercept variance
        if re_variance > 0:
            log(f"Random effects variance > 0 ({re_variance:.4f})")
        else:
            log(f"Singular fit detected (random effects variance = {re_variance})")
            sys.exit(1)

        # Check 2: All fixed effects have finite estimates (no NaN/Inf)
        if fe_df['estimate'].isna().any() or fe_df['estimate'].isin([float('inf'), float('-inf')]).any():
            log("Fixed effects contain NaN or Inf values")
            sys.exit(1)
        log("All fixed effects have finite estimates")

        # Check 3: 3-way interaction terms present
        required_interactions = ['TSVR_hours:Age_c:domain', 'log_TSVR:Age_c:domain']
        found_interactions = []
        for interaction in required_interactions:
            if fe_df['term'].str.contains(interaction, na=False).any():
                found_interactions.append(interaction)

        if len(found_interactions) == len(required_interactions):
            log(f"All required 3-way interaction terms present: {found_interactions}")
        else:
            missing = set(required_interactions) - set(found_interactions)
            log(f"Missing 3-way interaction terms: {missing}")
            sys.exit(1)

        log("Step 02 complete")
        log("Model ready for assumption validation (step02b) and model selection (step02c)")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
