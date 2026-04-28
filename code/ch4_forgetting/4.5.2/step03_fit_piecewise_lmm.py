#!/usr/bin/env python3
"""
==============================================================================
Step 03: Fit Piecewise LMM for RQ 5.5.2
==============================================================================
RQ: 5.5.2 - Source-Destination Consolidation (Two-Phase)

Purpose:
    Fit piecewise linear mixed model with 3-way interaction to test whether
    source and destination memories show differential consolidation patterns.

Formula:
    theta ~ Days_within * Segment * LocationType + (1 + Days_within | UID)

    Fixed effects (8 terms):
    - Intercept (Source-Early at Days_within=0)
    - Days_within (slope for Source-Early)
    - Segment[T.Late] (intercept shift for Late phase)
    - LocationType[T.Destination] (intercept shift for Destination)
    - Days_within:Segment[T.Late] (slope difference for Late phase)
    - Days_within:LocationType[T.Destination] (slope difference for Destination)
    - Segment[T.Late]:LocationType[T.Destination] (2-way interaction)
    - Days_within:Segment[T.Late]:LocationType[T.Destination] (3-way, PRIMARY HYPOTHESIS)

Random effects:
    - Random intercepts per UID (baseline individual differences)
    - Random slopes for Days_within per UID (individual forgetting rates)
    - Fallback to intercept-only if random slopes fail to converge

Input:
    - data/step02_lmm_input_long.csv (800 rows: 100 UID x 4 tests x 2 locations)

Output:
    - data/step03_piecewise_lmm_model.pkl (saved MixedLMResults object)
    - data/step03_piecewise_lmm_summary.txt (model summary text)
    - data/step03_lmm_coefficients.csv (fixed effects for downstream steps)
==============================================================================
"""

from pathlib import Path
import sys
import pickle
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# PATH SETUP

RQ_DIR = Path(__file__).resolve().parent.parent  # results/ch5/5.5.2

# Folder conventions 
DATA_DIR = RQ_DIR / "data"
LOGS_DIR = RQ_DIR / "logs"
RESULTS_DIR = RQ_DIR / "results"

# Input
INPUT_FILE = DATA_DIR / "step02_lmm_input_long.csv"

# Outputs
OUTPUT_MODEL = DATA_DIR / "step03_piecewise_lmm_model.pkl"
OUTPUT_SUMMARY = DATA_DIR / "step03_piecewise_lmm_summary.txt"
OUTPUT_COEFFS = DATA_DIR / "step03_lmm_coefficients.csv"
LOG_FILE = LOGS_DIR / "step03_fit_piecewise_lmm.log"

# Create directories if needed
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING

def log(msg: str) -> None:
    """Log message to console and file."""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log file
LOG_FILE.write_text("", encoding="utf-8")


# MAIN EXECUTION

if __name__ == "__main__":
    try:
        log("Step 3: Fit Piecewise LMM")
        # Load Data

        log("Loading LMM input data...")
        df = pd.read_csv(INPUT_FILE)
        log(f"{INPUT_FILE.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Verify expected structure
        expected_cols = ['UID', 'test', 'LocationType', 'theta', 'se',
                        'TSVR_hours', 'Segment', 'Days_within']
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Data summary
        log(f"LocationType distribution: {df['LocationType'].value_counts().to_dict()}")
        log(f"Segment distribution: {df['Segment'].value_counts().to_dict()}")
        log(f"Unique UIDs: {df['UID'].nunique()}")
        log(f"theta range: [{df['theta'].min():.3f}, {df['theta'].max():.3f}]")
        log(f"Days_within range: [{df['Days_within'].min():.3f}, {df['Days_within'].max():.3f}]")
        # Set Treatment Coding

        log("Setting treatment coding...")
        log("LocationType reference = Source")
        log("Segment reference = Early")

        # Set categorical with explicit ordering (first = reference)
        df['LocationType'] = pd.Categorical(df['LocationType'],
                                            categories=['Source', 'Destination'])
        df['Segment'] = pd.Categorical(df['Segment'],
                                       categories=['Early', 'Late'])
        # Fit LMM with Full Random Structure

        log("Fitting piecewise LMM with 3-way interaction...")
        log("Fixed: theta ~ Days_within * Segment * LocationType")
        log("Random: ~Days_within | UID (random intercepts + slopes)")
        log("Estimation: REML=False (ML for AIC comparison)")

        formula = "theta ~ Days_within * Segment * LocationType"
        converged = False
        random_structure_used = None

        # Try full random structure first
        try:
            log("Attempting full random structure: (1 + Days_within | UID)...")

            model = smf.mixedlm(formula, data=df, groups=df['UID'],
                               re_formula="~Days_within")
            lmm_result = model.fit(reml=False, method='lbfgs', maxiter=200)

            if lmm_result.converged:
                log("✓ Full random structure converged successfully")
                converged = True
                random_structure_used = "~Days_within (random intercepts + slopes)"
            else:
                log("Full random structure did not converge, trying fallback...")
                raise RuntimeError("Convergence failed")

        except Exception as e:
            log(f"Full random structure failed: {str(e)[:100]}")
            log("Falling back to intercept-only random structure...")

            try:
                model = smf.mixedlm(formula, data=df, groups=df['UID'],
                                   re_formula="~1")
                lmm_result = model.fit(reml=False, method='lbfgs', maxiter=200)

                if lmm_result.converged:
                    log("✓ Intercept-only random structure converged")
                    converged = True
                    random_structure_used = "~1 (random intercepts only)"
                else:
                    raise RuntimeError("Intercept-only convergence failed")

            except Exception as e2:
                log(f"Both random structures failed: {str(e2)[:100]}")
                raise ValueError(f"LMM failed to converge with any random structure")

        if not converged:
            raise ValueError("LMM did not converge")
        # Extract and Log Results

        log("Model converged successfully")
        log(f"Random structure used: {random_structure_used}")

        # Fixed effects
        log("Fixed effects:")
        fe_params = lmm_result.fe_params
        fe_se = lmm_result.bse_fe
        fe_z = lmm_result.tvalues
        fe_p = lmm_result.pvalues

        for term in fe_params.index:
            coef = fe_params[term]
            se = fe_se[term] if term in fe_se.index else np.nan
            z = fe_z[term] if term in fe_z.index else np.nan
            p = fe_p[term] if term in fe_p.index else np.nan
            log(f"  {term}: β={coef:.4f}, SE={se:.4f}, z={z:.2f}, p={p:.4f}")

        # Model fit
        log(f"Observations: {lmm_result.nobs}")
        log(f"Groups (UIDs): {lmm_result.nobs // 8}")  # 8 obs per UID (4 tests x 2 locations)
        log(f"Log-likelihood: {lmm_result.llf:.2f}")
        log(f"AIC: {lmm_result.aic:.2f}")
        log(f"BIC: {lmm_result.bic:.2f}")

        # Random effects variance
        log("Random effects:")
        log(f"  Variance (Group): {lmm_result.cov_re.iloc[0, 0]:.4f}")
        log(f"  Residual variance (Scale): {lmm_result.scale:.4f}")

        # Check for 8 fixed effects
        n_fe = len(fe_params)
        log(f"Number of fixed effects: {n_fe} (expected 8)")
        if n_fe != 8:
            log(f"Expected 8 fixed effects but got {n_fe}")
        # Save Model

        log(f"Saving model to {OUTPUT_MODEL.name}...")
        with open(OUTPUT_MODEL, 'wb') as f:
            pickle.dump(lmm_result, f)
        log(f"Model pickle ({OUTPUT_MODEL.stat().st_size} bytes)")
        # Save Summary

        log(f"Saving summary to {OUTPUT_SUMMARY.name}...")
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 5.5.2 Piecewise LMM Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Random structure: {random_structure_used}\n")
            f.write(f"Estimation: ML (REML=False)\n")
            f.write(f"Observations: {lmm_result.nobs}\n")
            f.write(f"Groups (UIDs): {lmm_result.nobs // 8}\n")
            f.write(f"Converged: {lmm_result.converged}\n\n")
            f.write("-" * 80 + "\n")
            f.write("MODEL SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(str(lmm_result.summary()))
            f.write("\n\n")
            f.write("-" * 80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("-" * 80 + "\n")
            f.write("Intercept: Source-Early baseline theta at Days_within=0\n")
            f.write("Days_within: Slope for Source-Early (forgetting rate)\n")
            f.write("Segment[T.Late]: Late phase intercept shift from Early\n")
            f.write("LocationType[T.Destination]: Destination intercept shift from Source\n")
            f.write("Days_within:Segment[T.Late]: Slope difference for Late phase\n")
            f.write("Days_within:LocationType[T.Destination]: Slope difference for Destination\n")
            f.write("Segment:LocationType: 2-way intercept interaction\n")
            f.write("Days_within:Segment:LocationType: 3-WAY INTERACTION (PRIMARY HYPOTHESIS)\n")
        log(f"Summary text file")
        # Save Coefficients CSV (for downstream steps)

        log(f"Saving coefficients to {OUTPUT_COEFFS.name}...")
        coef_df = pd.DataFrame({
            'term': fe_params.index,
            'estimate': fe_params.values,
            'SE': [fe_se[t] if t in fe_se.index else np.nan for t in fe_params.index],
            'z_score': [fe_z[t] if t in fe_z.index else np.nan for t in fe_params.index],
            'p_value': [fe_p[t] if t in fe_p.index else np.nan for t in fe_params.index]
        })
        # Add 95% CI
        coef_df['CI_lower'] = coef_df['estimate'] - 1.96 * coef_df['SE']
        coef_df['CI_upper'] = coef_df['estimate'] + 1.96 * coef_df['SE']
        coef_df.to_csv(OUTPUT_COEFFS, index=False)
        log(f"Coefficients CSV ({len(coef_df)} rows)")
        # Validation

        log("Checking model quality...")

        # Check convergence
        if not lmm_result.converged:
            raise ValueError("Model did not converge")
        log("Model converged")

        # Check fixed effects count
        if n_fe != 8:
            log(f"Expected 8 fixed effects, got {n_fe}")
        else:
            log("8 fixed effects as expected")

        # Check no NaN in coefficients
        nan_count = coef_df['estimate'].isna().sum()
        if nan_count > 0:
            raise ValueError(f"NaN in coefficients: {nan_count}")
        log("No NaN values in coefficients")

        # Check positive variance
        var_re = lmm_result.cov_re.iloc[0, 0]
        if var_re <= 0:
            log(f"Non-positive random effect variance: {var_re}")
        else:
            log(f"Positive random effect variance: {var_re:.4f}")
        # SUCCESS

        log("Step 03 complete")
        log(f"Model saved to: {OUTPUT_MODEL}")
        log(f"Summary saved to: {OUTPUT_SUMMARY}")
        log(f"Coefficients saved to: {OUTPUT_COEFFS}")

        # Report key finding
        interaction_term = "Days_within:Segment[T.Late]:LocationType[T.Destination]"
        if interaction_term in fe_p.index:
            p_val = fe_p[interaction_term]
            estimate = fe_params[interaction_term]
            log(f"3-way interaction: β={estimate:.4f}, p={p_val:.4f}")
            if p_val < 0.025:
                log("Interaction SIGNIFICANT at Bonferroni α=0.025")
            else:
                log("Interaction NOT significant at Bonferroni α=0.025")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
