#!/usr/bin/env python3
"""
Step 03: Fit Cross-Classified LMM with 3-Way Interaction

Purpose:
- Fit cross-classified Linear Mixed Model testing Time × Difficulty_c × paradigm interaction
- Random effects: (Time | UID) for participant-level trajectories
- Note: Item random effects (1 | Item) may need to be omitted for convergence

Formula: Response ~ Time * Difficulty_c * C(paradigm)
Random effects: ~Time (random intercept + slope for Time)
Groups: UID

Inputs:
- data/step02_lmm_input.csv (composite_ID, UID, Item, Response, paradigm, Difficulty_c, Time)

Outputs:
- data/step03_lmm_model_summary.txt (full model summary)
- data/step03_fixed_effects.csv (coefficients with dual p-values per D068)
- data/step03_random_effects.csv (variance components)
- data/step03_lmm_model.pkl (saved MixedLMResults object)

Validation:
- Model convergence
- LMM assumptions (normality, homoscedasticity, etc.)
- Hypothesis test (3-way interaction terms present, dual p-values per D068)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.regression.mixed_linear_model import MixedLMResults
from tools.validation import (
    validate_lmm_convergence,
    validate_lmm_assumptions_comprehensive,
    validate_hypothesis_test_dual_pvalues
)

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_fit_lmm.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def main():
    try:
        log("[START] Step 03: Fit Cross-Classified LMM")

        # =====================================================================
        # STEP 1: Load LMM Input Data
        # =====================================================================
        log("[LOAD] Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step02_lmm_input.csv"
        df = pd.read_csv(input_path)
        log(f"[LOADED] {len(df)} rows, {len(df.columns)} columns")
        log(f"[INFO] Columns: {list(df.columns)}")

        # Drop rows where Response is NaN
        n_before = len(df)
        df = df.dropna(subset=['Response'])
        n_after = len(df)
        n_dropped = n_before - n_after
        log(f"[INFO] Dropped {n_dropped} rows with missing Response ({n_after} rows remaining)")

        # Summary statistics
        log(f"[INFO] Participants: {df['UID'].nunique()}")
        log(f"[INFO] Items: {df['Item'].nunique()}")
        log(f"[INFO] Observations: {len(df)}")
        log(f"[INFO] Paradigms: {df['paradigm'].unique()}")
        log(f"[INFO] Response mean: {df['Response'].mean():.3f}")
        log(f"[INFO] Difficulty_c range: [{df['Difficulty_c'].min():.3f}, {df['Difficulty_c'].max():.3f}]")
        log(f"[INFO] Difficulty_c mean: {df['Difficulty_c'].mean():.6f}")
        log(f"[INFO] Time range: [{df['Time'].min():.1f}, {df['Time'].max():.1f}] hours")

        # =====================================================================
        # STEP 2: Fit Cross-Classified LMM
        # =====================================================================
        log("[ANALYSIS] Fitting cross-classified LMM...")
        log("[INFO] Formula: Response ~ Time * Difficulty_c * C(paradigm)")
        log("[INFO] Random effects: ~Time (random intercept + slope)")
        log("[INFO] Groups: UID (participant-level)")

        # Create categorical paradigm
        df['paradigm'] = pd.Categorical(df['paradigm'])

        # Define formula
        formula = "Response ~ Time * Difficulty_c * C(paradigm)"

        # Fit model with convergence strategy
        convergence_strategies = [
            {'re_formula': '~Time', 'description': 'Random intercept + slope for Time'},
            {'re_formula': '1', 'description': 'Random intercepts only'},
        ]

        model = None
        for i, strategy in enumerate(convergence_strategies, 1):
            log(f"[ATTEMPT {i}/{len(convergence_strategies)}] {strategy['description']}")

            try:
                # Fit model
                model = MixedLM.from_formula(
                    formula=formula,
                    groups=df['UID'],
                    re_formula=strategy['re_formula'],
                    data=df
                )

                result = model.fit(reml=False, maxiter=200, method='lbfgs')

                # Check convergence
                if result.converged:
                    log(f"[SUCCESS] Model converged with strategy {i}")
                    break
                else:
                    log(f"[WARNING] Model did not converge with strategy {i}")
                    if i < len(convergence_strategies):
                        log(f"[INFO] Trying next strategy...")
                        continue
                    else:
                        log(f"[ERROR] All strategies exhausted, model did not converge")
                        raise ValueError("Model failed to converge with all strategies")

            except Exception as e:
                log(f"[ERROR] Strategy {i} failed: {str(e)}")
                if i < len(convergence_strategies):
                    log(f"[INFO] Trying next strategy...")
                    continue
                else:
                    log(f"[ERROR] All strategies exhausted")
                    raise

        if model is None or not result.converged:
            raise ValueError("Failed to fit LMM model")

        log(f"[INFO] Model converged: {result.converged}")
        log(f"[INFO] Log-likelihood: {result.llf:.2f}")
        log(f"[INFO] AIC: {result.aic:.2f}")
        log(f"[INFO] BIC: {result.bic:.2f}")

        # =====================================================================
        # STEP 3: Extract Fixed Effects with Dual P-Values (D068)
        # =====================================================================
        log("[EXTRACT] Extracting fixed effects...")

        # Get fixed effects
        fixed_effects = pd.DataFrame({
            'term': result.params.index,
            'estimate': result.params.values,
            'SE': result.bse.values,
            'z_value': result.tvalues.values,
            'p_uncorrected': result.pvalues.values
        })

        # Apply Bonferroni correction
        n_tests = len(fixed_effects)
        alpha = 0.05
        alpha_bonf = alpha / n_tests
        fixed_effects['p_bonferroni'] = fixed_effects['p_uncorrected'] * n_tests
        fixed_effects['p_bonferroni'] = fixed_effects['p_bonferroni'].clip(upper=1.0)

        log(f"[INFO] Fixed effects: {len(fixed_effects)} terms")
        log(f"[INFO] Bonferroni correction: n_tests = {n_tests}, alpha_bonf = {alpha_bonf:.4f}")

        # =====================================================================
        # STEP 4: Extract Random Effects
        # =====================================================================
        log("[EXTRACT] Extracting random effects...")

        # Get random effects variance components
        random_effects = pd.DataFrame({
            'component': ['UID_intercept', 'UID_slope_Time', 'Residual'],
            'variance': [
                result.cov_re.iloc[0, 0] if result.cov_re.shape[0] > 0 else 0.0,
                result.cov_re.iloc[1, 1] if result.cov_re.shape[0] > 1 else 0.0,
                result.scale
            ]
        })
        random_effects['SD'] = np.sqrt(random_effects['variance'])

        log(f"[INFO] Random effects variance components:")
        for _, row in random_effects.iterrows():
            log(f"  {row['component']}: variance = {row['variance']:.4f}, SD = {row['SD']:.4f}")

        # =====================================================================
        # STEP 5: Save Outputs
        # =====================================================================
        log("[SAVE] Saving outputs...")

        # Save model summary
        summary_path = RQ_DIR / "data" / "step03_lmm_model_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(str(result.summary()))
        log(f"[SAVED] {summary_path}")

        # Save fixed effects
        fixed_effects_path = RQ_DIR / "data" / "step03_fixed_effects.csv"
        fixed_effects.to_csv(fixed_effects_path, index=False, encoding='utf-8')
        log(f"[SAVED] {fixed_effects_path}")

        # Save random effects
        random_effects_path = RQ_DIR / "data" / "step03_random_effects.csv"
        random_effects.to_csv(random_effects_path, index=False, encoding='utf-8')
        log(f"[SAVED] {random_effects_path}")

        # Save model object
        model_path = RQ_DIR / "data" / "step03_lmm_model.pkl"
        result.save(str(model_path))
        log(f"[SAVED] {model_path}")

        # =====================================================================
        # STEP 6: Run Validation Tools
        # =====================================================================
        log("[VALIDATION] Running validate_lmm_convergence...")
        validation_result = validate_lmm_convergence(lmm_result=result)
        validation_result = validate_lmm_convergence(lmm_result=result)
        validation_result = validate_lmm_convergence(lmm_result=result)
        validation_result = validate_lmm_convergence(lmm_result=result)
        validation_result = validate_lmm_convergence(lmm_result=result)
        log(f"[VALIDATION] Convergence result: {validation_result}")

        if not validation_result.get('all_valid', False):
            log(f"[WARNING] Convergence validation issues: {validation_result}")

        log("[VALIDATION] Running validate_lmm_assumptions_comprehensive...")
        assumptions_result = validate_lmm_assumptions_comprehensive(
            lmm_result=result,
            data=df,
            output_dir=RQ_DIR / "data",
            acf_lag1_threshold=0.1,
            alpha=0.05
        )
        log(f"[VALIDATION] Assumptions result: {assumptions_result}")

        if not assumptions_result.get('all_valid', False):
            log(f"[WARNING] Assumptions validation issues: {assumptions_result}")

        log("[VALIDATION] Running validate_hypothesis_test_dual_pvalues...")
        hypothesis_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=fixed_effects,
            required_terms=['Time:Difficulty_c:C(paradigm)[T.IFR]', 'Time:Difficulty_c:C(paradigm)[T.IRE]'],
            alpha_bonferroni=0.0033
        )
        log(f"[VALIDATION] Hypothesis test result: {hypothesis_result}")

        if not hypothesis_result.get('valid', False):
            log(f"[ERROR] Hypothesis test validation failed: {hypothesis_result}")
            raise ValueError(f"Hypothesis test validation failed: {hypothesis_result.get('message', 'Unknown error')}")

        log("[SUCCESS] Step 03 complete")
        return 0

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK]")
        traceback.print_exc()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        return 1

if __name__ == "__main__":
    sys.exit(main())
