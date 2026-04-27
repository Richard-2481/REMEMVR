#!/usr/bin/env python3
"""
Step 07: Fit Parallel LMMs and Compare AIC

Fits parallel LMMs for IRT theta, Full CTT, Purified CTT scores to test whether
Purified CTT yields better model fit (lower AIC).

Total: 9 models (3 score types x 3 congruence levels)
Formula: z_score ~ recip_TSVR + log_TSVR + (recip_TSVR | UID)
         (Recip+Log two-process forgetting per RQ 5.4.1 ROOT cascade)

UPDATED 2025-12-09: Changed from Log-only (TSVR_hours) to Recip+Log two-process
                    per RQ 5.4.1 ROOT cascade. Random slopes on recip_TSVR.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_fit_lmms.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 07: Fit Parallel LMMs and Compare AIC")

        # Load standardized scores
        input_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        log(f"[LOAD] Reading {input_path}")
        standardized_scores = pd.read_csv(input_path, encoding='utf-8')
        log(f"[LOADED] {len(standardized_scores)} rows")

        # Extract UID from composite_ID
        standardized_scores['UID'] = standardized_scores['composite_ID'].str.split('_').str[0]
        log(f"[CREATED] UID column (e.g., {standardized_scores['UID'].iloc[0]})")

        # Add time transformations for two-process forgetting model
        log(f"[TRANSFORM] Creating recip_TSVR and log_TSVR transformations")
        standardized_scores['recip_TSVR'] = 1.0 / (standardized_scores['TSVR_hours'] + 1)  # Rapid component
        standardized_scores['log_TSVR'] = np.log1p(standardized_scores['TSVR_hours'])      # Slow component
        log(f"[CREATED] recip_TSVR (rapid): 1/(t+1), range [{standardized_scores['recip_TSVR'].min():.3f}, {standardized_scores['recip_TSVR'].max():.3f}]")
        log(f"[CREATED] log_TSVR (slow): log(t+1), range [{standardized_scores['log_TSVR'].min():.3f}, {standardized_scores['log_TSVR'].max():.3f}]")

        # Define dimensions and score types
        dimensions = ['common', 'congruent', 'incongruent']
        score_types = {
            'IRT': 'theta',
            'Full': 'ctt_full',
            'Purified': 'ctt_purified'
        }

        # Store results
        results = []
        model_summaries = {
            'theta': [],
            'full': [],
            'purified': []
        }

        # Fit models
        log(f"[ANALYSIS] Fitting 9 LMMs (3 dimensions x 3 score types)")
        log(f"[NOTE] Formula: score ~ recip_TSVR + log_TSVR + (recip_TSVR | UID)")
        log(f"[NOTE] Recip+Log two-process forgetting per RQ 5.4.1 ROOT cascade")
        log(f"[NOTE] Will attempt ~recip_TSVR random slopes, fallback to ~1 if singular")

        for dimension in dimensions:
            log(f"\n[DIMENSION] {dimension.capitalize()}")

            aic_values = {}

            for score_label, score_prefix in score_types.items():
                # Construct column name
                score_col = f'z_{score_prefix}_{dimension}'

                log(f"  [MODEL] {score_label} ({score_col})")

                # Prepare data with Recip+Log transformations
                exog = sm.add_constant(standardized_scores[['recip_TSVR', 'log_TSVR']])
                exog_random = standardized_scores[['recip_TSVR']]

                # Try fitting with random slopes first (per ROOT specification)
                try:
                    log(f"    [ATTEMPT 1] Fitting with ~recip_TSVR random slopes...")
                    model = MixedLM(
                        endog=standardized_scores[score_col],
                        exog=exog,
                        groups=standardized_scores['UID'],
                        exog_re=exog_random,
                        missing='drop'
                    )

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        result = model.fit(reml=False, method='powell', maxiter=1000)

                    # Check for singular matrix or non-convergence
                    if not result.converged or any('singular' in str(w).lower() for w in result.warnings if result.warnings):
                        raise ValueError("Singular matrix or convergence failure")

                    aic = result.aic
                    aic_values[score_label] = aic

                    log(f"    [SUCCESS] Random slopes model converged")
                    log(f"    AIC = {aic:.2f}")
                    log(f"    Log-likelihood: {result.llf:.2f}")

                    # Store summary
                    re_spec = "~recip_TSVR"
                    if score_label == 'IRT':
                        model_summaries['theta'].append(f"\n{'='*80}\n{dimension.upper()} - IRT Theta (RE: {re_spec})\n{'='*80}\n{result.summary()}")
                    elif score_label == 'Full':
                        model_summaries['full'].append(f"\n{'='*80}\n{dimension.upper()} - Full CTT (RE: {re_spec})\n{'='*80}\n{result.summary()}")
                    elif score_label == 'Purified':
                        model_summaries['purified'].append(f"\n{'='*80}\n{dimension.upper()} - Purified CTT (RE: {re_spec})\n{'='*80}\n{result.summary()}")

                except Exception as e:
                    # Fallback to random intercepts only
                    log(f"    [FALLBACK] Random slopes failed ({str(e)[:50]}...), trying ~1")

                    try:
                        model = MixedLM(
                            endog=standardized_scores[score_col],
                            exog=exog,
                            groups=standardized_scores['UID'],
                            missing='drop'
                        )

                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore')
                            result = model.fit(reml=False, method='powell', maxiter=1000)

                        aic = result.aic
                        aic_values[score_label] = aic

                        log(f"    [SUCCESS] Random intercepts model converged")
                        log(f"    AIC = {aic:.2f}")
                        log(f"    Converged: {result.converged}")
                        log(f"    Log-likelihood: {result.llf:.2f}")

                        # Store summary
                        re_spec = "~1"
                        if score_label == 'IRT':
                            model_summaries['theta'].append(f"\n{'='*80}\n{dimension.upper()} - IRT Theta (RE: {re_spec})\n{'='*80}\n{result.summary()}")
                        elif score_label == 'Full':
                            model_summaries['full'].append(f"\n{'='*80}\n{dimension.upper()} - Full CTT (RE: {re_spec})\n{'='*80}\n{result.summary()}")
                        elif score_label == 'Purified':
                            model_summaries['purified'].append(f"\n{'='*80}\n{dimension.upper()} - Purified CTT (RE: {re_spec})\n{'='*80}\n{result.summary()}")

                    except Exception as e2:
                        log(f"    [ERROR] Random intercepts also failed: {str(e2)}")
                        log(f"    [FALLBACK] Using OLS estimate for AIC comparison")

                        # Final fallback to OLS
                        from statsmodels.regression.linear_model import OLS
                        ols_model = OLS(standardized_scores[score_col], exog, missing='drop')
                        ols_result = ols_model.fit()
                        aic = ols_result.aic

                        aic_values[score_label] = aic
                        log(f"    AIC (OLS fallback) = {aic:.2f}")

                        # Store OLS summary
                        if score_label == 'IRT':
                            model_summaries['theta'].append(f"\n{'='*80}\n{dimension.upper()} - IRT Theta (OLS FALLBACK)\n{'='*80}\n{ols_result.summary()}")
                        elif score_label == 'Full':
                            model_summaries['full'].append(f"\n{'='*80}\n{dimension.upper()} - Full CTT (OLS FALLBACK)\n{'='*80}\n{ols_result.summary()}")
                        elif score_label == 'Purified':
                            model_summaries['purified'].append(f"\n{'='*80}\n{dimension.upper()} - Purified CTT (OLS FALLBACK)\n{'='*80}\n{ols_result.summary()}")

            # Compute delta AIC (Purified - Full)
            delta_aic = aic_values['Purified'] - aic_values['Full']
            improvement = "Yes" if abs(delta_aic) > 2 else "No"

            log(f"  [SUMMARY] AIC comparison:")
            log(f"    IRT:      {aic_values['IRT']:.2f}")
            log(f"    Full:     {aic_values['Full']:.2f}")
            log(f"    Purified: {aic_values['Purified']:.2f}")
            log(f"    Delta (Purified - Full): {delta_aic:+.2f}")
            log(f"    Improvement (|delta| > 2): {improvement}")

            results.append({
                'dimension': dimension.capitalize(),
                'AIC_IRT': aic_values['IRT'],
                'AIC_Full': aic_values['Full'],
                'AIC_Purified': aic_values['Purified'],
                'delta_AIC_Full_Purified': delta_aic,
                'improvement': improvement,
                'N': len(standardized_scores)
            })

        # Create comparison DataFrame
        lmm_comparison = pd.DataFrame(results)

        # Save comparison table
        output_path = RQ_DIR / "data" / "step07_lmm_model_comparison.csv"
        log(f"\n[SAVE] Writing {output_path}")
        lmm_comparison.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {len(lmm_comparison)} rows, {len(lmm_comparison.columns)} columns")

        # Save model summaries
        for score_type, summaries in [('theta', 'theta'), ('full', 'full'), ('purified', 'purified')]:
            summary_path = RQ_DIR / "data" / f"step07_lmm_summaries_{score_type}.txt"
            log(f"[SAVE] Writing {summary_path}")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(model_summaries[score_type]))
            log(f"[SAVED] {score_type} model summaries")

        log("\n[SUCCESS] Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
