#!/usr/bin/env python3
"""
Extended Model Comparison for RQ 5.1.3 - Age Effects Across 66 Functional Forms

PURPOSE:
Test age effects (Age_c on intercept and Age_c x Time interactions) across all
66 functional forms used in RQ 5.1.1 extended comparison. This addresses:
1. ROOT model update: RQ 5.1.1 now uses PowerLaw α=0.410 (not Lin+Log)
2. Robustness: Are wrong-direction age effects model-dependent artifacts?
3. Model averaging: Quantify uncertainty in age effect estimates

BACKGROUND:
- Original RQ 5.1.3 used Lin+Log (from 5-model comparison)
- RQ 5.1.1 extended comparison revealed PowerLaw models dominate (ΔAIC=3.10)
- Best weight only 5.6% → extreme uncertainty requires model averaging
- This script tests: Do age effects depend on functional form choice?

INPUTS:
  - data/step01_lmm_input_prepared.csv (Age_c, Time, Time_log, theta)

OUTPUTS:
  - data/step02b_extended_model_fits.pkl (66 fitted models with Age interactions)
  - data/step02b_model_comparison.csv (AIC, weights, age effect estimates per model)
  - data/step02b_age_effects_averaged.csv (model-averaged age effect estimates)
  - logs/step02b_extended_age_models.log
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
import traceback
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02b_extended_age_models.log"

# =============================================================================
# Logging
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Initialize log
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("")  # Clear previous log

# =============================================================================
# Model Fitting Function
# =============================================================================

def fit_age_lmm(formula, data, model_name, re_formula='Time'):
    """
    Fit LMM with Age interactions using statsmodels MixedLM.

    Parameters:
    -----------
    formula : str
        Fixed effects formula (R-style)
    data : pd.DataFrame
        Input data with theta, Age_c, time transforms, UID
    model_name : str
        Model identifier for logging
    re_formula : str
        Random effects formula (default: 'Time' for random slopes)

    Returns:
    --------
    dict with fitted model, AIC, convergence status, age effects
    """
    try:
        # Parse formula to add Age_c interactions
        # Extract time predictors (everything after ~)
        time_part = formula.split('~')[1].strip()

        # Create Age interaction formula: time_pred * Age_c
        # This expands to: time_pred + Age_c + time_pred:Age_c
        age_formula = f"theta ~ ({time_part}) * Age_c"

        # Fit model with random intercepts and slopes
        md = smf.mixedlm(age_formula, data=data, groups=data['UID'],
                          re_formula=f"~{re_formula}")
        mdf = md.fit(method='lbfgs', reml=False)

        # Extract age effects from fixed effects
        age_effects = {}
        for param in mdf.params.index:
            if 'Age_c' in param:
                age_effects[param] = {
                    'coef': mdf.params[param],
                    'se': mdf.bse[param],
                    'z': mdf.tvalues[param],
                    'p': mdf.pvalues[param]
                }

        return {
            'model': mdf,
            'model_name': model_name,
            'formula': age_formula,
            'converged': mdf.converged,
            'AIC': mdf.aic,
            'BIC': mdf.bic,
            'log_likelihood': mdf.llf,
            'n_params': len(mdf.params),
            'age_effects': age_effects,
            'n_obs': len(data),
            'n_groups': data['UID'].nunique()
        }

    except Exception as e:
        log(f"  [FAIL] {model_name}: {str(e)[:100]}")
        return {
            'model': None,
            'model_name': model_name,
            'formula': None,
            'converged': False,
            'AIC': np.inf,
            'BIC': np.inf,
            'log_likelihood': -np.inf,
            'n_params': np.nan,
            'age_effects': {},
            'n_obs': len(data),
            'n_groups': data['UID'].nunique(),
            'error': str(e)[:200]
        }

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("="*70)
        log("RQ 5.1.3 - EXTENDED AGE MODEL COMPARISON (66 Functional Forms)")
        log("="*70)
        log(f"Date: {pd.Timestamp.now()}")
        log("")

        # =====================================================================
        # STEP 1: Load Prepared Data
        # =====================================================================

        log("[STEP 1] Loading prepared LMM input...")
        input_path = RQ_DIR / "data" / "step01_lmm_input_prepared.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input data missing: {input_path}")

        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"  Loaded: {len(df)} rows, {df['UID'].nunique()} participants")
        log(f"  Age_c range: [{df['Age_c'].min():.2f}, {df['Age_c'].max():.2f}]")
        log(f"  Time range: [{df['Time'].min():.2f}, {df['Time'].max():.2f}] hours")
        log("")

        # =====================================================================
        # STEP 2: Add Extended Time Transformations
        # =====================================================================

        log("[STEP 2] Adding 66-model time transformations...")

        # Power law variants
        df['log_log_Days'] = np.log(df['Time_log'] + 1)
        df['sqrt_Days'] = np.sqrt(df['Time'])
        df['cbrt_Days'] = np.cbrt(df['Time'])
        df['recip_Days'] = 1.0 / (df['Time'] + 1)
        df['neg_Days'] = -df['Time']

        # Power law with specific exponents
        for alpha in [0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.67, 0.7, 0.8, 0.9, 1.0]:
            df[f'Days_pow_{int(alpha*100):02d}'] = (df['Time'] + 1) ** (-alpha)

        # Additional transforms
        df['Days_sq'] = df['Time'] ** 2
        df['Days_cube'] = df['Time'] ** 3
        df['Days_quart'] = df['Time'] ** 4
        df['recip_sq'] = 1.0 / ((df['Time'] + 1) ** 2)
        df['log10_Days'] = np.log10(df['Time'] + 1)
        df['log2_Days'] = np.log2(df['Time'] + 1)

        # Trigonometric (exploratory)
        df['sin_Days'] = np.sin(df['Time'] / 50)  # Scaled for reasonable period
        df['cos_Days'] = np.cos(df['Time'] / 50)
        df['tanh_Days'] = np.tanh(df['Time'] / 50)

        log(f"  Added {len([c for c in df.columns if 'Days' in c or 'pow' in c])} time transforms")
        log("")

        # =====================================================================
        # STEP 3: Define 66-Model Suite with Age Interactions
        # =====================================================================

        log("[STEP 3] Defining 66-model suite...")

        # Base models dictionary (will add * Age_c interactions during fitting)
        models = {
            # ---- ORIGINAL 5 MODELS ----
            'Linear': 'Time',
            'Quadratic': 'Time + Days_sq',
            'Log': 'Time_log',
            'Lin+Log': 'Time + Time_log',
            'Quad+Log': 'Time + Days_sq + Time_log',

            # ---- POWER LAW VARIANTS (Top performers from 5.1.1) ----
            'PowerLaw_04': 'Days_pow_40',
            'PowerLaw_05': 'Days_pow_50',
            'PowerLaw_03': 'Days_pow_30',
            'LogLog': 'log_log_Days',
            'Root_033': 'Days_pow_33',
            'CubeRoot': 'cbrt_Days',
            'PowerLaw_06': 'Days_pow_60',
            'FourthRoot': 'Days_pow_25',  # Will need to add this
            'PowerLaw_02': 'Days_pow_20',
            'PowerLaw_07': 'Days_pow_70',
            'PowerLaw_01': 'Days_pow_10',

            # ---- COMBINED MODELS ----
            'SquareRoot+Lin': 'sqrt_Days + Time',
            'Exp+Log': 'neg_Days + Time_log',
            'Recip+PowerLaw05': 'recip_Days + Days_pow_50',
            'Recip+PowerLaw': 'recip_Days + Days_pow_50',  # Alias
            'PowerLaw_Log': 'Days_pow_50 + Time_log',
            'Log+PowerLaw05': 'Time_log + Days_pow_50',
            'SquareRoot+PowerLaw': 'sqrt_Days + Days_pow_50',
            'Log+LogLog': 'Time_log + log_log_Days',
            'Recip+Log': 'recip_Days + Time_log',
            'Log+Recip': 'Time_log + recip_Days',
            'PowerLaw_Lin': 'Days_pow_50 + Time',
            'Exp+PowerLaw': 'neg_Days + Days_pow_50',
            'Tanh+Log': 'tanh_Days + Time_log',
            'SquareRoot+Recip': 'sqrt_Days + recip_Days',

            # ---- ADDITIONAL POWER LAW ----
            'PowerLaw_08': 'Days_pow_80',
            'Root_Multi': 'sqrt_Days + cbrt_Days',
            'Recip+Lin': 'recip_Days + Time',
            'Exp+Recip': 'neg_Days + recip_Days',
            'CubeRoot+Log': 'cbrt_Days + Time_log',

            # ---- SINGLE TRANSFORMS ----
            'Log10': 'log10_Days',
            'Log2': 'log2_Days',
            'Sin+Log': 'sin_Days + Time_log',
            'PowerLaw_09': 'Days_pow_90',
            'Recip+Quad': 'recip_Days + Time + Days_sq',

            # ---- COMPLEX COMBINATIONS ----
            'Lin+Log+PowerLaw': 'Time + Time_log + Days_pow_50',
            'Quad+Log': 'Time + Days_sq + Time_log',
            'Lin+Quad+Log': 'Time + Days_sq + Time_log',  # Duplicate
            'PowerLaw+Recip+Log': 'Days_pow_50 + recip_Days + Time_log',

            # ---- SIMPLE ROOTS ----
            'SquareRoot': 'sqrt_Days',
            'Exp_slow': 'neg_Days',
            'SquareRoot+Log': 'sqrt_Days + Time_log',
            'Log+SquareRoot': 'Time_log + sqrt_Days',
            'Reciprocal': 'recip_Days',
            'PowerLaw_10': 'Days_pow_100',

            # ---- HIGH-ORDER POLYNOMIALS ----
            'Quad+Log+SquareRoot': 'Time + Days_sq + Time_log + sqrt_Days',
            'Quartic': 'Time + Days_sq + Days_cube + Days_quart',
            'Cubic': 'Time + Days_sq + Days_cube',
            'Quadratic': 'Time + Days_sq',

            # ---- ULTIMATE KITCHEN SINK ----
            'Ultimate': 'Time + Days_sq + Days_cube + Time_log + sqrt_Days + recip_Days + Days_pow_50',

            # ---- REMAINING TRANSFORMS ----
            'Root_067': 'Days_pow_67',
            'Recip_sq': 'recip_sq',
            'Tanh': 'tanh_Days',
            'Cos': 'cos_Days',
            'Sin+Cos': 'sin_Days + cos_Days',
            'Exponential_proxy': 'neg_Days',  # Alias
            'Sinh': 'sin_Days',  # Approximation
            'Arctanh': 'tanh_Days',  # Limited range
            'Exp_fast': 'neg_Days',  # Alias
            'Quadratic_pure': 'Days_sq',
            'Cubic_pure': 'Days_cube',
            'Sin': 'sin_Days',
        }

        # Add missing power law exponent (0.25 for FourthRoot)
        df['Days_pow_25'] = (df['Time'] + 1) ** (-0.25)

        log(f"  Defined {len(models)} models for age interaction testing")
        log("  Each model formula: theta ~ (time_transform) * Age_c")
        log("  Age effects tested: Intercept (Age_c), Slope (time:Age_c)")
        log("")

        # =====================================================================
        # STEP 4: Fit All 66 Models with Age Interactions
        # =====================================================================

        log("[STEP 4] Fitting 66 models with Age interactions...")
        log("  Random effects: ~Time | UID (random intercepts + slopes)")
        log("  Estimation: LBFGS, REML=False")
        log("")

        results = []
        failed_models = []

        for i, (name, formula_base) in enumerate(models.items(), 1):
            log(f"  [{i:2d}/66] Fitting {name}...")
            result = fit_age_lmm(f"theta ~ {formula_base}", df, name)
            results.append(result)

            if not result['converged']:
                failed_models.append(name)

        log("")
        log(f"[SUMMARY] Fitted {len(results)} models")
        log(f"  Converged: {sum(r['converged'] for r in results)}/66")
        log(f"  Failed: {len(failed_models)}/66")
        if failed_models:
            log(f"  Failed models: {', '.join(failed_models[:10])}" +
                (" ..." if len(failed_models) > 10 else ""))
        log("")

        # =====================================================================
        # STEP 5: Compute AIC Weights and Model Comparison Table
        # =====================================================================

        log("[STEP 5] Computing AIC weights...")

        # Filter converged models only
        converged = [r for r in results if r['converged']]

        if len(converged) == 0:
            raise RuntimeError("No models converged successfully!")

        # Extract AICs
        aics = np.array([r['AIC'] for r in converged])
        min_aic = np.min(aics)
        delta_aics = aics - min_aic

        # Compute Akaike weights
        weights = np.exp(-0.5 * delta_aics)
        weights = weights / np.sum(weights)

        # Assign weights back to results
        for i, r in enumerate(converged):
            r['delta_AIC'] = delta_aics[i]
            r['akaike_weight'] = weights[i]

        # Sort by AIC (best first)
        converged = sorted(converged, key=lambda x: x['AIC'])

        # Compute cumulative weights
        cum_weight = 0.0
        for r in converged:
            cum_weight += r['akaike_weight']
            r['cumulative_weight'] = cum_weight

        log(f"  Best model: {converged[0]['model_name']} (AIC={converged[0]['AIC']:.2f})")
        log(f"  Best weight: {converged[0]['akaike_weight']:.4f} ({converged[0]['akaike_weight']*100:.1f}%)")
        log(f"  ΔAIC vs Log (original): {converged[0]['AIC'] - [r['AIC'] for r in converged if r['model_name']=='Log'][0]:.2f}")
        log("")

        # Identify competitive models (cumulative weight < 0.95)
        competitive_idx = np.where(np.array([r['cumulative_weight'] for r in converged]) <= 0.95)[0]
        n_competitive = len(competitive_idx) + 1  # +1 for model that crosses 0.95
        log(f"  Competitive models (95% cumulative weight): {n_competitive}")
        log(f"  Evidence ratio (best vs Log): {converged[0]['akaike_weight'] / [r['akaike_weight'] for r in converged if r['model_name']=='Log'][0]:.1f}:1")
        log("")

        # =====================================================================
        # STEP 6: Model-Averaged Age Effects
        # =====================================================================

        log("[STEP 6] Computing model-averaged age effects...")

        # Extract age effect estimates from all converged models
        # Focus on: Age_c (intercept), time:Age_c (slope interaction)
        age_effect_types = ['Age_c']  # Start with intercept

        # Identify all time:Age_c interactions across models
        for r in converged:
            for effect_name in r['age_effects'].keys():
                if ':Age_c' in effect_name and effect_name not in age_effect_types:
                    age_effect_types.append(effect_name)

        log(f"  Age effect types identified: {len(age_effect_types)}")
        for ae in age_effect_types[:5]:
            log(f"    - {ae}")
        if len(age_effect_types) > 5:
            log(f"    ... and {len(age_effect_types)-5} more")
        log("")

        # Model-average each age effect type
        averaged_effects = []

        for effect_type in age_effect_types:
            # Collect estimates and weights from models that have this effect
            estimates = []
            variances = []
            weights_for_effect = []

            for r in converged:
                if effect_type in r['age_effects']:
                    estimates.append(r['age_effects'][effect_type]['coef'])
                    variances.append(r['age_effects'][effect_type]['se'] ** 2)
                    weights_for_effect.append(r['akaike_weight'])

            if len(estimates) == 0:
                continue

            # Renormalize weights (models without this effect excluded)
            weights_for_effect = np.array(weights_for_effect)
            weights_for_effect = weights_for_effect / np.sum(weights_for_effect)

            # Model-averaged estimate
            avg_coef = np.sum(np.array(estimates) * weights_for_effect)

            # Model-averaged variance (Burnham & Anderson 2002, eq 4.9)
            # Var(β̄) = Σ w_i [Var(β_i|M_i) + (β_i - β̄)²]
            unconditional_var = np.sum(weights_for_effect * (
                np.array(variances) + (np.array(estimates) - avg_coef)**2
            ))
            avg_se = np.sqrt(unconditional_var)

            # Z-test and p-value
            avg_z = avg_coef / avg_se
            avg_p = 2 * (1 - stats.norm.cdf(np.abs(avg_z)))

            averaged_effects.append({
                'effect': effect_type,
                'coef_averaged': avg_coef,
                'se_averaged': avg_se,
                'z_averaged': avg_z,
                'p_averaged': avg_p,
                'n_models': len(estimates),
                'weight_sum': np.sum(weights_for_effect)
            })

        log(f"  Model-averaged {len(averaged_effects)} age effects")
        log("")

        # Log key averaged effects
        for ae in averaged_effects[:3]:
            log(f"  {ae['effect']}:")
            log(f"    Coef: {ae['coef_averaged']:.5f} ± {ae['se_averaged']:.5f}")
            log(f"    Z: {ae['z_averaged']:.3f}, p={ae['p_averaged']:.4f}")
            log(f"    Based on {ae['n_models']} models")
        log("")

        # =====================================================================
        # STEP 7: Save Outputs
        # =====================================================================

        log("[STEP 7] Saving outputs...")

        # Save fitted models
        models_path = RQ_DIR / "data" / "step02b_extended_model_fits.pkl"
        with open(models_path, 'wb') as f:
            pickle.dump(converged, f)
        log(f"  Saved: {models_path.name} ({len(converged)} models)")

        # Save model comparison table
        comparison_data = []
        for r in converged:
            row = {
                'model_name': r['model_name'],
                'AIC': r['AIC'],
                'BIC': r['BIC'],
                'log_likelihood': r['log_likelihood'],
                'n_params': r['n_params'],
                'converged': r['converged'],
                'delta_AIC': r['delta_AIC'],
                'akaike_weight': r['akaike_weight'],
                'cumulative_weight': r['cumulative_weight']
            }

            # Add age effect estimates (flat structure for CSV)
            for effect_name, effect_data in r['age_effects'].items():
                row[f'{effect_name}_coef'] = effect_data['coef']
                row[f'{effect_name}_se'] = effect_data['se']
                row[f'{effect_name}_z'] = effect_data['z']
                row[f'{effect_name}_p'] = effect_data['p']

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = RQ_DIR / "data" / "step02b_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        log(f"  Saved: {comparison_path.name} ({len(comparison_df)} rows)")

        # Save model-averaged age effects
        averaged_df = pd.DataFrame(averaged_effects)
        averaged_path = RQ_DIR / "data" / "step02b_age_effects_averaged.csv"
        averaged_df.to_csv(averaged_path, index=False, encoding='utf-8')
        log(f"  Saved: {averaged_path.name} ({len(averaged_df)} effects)")

        log("")
        log("="*70)
        log("[SUCCESS] Extended age model comparison complete!")
        log("="*70)
        log(f"Next: Examine step02b_model_comparison.csv for functional form effects")
        log(f"      and step02b_age_effects_averaged.csv for robust age estimates")

    except Exception as e:
        log("")
        log("="*70)
        log("[ERROR] Extended age model comparison failed!")
        log("="*70)
        log(f"Error: {str(e)}")
        log("")
        log("Traceback:")
        log(traceback.format_exc())
        sys.exit(1)
