"""
Step 5d: Refit LMM with Random Slopes (CORRECTED MODEL)

Purpose: Refit LocationType × Time interaction model with random slopes
         (the model that SHOULD have been used from the start).

CRITICAL: Original analysis used intercepts-only. Random slopes comparison
         showed ΔAIC = 60.82 improvement - this is a MAJOR model misspecification.
         We must refit with proper random structure.

Author: rq_platinum agent
Date: 2025-12-27
"""

import sys
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def fit_lmm_with_slopes(data):
    """Fit LMM with random intercepts + slopes."""
    print("\n" + "="*80)
    print("FITTING LMM WITH RANDOM SLOPES")
    print("="*80)

    formula = "theta ~ C(location) * log_TSVR"
    print(f"\nFormula: {formula}")
    print(f"Random effects: ~log_TSVR (intercepts + slopes)")
    print(f"Groups: UID")
    print(f"REML: False")

    model = MixedLM.from_formula(
        formula=formula,
        data=data,
        groups=data['UID'],
        re_formula="~log_TSVR"  # Intercepts + slopes
    )

    result = model.fit(reml=False, method='lbfgs', maxiter=1000)

    print(f"\nConverged: {result.converged}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    print(f"Log-Likelihood: {result.llf:.2f}")

    print(f"\n{result.summary()}")

    return result

def extract_fixed_effects(result):
    """Extract fixed effects table with proper slicing."""
    # Get number of fixed effects (excludes random effects)
    n_fe = len(result.model.exog_names)

    # Extract fixed effect parameters
    params = result.params[:n_fe]
    bse = result.bse[:n_fe]
    tvalues = result.tvalues[:n_fe]
    pvalues = result.pvalues[:n_fe]

    # Compute 95% CIs
    ci_lower = params - 1.96 * bse
    ci_upper = params + 1.96 * bse

    # Create DataFrame
    fe_table = pd.DataFrame({
        'term': result.model.exog_names,
        'coefficient': params,
        'se': bse,
        'z': tvalues,
        'p_value': pvalues,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })

    return fe_table

def interpret_interaction(fe_table):
    """Interpret the LocationType × log_TSVR interaction."""
    print("\n" + "="*80)
    print("INTERACTION INTERPRETATION")
    print("="*80)

    # Find interaction term
    interaction_row = fe_table[fe_table['term'].str.contains('log_TSVR', na=False) &
                                fe_table['term'].str.contains('location', na=False)]

    if len(interaction_row) == 0:
        print("⚠️  Interaction term not found in model")
        return

    interaction_row = interaction_row.iloc[0]

    coef = interaction_row['coefficient']
    se = interaction_row['se']
    p = interaction_row['p_value']
    ci_lower = interaction_row['ci_lower']
    ci_upper = interaction_row['ci_upper']

    print(f"\nLocationType × log_TSVR Interaction:")
    print(f"  β = {coef:.4f}")
    print(f"  SE = {se:.4f}")
    print(f"  z = {coef/se:.2f}")
    print(f"  p = {p:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    if p < 0.05:
        print(f"\n✅ SIGNIFICANT INTERACTION (p = {p:.4f})")
        print("   → Source and destination show DIFFERENT decline rates")
        if coef > 0:
            print("   → Destination declines SLOWER than source")
        else:
            print("   → Destination declines FASTER than source")
        print("\n🔴 NOTE: This CONTRADICTS original intercepts-only finding (p=0.553)")
        print("   → Original model UNDERESTIMATED interaction by failing to account")
        print("      for individual differences in slopes (random effects misspecification)")
    else:
        print(f"\n❌ NON-SIGNIFICANT INTERACTION (p = {p:.4f})")
        print("   → Source and destination show EQUIVALENT decline rates")
        print("   → Consistent with original intercepts-only finding")

    return coef, se, p

def save_results(result, fe_table, output_dir='results/ch6/6.8.1/data'):
    """Save model outputs."""
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)

    # Save full summary
    summary_path = f'{output_dir}/step05d_lmm_with_slopes_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Linear Mixed Model Summary - WITH RANDOM SLOPES\n")
        f.write("RQ: ch6/6.8.1 - Source-Destination Confidence Trajectories\n")
        f.write("="*80 + "\n\n")
        f.write(str(result.summary()))
        f.write("\n\n" + "="*80 + "\n")
        f.write("Model Specification\n")
        f.write("="*80 + "\n")
        f.write(f"Formula: theta ~ C(location) * log_TSVR\n")
        f.write(f"Random effects: ~log_TSVR (intercepts + slopes)\n")
        f.write(f"Groups: UID\n")
        f.write(f"REML: False\n")
        f.write(f"Converged: {result.converged}\n")
        f.write(f"AIC: {result.aic:.4f}\n")
        f.write(f"BIC: {result.bic:.4f}\n")
        f.write(f"Log-Likelihood: {result.llf:.4f}\n")
        f.write(f"N observations: {result.nobs}\n")
        f.write(f"N groups: {result.model.n_groups}\n")

    print(f"✅ Saved: {summary_path}")

    # Save fixed effects table
    fe_path = f'{output_dir}/step05d_lmm_with_slopes_coefficients.csv'
    fe_table.to_csv(fe_path, index=False, float_format='%.6f')
    print(f"✅ Saved: {fe_path}")

    # Save random effects variance components
    var_path = f'{output_dir}/step05d_random_effects_variance.csv'
    var_df = pd.DataFrame({
        'component': ['Intercept_variance', 'Slope_variance', 'Intercept_Slope_cov', 'Residual_variance'],
        'value': [
            result.cov_re.iloc[0, 0],
            result.cov_re.iloc[1, 1],
            result.cov_re.iloc[0, 1],
            result.scale
        ]
    })
    var_df.to_csv(var_path, index=False, float_format='%.6f')
    print(f"✅ Saved: {var_path}")

def main():
    """Execute LMM with random slopes."""
    print("="*80)
    print("LMM WITH RANDOM SLOPES - RQ 6.8.1")
    print("="*80)
    print("\n🔴 CRITICAL CORRECTION")
    print("Original analysis used random intercepts only (re_formula='~1')")
    print("Random slopes comparison showed ΔAIC = 60.82 improvement with slopes")
    print("This is MASSIVE - original model was severely misspecified")
    print("\nRefitting with CORRECT random structure (intercepts + slopes)...\n")

    # Load data
    data = pd.read_csv('results/ch6/6.8.1/data/step04_lmm_input.csv')
    print(f"N observations: {len(data)}")
    print(f"N participants: {data['UID'].nunique()}")

    # Fit model
    result = fit_lmm_with_slopes(data)

    # Extract fixed effects
    fe_table = extract_fixed_effects(result)

    print("\n" + "="*80)
    print("FIXED EFFECTS TABLE")
    print("="*80)
    print(fe_table.to_string(index=False))

    # Interpret interaction
    interact_coef, interact_se, interact_p = interpret_interaction(fe_table)

    # Save results
    save_results(result, fe_table)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 KEY FINDING:")
    print(f"   Interaction β = {interact_coef:.4f}, p = {interact_p:.4f}")
    if interact_p < 0.05:
        print(f"   🔴 SIGNIFICANT - source-destination dissociation EXISTS in confidence")
        print(f"   🔴 CONTRADICTS original NULL finding (model misspecification artifact)")
    else:
        print(f"   ✅ NON-SIGNIFICANT - no source-destination dissociation (NULL robust)")

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
