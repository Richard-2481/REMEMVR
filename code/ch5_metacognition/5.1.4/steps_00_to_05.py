"""
RQ 6.1.4 - ICC Decomposition Analysis (Steps 00-05)

PURPOSE:
Decompose variance in confidence trajectories into trait-like (intercept: baseline confidence)
and state-like (residual: within-person fluctuation) components. Tests whether 5-level ordinal
confidence data reveals detectable slope variance that dichotomous accuracy data from Chapter 5
could not resolve (measurement precision hypothesis).

KEY QUESTION:
Is confidence decline trait-like (individual differences in forgetting rate) or state-like (universal)?

INPUT:
- results/ch6/6.1.1/data/step04_lmm_input.csv (400 rows: 100 participants × 4 tests)
  - Columns: composite_ID, UID, test, theta_All, se_All, TSVR_hours, ...

OUTPUT:
- data/step00_model_metadata.txt (model specification and convergence)
- data/step01_variance_components.csv (4 components: var_intercept, var_slope, cov_int_slope, var_residual)
- data/step02_icc_estimates.csv (3 ICCs: ICC_intercept, ICC_slope_simple, ICC_slope_conditional)
- data/step03_random_effects.csv (100 rows: participant-level intercepts + slopes for RQ 6.1.5)
- data/step04_intercept_slope_correlation.csv (correlation with D068 dual p-values)
- data/step05_ch5_icc_comparison.csv (CRITICAL: comparison with Ch5 ICC_slope=0.0005)

METHODOLOGY:
- Re-fit best CONVERGED model (Recip_sq) from RQ 6.1.1 kitchen sink comparison
- Recip_sq formula: theta ~ 1/(TSVR_hours+1)^2 with random intercept + slope
- Extract variance components from cov_re matrix
- Compute ICCs following Hoffman & Stawski (2009)
- Test measurement artifact vs universal forgetting hypothesis

Date: 2025-12-11
RQ: ch6/6.1.4
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = RQ_DIR / "logs" / "steps_00_to_05.log"
DATA_DIR = RQ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
(RQ_DIR / "logs").mkdir(exist_ok=True)

def log(msg):
    """Log message to file and stdout with flush"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def step00_refit_best_model():
    """
    Step 00: Re-fit best converged LMM model from RQ 6.1.1

    NOTE: Cannot load pickle due to patsy eval_env error.
    Re-fitting from scratch using step04_lmm_input.csv.
    Best CONVERGED model: Recip_sq (1/(t+1)^2)
    """
    import statsmodels.formula.api as smf

    log("=" * 80)
    log("[STEP 00] Re-fit Best LMM Model (Recip_sq)")
    log("=" * 80)

    # Load LMM input from RQ 6.1.1
    input_path = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step04_lmm_input.csv"
    log(f"Loading LMM input from RQ 6.1.1: {input_path}")
    df = pd.read_csv(input_path)
    log(f"  ✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    log(f"  ✓ N participants: {df['UID'].nunique()}")
    log(f"  ✓ TSVR range: [{df['TSVR_hours'].min():.2f}, {df['TSVR_hours'].max():.2f}] hours")

    # Create reciprocal squared transformation: 1/(t+1)^2
    df['Recip_sq'] = 1.0 / (df['TSVR_hours'] + 1) ** 2
    log(f"  ✓ Created Recip_sq = 1/(TSVR_hours+1)^2")
    log(f"  ✓ Recip_sq range: [{df['Recip_sq'].min():.6f}, {df['Recip_sq'].max():.6f}]")

    # Fit LMM with random intercept + slope on Recip_sq
    formula = "theta_All ~ Recip_sq"
    re_formula = "~Recip_sq"  # Random intercept + random slope on Recip_sq

    log(f"\nFitting LMM:")
    log(f"  Formula: {formula}")
    log(f"  Random effects: (1 + Recip_sq | UID)")
    log(f"  Method: ML (reml=False for AIC comparison)")

    model = smf.mixedlm(formula, df, groups=df["UID"], re_formula=re_formula)
    result = model.fit(reml=False)

    log(f"\nModel fit complete:")
    log(f"  ✓ Converged: {result.converged}")
    log(f"  ✓ AIC: {result.aic:.4f}")
    log(f"  ✓ BIC: {result.bic:.4f}")
    log(f"  ✓ Log-likelihood: {result.llf:.4f}")
    log(f"  ✓ N observations: {result.nobs}")
    log(f"  ✓ N groups: {result.model.n_groups}")

    # Check convergence
    if not result.converged:
        log("Model did not converge - results may be unreliable")

    # Save model metadata
    metadata_path = DATA_DIR / "step00_model_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RQ 6.1.4 - Step 00: Model Metadata\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Source: RQ 6.1.1 best CONVERGED model (Recip_sq)\n")
        f.write(f"Data: results/ch6/6.1.1/data/step04_lmm_input.csv\n\n")
        f.write("Model Specification:\n")
        f.write(f"  Formula: theta_All ~ Recip_sq\n")
        f.write(f"  Recip_sq = 1/(TSVR_hours+1)^2\n")
        f.write(f"  Random effects: (1 + Recip_sq | UID)\n")
        f.write(f"  Method: ML (REML=False)\n\n")
        f.write("Fit Statistics:\n")
        f.write(f"  Converged: {result.converged}\n")
        f.write(f"  AIC: {result.aic:.4f}\n")
        f.write(f"  BIC: {result.bic:.4f}\n")
        f.write(f"  Log-likelihood: {result.llf:.4f}\n")
        f.write(f"  N observations: {result.nobs}\n")
        f.write(f"  N groups: {result.model.n_groups}\n\n")
        f.write("Fixed Effects:\n")
        f.write(str(result.summary().tables[1]) + "\n\n")
        f.write("Random Effects Covariance:\n")
        f.write(str(result.cov_re) + "\n")

    log(f"  ✓ Saved metadata: {metadata_path}")

    return result, df


def step01_extract_variance_components(result):
    """
    Step 01: Extract 4 variance components from LMM random effects

    Components:
    - var_intercept: Random intercept variance (baseline confidence differences)
    - var_slope: Random slope variance (forgetting rate differences)
    - cov_int_slope: Covariance between intercepts and slopes
    - var_residual: Residual variance (within-person fluctuation)
    """
    log("\n" + "=" * 80)
    log("[STEP 01] Extract Variance Components")
    log("=" * 80)

    # Extract random effects covariance matrix
    cov_re = result.cov_re
    log(f"\nRandom effects covariance matrix:")
    log(f"{cov_re}")

    # Get variance components
    # cov_re is a DataFrame with row/col names from re_formula
    # For re_formula="~Recip_sq", we have: Group (intercept) and Recip_sq (slope)

    # Get the names
    re_names = list(cov_re.columns)
    log(f"  Random effect names: {re_names}")

    # Extract values
    var_intercept = cov_re.iloc[0, 0]  # Group variance (intercept)
    var_slope = cov_re.iloc[1, 1] if len(re_names) > 1 else 0.0  # Slope variance
    cov_int_slope = cov_re.iloc[0, 1] if len(re_names) > 1 else 0.0  # Covariance
    var_residual = result.scale  # Residual variance

    log(f"\n[VARIANCE COMPONENTS]:")
    log(f"  var_intercept: {var_intercept:.6f}")
    log(f"  var_slope: {var_slope:.6f}")
    log(f"  cov_int_slope: {cov_int_slope:.6f}")
    log(f"  var_residual: {var_residual:.6f}")

    # Compute correlation from covariance
    if var_intercept > 0 and var_slope > 0:
        cor_int_slope = cov_int_slope / np.sqrt(var_intercept * var_slope)
    else:
        cor_int_slope = 0.0
    log(f"  cor_int_slope: {cor_int_slope:.4f}")

    # Validate variance components
    if var_intercept < 0:
        log("Negative intercept variance - model estimation error")
        raise ValueError("Negative intercept variance detected")
    if var_slope < 0:
        log("Negative slope variance - model estimation error")
        raise ValueError("Negative slope variance detected")
    if var_residual < 0:
        log("Negative residual variance - model estimation error")
        raise ValueError("Negative residual variance detected")

    # Check covariance bounds
    max_cov = np.sqrt(var_intercept * var_slope) if var_slope > 0 else 0
    if abs(cov_int_slope) > max_cov + 1e-6:
        log(f"Covariance |{cov_int_slope:.6f}| exceeds bound {max_cov:.6f}")

    # Save variance components
    components = pd.DataFrame({
        'component': ['var_intercept', 'var_slope', 'cov_int_slope', 'var_residual'],
        'value': [var_intercept, var_slope, cov_int_slope, var_residual],
        'SE': [np.nan, np.nan, np.nan, np.nan]  # SE not readily available from statsmodels
    })

    output_path = DATA_DIR / "step01_variance_components.csv"
    components.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return {
        'var_intercept': var_intercept,
        'var_slope': var_slope,
        'cov_int_slope': cov_int_slope,
        'var_residual': var_residual,
        'cor_int_slope': cor_int_slope
    }


def step02_compute_icc_estimates(variance_components, df):
    """
    Step 02: Compute 3 ICC estimates following Hoffman & Stawski (2009)

    ICCs:
    - ICC_intercept: Proportion of variance attributable to stable baseline differences
    - ICC_slope_simple: Proportion of slope variance relative to total change variance
    - ICC_slope_conditional: Slope variance at final timepoint (Day 6 = 144 hours)
    """
    log("\n" + "=" * 80)
    log("[STEP 02] Compute ICC Estimates (Hoffman & Stawski 2009)")
    log("=" * 80)

    var_int = variance_components['var_intercept']
    var_slope = variance_components['var_slope']
    var_res = variance_components['var_residual']

    # Get mean time (TSVR_hours) and max time for calculations
    mean_time = df['TSVR_hours'].mean()
    max_time = df['TSVR_hours'].max()  # Day 6 ~144-150 hours

    log(f"\n:")
    log(f"  Mean TSVR_hours: {mean_time:.2f}")
    log(f"  Max TSVR_hours (Day 6): {max_time:.2f}")

    # Note: For Recip_sq model, the "slope" is on the transformed time scale
    # We need to compute ICC using the transformed time scale
    mean_recip_sq = 1.0 / (mean_time + 1) ** 2
    max_recip_sq = 1.0 / (max_time + 1) ** 2

    log(f"  Mean Recip_sq: {mean_recip_sq:.6f}")
    log(f"  Max Recip_sq (Day 6): {max_recip_sq:.6f}")

    # ICC_intercept: Proportion of total variance at mean time
    # Total variance = var_int + var_slope*time^2 + 2*cov*time + var_res
    # For random slope on Recip_sq, use mean_recip_sq
    total_var_mean = (var_int +
                      var_slope * mean_recip_sq**2 +
                      2 * variance_components['cov_int_slope'] * mean_recip_sq +
                      var_res)

    ICC_intercept = var_int / total_var_mean if total_var_mean > 0 else 0

    log(f"\n(baseline individual differences):")
    log(f"  Formula: var_intercept / total_variance_at_mean_time")
    log(f"  Total variance at mean time: {total_var_mean:.6f}")
    log(f"  ICC_intercept: {ICC_intercept:.4f}")

    # ICC_slope_simple: Proportion of slope variance
    # Simple approach: var_slope / (var_slope + var_residual)
    ICC_slope_simple = var_slope / (var_slope + var_res) if (var_slope + var_res) > 0 else 0

    log(f"\n(forgetting rate individual differences):")
    log(f"  Formula: var_slope / (var_slope + var_residual)")
    log(f"  ICC_slope_simple: {ICC_slope_simple:.4f}")

    # ICC_slope_conditional: Slope variance at final timepoint
    # At Day 6, what proportion of variance is due to slope differences?
    total_var_max = (var_int +
                     var_slope * max_recip_sq**2 +
                     2 * variance_components['cov_int_slope'] * max_recip_sq +
                     var_res)

    ICC_slope_conditional = var_slope * max_recip_sq**2 / total_var_max if total_var_max > 0 else 0

    log(f"\n(slope variance at Day 6):")
    log(f"  Formula: var_slope * time^2 / total_variance_at_day6")
    log(f"  Total variance at Day 6: {total_var_max:.6f}")
    log(f"  ICC_slope_conditional: {ICC_slope_conditional:.6f}")

    # Interpret ICCs
    def interpret_icc(icc):
        if icc < 0.05:
            return "negligible"
        elif icc < 0.10:
            return "small"
        elif icc < 0.30:
            return "moderate"
        else:
            return "substantial"

    # Create output
    icc_df = pd.DataFrame({
        'icc_type': ['ICC_intercept', 'ICC_slope_simple', 'ICC_slope_conditional'],
        'value': [ICC_intercept, ICC_slope_simple, ICC_slope_conditional],
        'interpretation': [
            interpret_icc(ICC_intercept),
            interpret_icc(ICC_slope_simple),
            interpret_icc(ICC_slope_conditional)
        ]
    })

    output_path = DATA_DIR / "step02_icc_estimates.csv"
    icc_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    log(f"\n:")
    log(f"  ICC_intercept:       {ICC_intercept:.4f} ({interpret_icc(ICC_intercept)})")
    log(f"  ICC_slope_simple:    {ICC_slope_simple:.4f} ({interpret_icc(ICC_slope_simple)})")
    log(f"  ICC_slope_conditional: {ICC_slope_conditional:.6f} ({interpret_icc(ICC_slope_conditional)})")

    return {
        'ICC_intercept': ICC_intercept,
        'ICC_slope_simple': ICC_slope_simple,
        'ICC_slope_conditional': ICC_slope_conditional
    }


def step03_extract_random_effects(result, df):
    """
    Step 03: Extract 100 participant-level random effects

    CRITICAL: This output is REQUIRED for RQ 6.1.5 (Clustering Analysis)

    Output:
    - UID: Participant identifier
    - random_intercept: Deviation from population mean intercept
    - random_slope: Deviation from population mean slope
    """
    log("\n" + "=" * 80)
    log("[STEP 03] Extract Participant-Level Random Effects")
    log("=" * 80)

    # Get random effects from model
    random_effects = result.random_effects

    log(f"\nN groups: {len(random_effects)}")

    # Convert to DataFrame
    re_list = []
    for uid, re_array in random_effects.items():
        # re_array is a Series with index names from re_formula
        # For "~Recip_sq": ['Group', 'Recip_sq']
        re_dict = {'UID': str(uid)}

        if hasattr(re_array, 'index'):
            # Extract by index name
            re_dict['random_intercept'] = re_array.iloc[0]  # Group (intercept)
            if len(re_array) > 1:
                re_dict['random_slope'] = re_array.iloc[1]  # Recip_sq (slope)
            else:
                re_dict['random_slope'] = 0.0
        else:
            # Array format
            re_dict['random_intercept'] = re_array[0]
            re_dict['random_slope'] = re_array[1] if len(re_array) > 1 else 0.0

        re_list.append(re_dict)

    re_df = pd.DataFrame(re_list)

    log(f"\n:")
    log(f"  ✓ N participants: {len(re_df)}")
    log(f"  ✓ N NaN intercepts: {re_df['random_intercept'].isna().sum()}")
    log(f"  ✓ N NaN slopes: {re_df['random_slope'].isna().sum()}")
    log(f"  ✓ Random intercept range: [{re_df['random_intercept'].min():.4f}, {re_df['random_intercept'].max():.4f}]")
    log(f"  ✓ Random slope range: [{re_df['random_slope'].min():.6f}, {re_df['random_slope'].max():.6f}]")

    # Check for exactly 100 participants
    if len(re_df) != 100:
        log(f"Expected 100 participants, found {len(re_df)}")

    # Save
    output_path = DATA_DIR / "step03_random_effects.csv"
    re_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")
    log(f"  ✓ NOTE: This file is REQUIRED for RQ 6.1.5 (Clustering Analysis)")

    return re_df


def step04_test_intercept_slope_correlation(re_df):
    """
    Step 04: Test correlation between baseline confidence (intercept) and forgetting rate (slope)

    Decision D068: Dual p-value reporting (uncorrected + Bonferroni)

    Question: Do high baseline confidence individuals show faster or slower decline?
    - Positive r: Higher baseline -> slower forgetting (protective effect)
    - Negative r: Higher baseline -> faster forgetting (regression to mean)
    """
    from scipy import stats

    log("\n" + "=" * 80)
    log("[STEP 04] Test Intercept-Slope Correlation (D068 Dual P-values)")
    log("=" * 80)

    intercepts = re_df['random_intercept'].values
    slopes = re_df['random_slope'].values

    # Pearson correlation
    r, p_uncorrected = stats.pearsonr(intercepts, slopes)

    # 95% CI using Fisher z-transformation
    n = len(intercepts)
    z = np.arctanh(r)
    se_z = 1.0 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se_z
    z_upper = z + 1.96 * se_z
    CI_lower = np.tanh(z_lower)
    CI_upper = np.tanh(z_upper)

    # Bonferroni correction (single test, so p_bonferroni = p_uncorrected)
    n_tests = 1  # Only one planned comparison
    p_bonferroni = min(p_uncorrected * n_tests, 1.0)

    # Interpret
    def interpret_correlation(r, p):
        direction = "positive" if r > 0 else "negative"
        magnitude = "negligible" if abs(r) < 0.10 else "small" if abs(r) < 0.30 else "moderate" if abs(r) < 0.50 else "large"
        significance = "significant" if p < 0.05 else "non-significant"

        if r > 0:
            meaning = "Higher baseline confidence associated with slower decline"
        else:
            meaning = "Higher baseline confidence associated with faster decline"

        return f"{magnitude} {direction} correlation ({significance}); {meaning}"

    interpretation = interpret_correlation(r, p_uncorrected)

    log(f"\n[CORRELATION TEST]:")
    log(f"  Pearson r: {r:.4f}")
    log(f"  95% CI: [{CI_lower:.4f}, {CI_upper:.4f}]")
    log(f"  N: {n}")
    log(f"\n[D068 DUAL P-VALUES]:")
    log(f"  p_uncorrected: {p_uncorrected:.4f}")
    log(f"  p_bonferroni: {p_bonferroni:.4f} (N tests = {n_tests})")
    log(f"\n:")
    log(f"  {interpretation}")

    # Save
    corr_df = pd.DataFrame({
        'correlation_r': [r],
        'CI_lower': [CI_lower],
        'CI_upper': [CI_upper],
        'N': [n],
        'p_uncorrected': [p_uncorrected],
        'p_bonferroni': [p_bonferroni],
        'interpretation': [interpretation]
    })

    output_path = DATA_DIR / "step04_intercept_slope_correlation.csv"
    corr_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")
    log(f"  ✓ Decision D068 compliance: PASS (dual p-values reported)")

    return {
        'r': r,
        'p_uncorrected': p_uncorrected,
        'p_bonferroni': p_bonferroni,
        'CI_lower': CI_lower,
        'CI_upper': CI_upper
    }


def step05_compare_icc_ch5(icc_estimates):
    """
    Step 05: CRITICAL COMPARISON - Test if ICC_slope differs between ordinal and binary data

    Compare:
    - ICC_slope_confidence (this RQ, 5-level ordinal data)
    - ICC_slope_accuracy (Chapter 5 RQ 5.1.4, dichotomous data = 0.0005)

    Hypothesis Test:
    - Measurement Artifact Hypothesis: ICC_slope_confidence > 0.10 (ordinal reveals variance)
    - Universal Forgetting Hypothesis: ICC_slope_confidence ≈ 0 (both near zero)
    """
    log("\n" + "=" * 80)
    log("[STEP 05] Compare ICC_slope with Chapter 5 Accuracy Data")
    log("=" * 80)

    ICC_slope_confidence = icc_estimates['ICC_slope_simple']
    ICC_slope_accuracy = 0.0005  # Hard-coded from Chapter 5 RQ 5.1.4

    log(f"\n:")
    log(f"  ICC_slope_confidence (5-level ordinal): {ICC_slope_confidence:.4f}")
    log(f"  ICC_slope_accuracy (dichotomous):       {ICC_slope_accuracy:.4f}")

    # Compute difference and ratio
    delta_ICC = ICC_slope_confidence - ICC_slope_accuracy
    ratio_ICC = ICC_slope_confidence / ICC_slope_accuracy if ICC_slope_accuracy > 0 else float('inf')

    log(f"\n:")
    log(f"  Delta ICC (confidence - accuracy): {delta_ICC:.4f}")
    log(f"  Ratio ICC (confidence / accuracy): {ratio_ICC:.1f}x")

    # Classify hypothesis
    if ICC_slope_confidence > 0.10:
        hypothesis_supported = "Measurement Artifact"
        interpretation = (f"ICC_slope_confidence ({ICC_slope_confidence:.4f}) > 0.10 threshold. "
                         f"Ordinal data reveals {ratio_ICC:.0f}x more slope variance than dichotomous data. "
                         f"Chapter 5 finding of near-zero slope variance was a measurement limitation, "
                         f"not a substantive finding about forgetting dynamics.")
    elif ICC_slope_confidence < 0.05:
        hypothesis_supported = "Universal Forgetting"
        interpretation = (f"ICC_slope_confidence ({ICC_slope_confidence:.4f}) < 0.05 threshold. "
                         f"Both ordinal and dichotomous data show near-zero slope variance. "
                         f"Forgetting rate shows minimal trait variance regardless of measurement precision - "
                         f"forgetting trajectories are remarkably universal across individuals.")
    else:
        hypothesis_supported = "Inconclusive"
        interpretation = (f"ICC_slope_confidence ({ICC_slope_confidence:.4f}) in ambiguous range [0.05, 0.10]. "
                         f"Cannot definitively support either hypothesis. "
                         f"Some slope variance detected but magnitude unclear.")

    log(f"\n[HYPOTHESIS TEST]:")
    log(f"  Hypothesis supported: {hypothesis_supported}")
    log(f"\n:")
    log(f"  {interpretation}")

    # Save
    comparison_df = pd.DataFrame({
        'ICC_slope_confidence': [ICC_slope_confidence],
        'ICC_slope_accuracy': [ICC_slope_accuracy],
        'delta_ICC': [delta_ICC],
        'ratio_ICC': [ratio_ICC],
        'hypothesis_supported': [hypothesis_supported],
        'interpretation': [interpretation]
    })

    output_path = DATA_DIR / "step05_ch5_icc_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return {
        'ICC_slope_confidence': ICC_slope_confidence,
        'ICC_slope_accuracy': ICC_slope_accuracy,
        'delta_ICC': delta_ICC,
        'ratio_ICC': ratio_ICC,
        'hypothesis_supported': hypothesis_supported
    }


if __name__ == "__main__":
    try:
        log("=" * 80)
        log(f"RQ 6.1.4 - ICC Decomposition Analysis")
        log(f"Started: {datetime.now().isoformat()}")
        log("=" * 80)

        # Step 0: Re-fit best LMM model
        result, df = step00_refit_best_model()

        # Step 1: Extract variance components
        variance_components = step01_extract_variance_components(result)

        # Step 2: Compute ICC estimates
        icc_estimates = step02_compute_icc_estimates(variance_components, df)

        # Step 3: Extract participant-level random effects
        re_df = step03_extract_random_effects(result, df)

        # Step 4: Test intercept-slope correlation
        corr_results = step04_test_intercept_slope_correlation(re_df)

        # Step 5: Compare with Chapter 5
        comparison = step05_compare_icc_ch5(icc_estimates)

        log("\n" + "=" * 80)
        log("RQ 6.1.4 Complete")
        log("=" * 80)
        log(f"\n:")
        log(f"  ICC_intercept (baseline variance): {icc_estimates['ICC_intercept']:.4f}")
        log(f"  ICC_slope (forgetting rate variance): {icc_estimates['ICC_slope_simple']:.4f}")
        log(f"  Intercept-slope correlation: r={corr_results['r']:.4f}, p={corr_results['p_uncorrected']:.4f}")
        log(f"  Hypothesis supported: {comparison['hypothesis_supported']}")
        log(f"\n  Completed: {datetime.now().isoformat()}")

    except Exception as e:
        log(f"\n{e}")
        import traceback
        log(traceback.format_exc())
        raise
