#!/usr/bin/env python3
"""
Step 9-13: METHOD B - Linear Mixed Model Slope Comparison
RQ 6.9.1: Extended analysis using all 400 observations

This script implements the LMM slope comparison to contrast with METHOD A
(individual decline rates). Uses all 4 timepoints instead of just T1 and T4.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for tools import
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from tools.plotting import convert_theta_to_probability

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step09-13_lmm_slopes.log"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def fit_lmm(df, theta_col, model_name):
    """
    Fit Linear Mixed Model: theta ~ TSVR_hours + (1|UID)

    Returns model object and extracted coefficients
    """
    log(f"\n[FIT] Fitting {model_name} LMM...")

    # Prepare data
    model_df = df[['UID', 'TSVR_hours', theta_col]].copy()
    model_df.columns = ['UID', 'TSVR_hours', 'theta']

    # Fit model using statsmodels
    formula = "theta ~ TSVR_hours"
    model = smf.mixedlm(formula, model_df, groups=model_df['UID'])
    result = model.fit(method='lbfgs')

    # Extract coefficients
    intercept = result.params['Intercept']
    slope = result.params['TSVR_hours']
    intercept_se = result.bse['Intercept']
    slope_se = result.bse['TSVR_hours']
    slope_p = result.pvalues['TSVR_hours']

    # Model fit statistics
    aic = result.aic
    bic = result.bic

    log(f"  Intercept: {intercept:.4f} (SE={intercept_se:.4f})")
    log(f"  Slope: {slope:.6f} (SE={slope_se:.6f}, p={slope_p:.4f})")
    log(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")

    return result, {
        'model': model_name,
        'intercept': intercept,
        'slope': slope,
        'intercept_se': intercept_se,
        'slope_se': slope_se,
        'slope_p': slope_p,
        'aic': aic,
        'bic': bic
    }

def test_slope_difference(acc_stats, conf_stats):
    """
    Test if slopes differ using z-test for difference between coefficients.

    z = (b1 - b2) / sqrt(SE1^2 + SE2^2)
    """
    log("\n[TEST] Testing slope difference...")

    acc_slope = acc_stats['slope']
    conf_slope = conf_stats['slope']
    acc_se = acc_stats['slope_se']
    conf_se = conf_stats['slope_se']

    # Compute z-statistic
    slope_diff = conf_slope - acc_slope
    se_diff = np.sqrt(acc_se**2 + conf_se**2)
    z_stat = slope_diff / se_diff

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    log(f"  Slope difference: {slope_diff:.6f}")
    log(f"  SE of difference: {se_diff:.6f}")
    log(f"  z-statistic: {z_stat:.3f}")
    log(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        log(f"  [SIGNIFICANT] Slopes differ significantly between measures")
    else:
        log(f"  [NOT SIGNIFICANT] No significant slope difference")

    return slope_diff, se_diff, z_stat, p_value

def load_item_parameters():
    """
    Load item parameters from Ch5 (accuracy) and Ch6 (confidence).

    Follows v1 methodology:
    1. Filter items by discrimination threshold (0.25 to 4.0)
    2. Calculate mean from filtered items only

    Returns mean discrimination and difficulty for each measure.
    """
    log("\n[LOAD] Loading item parameters...")

    MIN_DISCRIM = 0.25
    MAX_DISCRIM = 4.0

    # Ch5 5.1.1: Accuracy (2PL)
    acc_items = pd.read_csv(PROJECT_ROOT / "results/ch5/5.1.1/logs/step01_item_parameters.csv")

    # Filter by discrimination threshold
    acc_filtered = acc_items[
        (acc_items['a'] >= MIN_DISCRIM) &
        (acc_items['a'] <= MAX_DISCRIM)
    ]

    acc_a = acc_filtered['a'].mean()
    acc_b = acc_filtered['b'].mean()

    log(f"  Accuracy (Ch5 5.1.1):")
    log(f"    Filtered {len(acc_filtered)}/{len(acc_items)} items (discrim {MIN_DISCRIM}-{MAX_DISCRIM})")
    log(f"    Mean a={acc_a:.3f}, mean b={acc_b:.3f}")

    # Ch6 6.1.1: Confidence (GRM)
    conf_items = pd.read_csv(PROJECT_ROOT / "results/ch6/6.1.1/data/step03_item_parameters.csv")

    # Filter by discrimination threshold
    conf_filtered = conf_items[
        (conf_items['a'] >= MIN_DISCRIM) &
        (conf_items['a'] <= MAX_DISCRIM)
    ]

    conf_a = conf_filtered['a'].mean()
    # For GRM, average item difficulty is mean of all thresholds
    conf_b = conf_filtered[['b1', 'b2', 'b3', 'b4']].mean(axis=1).mean()

    log(f"  Confidence (Ch6 6.1.1):")
    log(f"    Filtered {len(conf_filtered)}/{len(conf_items)} items (discrim {MIN_DISCRIM}-{MAX_DISCRIM})")
    log(f"    Mean a={conf_a:.3f}, mean b={conf_b:.3f} (mean of thresholds)")

    return acc_a, acc_b, conf_a, conf_b

def main():
    """Execute METHOD B analysis."""
    log("[START] METHOD B: LMM Slope Comparison Analysis")
    log("="*60)

    # ==========================================================================
    # STEP 9: Fit separate LMMs
    # ==========================================================================
    log("\n[STEP 9] Fitting separate LMMs for accuracy and confidence...")

    # Load merged data from Step 1
    df = pd.read_csv(RQ_DIR / "data" / "step01_merged_trajectories.csv")
    log(f"[LOADED] Data: {len(df)} observations")

    # Fit accuracy model
    acc_model, acc_stats = fit_lmm(df, 'theta_acc', 'Accuracy')

    # Fit confidence model
    conf_model, conf_stats = fit_lmm(df, 'theta_conf', 'Confidence')

    # Save model summaries
    model_df = pd.DataFrame([acc_stats, conf_stats])
    model_df.to_csv(RQ_DIR / "data" / "step09_lmm_models.csv", index=False)
    log(f"[SAVED] step09_lmm_models.csv")

    # ==========================================================================
    # STEP 10: Compare slopes on theta scale
    # ==========================================================================
    log("\n[STEP 10] Comparing slopes on theta scale...")

    acc_slope = acc_stats['slope']
    conf_slope = conf_stats['slope']

    # Compute differences and ratios
    slope_diff_theta = conf_slope - acc_slope
    slope_ratio_theta = conf_slope / acc_slope if acc_slope != 0 else np.nan

    log(f"  Accuracy slope: {acc_slope:.6f} theta/hour")
    log(f"  Confidence slope: {conf_slope:.6f} theta/hour")
    log(f"  Slope difference: {slope_diff_theta:.6f}")
    log(f"  Slope ratio: {slope_ratio_theta:.3f}")

    # Interpretation
    if slope_diff_theta > 0:
        interpretation = "Confidence declines SLOWER than accuracy (opposite of hedging hypothesis)"
    elif slope_diff_theta < 0:
        interpretation = "Confidence declines FASTER than accuracy (supports hedging hypothesis)"
    else:
        interpretation = "Parallel decline"

    log(f"  Interpretation: {interpretation}")

    # Test slope difference using z-test
    slope_diff_test, se_diff, z_stat, interaction_p = test_slope_difference(acc_stats, conf_stats)

    # Save slope comparison
    slope_comparison = pd.DataFrame([{
        'acc_slope_theta': acc_slope,
        'conf_slope_theta': conf_slope,
        'slope_diff_theta': slope_diff_theta,
        'slope_ratio_theta': slope_ratio_theta,
        'z_statistic': z_stat,
        'p_value': interaction_p,
        'interpretation': interpretation
    }])
    slope_comparison.to_csv(RQ_DIR / "data" / "step10_slope_comparison_theta.csv", index=False)
    log(f"[SAVED] step10_slope_comparison_theta.csv")

    # ==========================================================================
    # STEP 11: Transform LMM slopes to probability scale
    # ==========================================================================
    log("\n[STEP 11] Transforming LMM slopes to probability scale...")

    # Load calibrated item parameters
    acc_a, acc_b, conf_a, conf_b = load_item_parameters()

    # Use baseline theta values to evaluate derivative
    baseline_acc = df[df['test'] == 1]['theta_acc'].mean()
    baseline_conf = df[df['test'] == 1]['theta_conf'].mean()

    # Transform baseline theta to probability using calibrated parameters
    prob_baseline_acc = convert_theta_to_probability(baseline_acc, discrimination=acc_a, difficulty=acc_b)
    prob_baseline_conf = convert_theta_to_probability(baseline_conf, discrimination=conf_a, difficulty=conf_b)

    # Compute derivative dP/dθ at baseline
    # For 2PL: dP/dθ = a * P(1-P)
    dP_dtheta_acc = acc_a * prob_baseline_acc * (1 - prob_baseline_acc)
    dP_dtheta_conf = conf_a * prob_baseline_conf * (1 - prob_baseline_conf)

    # Transform theta slopes to probability slopes using chain rule
    # dP/dt = (dP/dθ) × (dθ/dt)
    prob_slope_acc = dP_dtheta_acc * acc_slope
    prob_slope_conf = dP_dtheta_conf * conf_slope

    # Compute difference and ratio
    prob_slope_diff = prob_slope_conf - prob_slope_acc
    prob_slope_ratio = prob_slope_conf / prob_slope_acc if prob_slope_acc != 0 else np.nan

    log(f"  Item parameters used:")
    log(f"    Accuracy: a={acc_a:.3f}, b={acc_b:.3f}")
    log(f"    Confidence: a={conf_a:.3f}, b={conf_b:.3f}")
    log(f"  Baseline theta and probabilities:")
    log(f"    Accuracy: theta={baseline_acc:.3f} -> P={prob_baseline_acc:.1%}")
    log(f"    Confidence: theta={baseline_conf:.3f} -> P={prob_baseline_conf:.1%}")
    log(f"  Probability derivatives at baseline (dP/dθ = a*P(1-P)):")
    log(f"    Accuracy: {dP_dtheta_acc:.4f}")
    log(f"    Confidence: {dP_dtheta_conf:.4f}")
    log(f"  Probability slopes (dP/dt):")
    log(f"    Accuracy: {prob_slope_acc:.6f} probability/hour")
    log(f"    Confidence: {prob_slope_conf:.6f} probability/hour")
    log(f"  Slope difference: {prob_slope_diff:.6f} probability/hour")
    log(f"  Slope ratio: {prob_slope_ratio:.3f}")

    # Interpretation
    if prob_slope_diff > 0:
        prob_interpretation = "Confidence declines SLOWER than accuracy on probability scale"
    elif prob_slope_diff < 0:
        prob_interpretation = "Confidence declines FASTER than accuracy on probability scale"
    else:
        prob_interpretation = "Parallel decline on probability scale"

    log(f"  Interpretation: {prob_interpretation}")

    # Save probability comparison
    prob_comparison = pd.DataFrame([{
        'acc_a': acc_a,
        'acc_b': acc_b,
        'conf_a': conf_a,
        'conf_b': conf_b,
        'baseline_theta_acc': baseline_acc,
        'baseline_theta_conf': baseline_conf,
        'prob_baseline_acc': prob_baseline_acc,
        'prob_baseline_conf': prob_baseline_conf,
        'dP_dtheta_acc': dP_dtheta_acc,
        'dP_dtheta_conf': dP_dtheta_conf,
        'prob_slope_acc': prob_slope_acc,
        'prob_slope_conf': prob_slope_conf,
        'prob_slope_diff': prob_slope_diff,
        'prob_slope_ratio': prob_slope_ratio,
        'interpretation': prob_interpretation
    }])
    prob_comparison.to_csv(RQ_DIR / "data" / "step11_slope_comparison_probability.csv", index=False)
    log(f"[SAVED] step11_slope_comparison_probability.csv")

    # ==========================================================================
    # STEP 12: Method comparison
    # ==========================================================================
    log("\n[STEP 12] Comparing METHOD A vs METHOD B...")

    # Load METHOD A results
    method_a = pd.read_csv(RQ_DIR / "data" / "step02_individual_decline_rates.csv")
    method_a_mean_diff = method_a['difference'].mean()
    method_a_p = pd.read_csv(RQ_DIR / "data" / "step03_paired_ttest_results.csv").iloc[0]['p_uncorrected']

    # METHOD B results
    method_b_diff = slope_diff_theta
    method_b_p = interaction_p

    log(f"  METHOD A (individual rates): mean_diff={method_a_mean_diff:.6f}, p={method_a_p:.4f}")
    log(f"  METHOD B (LMM slopes): slope_diff={method_b_diff:.6f}, p={method_b_p:.4f}")

    # Check agreement
    if (method_a_mean_diff > 0 and method_b_diff > 0) or (method_a_mean_diff < 0 and method_b_diff < 0):
        agreement = "AGREE on direction"
    else:
        agreement = "DIVERGE on direction"

    if (method_a_p < 0.05 and method_b_p < 0.05) or (method_a_p >= 0.05 and method_b_p >= 0.05):
        significance_agreement = "AGREE on significance"
    else:
        significance_agreement = "DIVERGE on significance"

    log(f"  Direction: {agreement}")
    log(f"  Significance: {significance_agreement}")

    # Compute correlation between individual rates and model predictions
    # For each person, compute model-predicted decline
    model_predictions = []
    for uid in method_a['UID']:
        uid_data = df[df['UID'] == uid]
        t1_hours = uid_data[uid_data['test'] == 1]['TSVR_hours'].values[0]
        t4_hours = uid_data[uid_data['test'] == 4]['TSVR_hours'].values[0]

        # Model-predicted decline
        pred_acc_decline = acc_slope * (t4_hours - t1_hours)
        pred_conf_decline = conf_slope * (t4_hours - t1_hours)
        pred_diff = pred_conf_decline - pred_acc_decline

        model_predictions.append(pred_diff)

    # Correlation between methods
    corr = np.corrcoef(method_a['difference'], model_predictions)[0, 1]
    log(f"  Correlation between methods: r={corr:.3f}")

    # Save comparison
    comparison = pd.DataFrame([{
        'method_a_mean_diff': method_a_mean_diff,
        'method_a_p': method_a_p,
        'method_b_slope_diff': method_b_diff,
        'method_b_p': method_b_p,
        'direction_agreement': agreement,
        'significance_agreement': significance_agreement,
        'correlation': corr
    }])
    comparison.to_csv(RQ_DIR / "data" / "step12_method_comparison.csv", index=False)
    log(f"[SAVED] step12_method_comparison.csv")

    # ==========================================================================
    # STEP 13: Extended trajectory analysis
    # ==========================================================================
    log("\n[STEP 13] Extended trajectory analysis...")

    # Note: Ch5 5.1.1 found Logarithmic best, Ch6 6.1.1 found Sin+Cos best
    # This suggests different functional forms which could explain divergence

    log("  Ch5 5.1.1 best model: Logarithmic")
    log("  Ch6 6.1.1 best model: Sin+Cos")
    log("  Different functional forms may explain why simple linear comparison misleading")

    # Compute area between curves (simplified - using linear approximation)
    time_points = df['TSVR_hours'].unique()
    time_points.sort()

    acc_trajectory = []
    conf_trajectory = []
    for t in time_points:
        acc_trajectory.append(df[df['TSVR_hours'] == t]['theta_acc'].mean())
        conf_trajectory.append(df[df['TSVR_hours'] == t]['theta_conf'].mean())

    # Area between curves (using trapezoidal rule)
    # Note: scipy.integrate.trapz renamed to trapezoid in newer versions
    try:
        from scipy.integrate import trapezoid
        abc = trapezoid(np.abs(np.array(acc_trajectory) - np.array(conf_trajectory)), time_points)
    except ImportError:
        # Fallback to numpy trapz
        abc = np.trapz(np.abs(np.array(acc_trajectory) - np.array(conf_trajectory)), time_points)
    log(f"  Area between curves: {abc:.2f} theta*hours")

    # Save trajectory analysis
    traj_analysis = pd.DataFrame([{
        'acc_functional_form': 'Logarithmic (Ch5 5.1.1)',
        'conf_functional_form': 'Sin+Cos (Ch6 6.1.1)',
        'area_between_curves': abc,
        'interpretation': 'Different functional forms suggest linear comparison oversimplifies'
    }])
    traj_analysis.to_csv(RQ_DIR / "data" / "step13_trajectory_analysis.csv", index=False)
    log(f"[SAVED] step13_trajectory_analysis.csv")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    log("\n" + "="*60)
    log("[SUMMARY] METHOD B Analysis Complete")
    log(f"\nTheta scale (LMM slopes):")
    log(f"  Accuracy slope: {acc_slope:.6f} theta/hour")
    log(f"  Confidence slope: {conf_slope:.6f} theta/hour")
    log(f"  Slope difference: {slope_diff_theta:.6f}")
    log(f"  Slope ratio: {slope_ratio_theta:.3f}")
    log(f"  z-statistic: {z_stat:.3f}, p={interaction_p:.4f}")
    log(f"  Interpretation: {interpretation}")

    log(f"\nProbability scale (transformed LMM slopes):")
    log(f"  Accuracy slope: {prob_slope_acc:.6f} probability/hour")
    log(f"  Confidence slope: {prob_slope_conf:.6f} probability/hour")
    log(f"  Slope difference: {prob_slope_diff:.6f}")
    log(f"  Slope ratio: {prob_slope_ratio:.3f}")
    log(f"  Interpretation: {prob_interpretation}")

    log(f"\nMethod comparison:")
    log(f"  Methods {agreement} on direction")
    log(f"  Methods {significance_agreement} on significance")

    if agreement == "DIVERGE on direction":
        log("\n[WARNING] METHOD A and METHOD B produce opposite conclusions!")
        log("This demonstrates how excluding half the data (T2, T3) can be misleading")

    log("\n[SUCCESS] Extended analysis complete")

if __name__ == "__main__":
    # Clear log file
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("")

    try:
        main()
        sys.exit(0)
    except Exception as e:
        log(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)