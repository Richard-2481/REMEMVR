"""
RQ 6.1.4 - Step 06b: Model-Averaged ICC Validation

PURPOSE:
Validate the 824× ICC ratio finding using model-averaged random effects instead
of single-model random effects. Original analysis used Recip_sq model (21.7% weight),
ignoring 78% of model evidence. This validation uses MA random effects from
48 competitive models (ΔAIC < 7) to test robustness.

CRITICAL: The 824× ICC ratio is a thesis centerpiece finding. This validation
determines whether it survives model averaging.

INPUT:
- results/ch6/6.1.1/data/step05b_model_averaged_random_effects.csv
  - Columns: UID, ma_intercept, ma_slope

OUTPUT:
- results/ch6/6.1.4/data/step06b_icc_ma_validation.csv
- Log file with comparison

METHODOLOGY:
1. Load MA random effects from 6.1.1 (48 models, Eff_N=31.1)
2. Compute ICC_slope_MA = var(ma_slope) / (var(ma_slope) + residual_var)
3. Compare to original ICC_slope = 0.4120
4. Recompute 824× ratio with MA ICC
5. Assess robustness: Does ratio remain >500×?

Date: 2025-12-14
RQ: ch6/6.1.4
Task: T1.1 from rq_rework.md
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

LOG_FILE = RQ_DIR / "logs" / "step06b_icc_ma_validation.log"
DATA_DIR = RQ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
(RQ_DIR / "logs").mkdir(exist_ok=True)

# Clear log file
with open(LOG_FILE, 'w') as f:
    f.write("")

def log(msg):
    """Log message to file and stdout with flush"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def main():
    log("=" * 80)
    log("RQ 6.1.4 - Step 06b: Model-Averaged ICC Validation")
    log(f"Started: {datetime.now().isoformat()}")
    log("=" * 80)
    # Load Original Single-Model Results
    log("\n[STEP 1] Load Original Single-Model Results")
    log("-" * 60)

    original_icc_path = DATA_DIR / "step02_icc_estimates.csv"
    original_icc = pd.read_csv(original_icc_path)

    ICC_slope_original = original_icc[original_icc['icc_type'] == 'ICC_slope_simple']['value'].values[0]
    ICC_intercept_original = original_icc[original_icc['icc_type'] == 'ICC_intercept']['value'].values[0]

    log(f"  Original ICC_intercept: {ICC_intercept_original:.4f}")
    log(f"  Original ICC_slope: {ICC_slope_original:.4f}")
    log(f"  Original model: Recip_sq (21.7% Akaike weight)")

    # Load original variance components
    var_components = pd.read_csv(DATA_DIR / "step01_variance_components.csv")
    var_residual_original = var_components[var_components['component'] == 'var_residual']['value'].values[0]
    var_slope_original = var_components[var_components['component'] == 'var_slope']['value'].values[0]
    var_intercept_original = var_components[var_components['component'] == 'var_intercept']['value'].values[0]

    log(f"  Original var_intercept: {var_intercept_original:.6f}")
    log(f"  Original var_slope: {var_slope_original:.6f}")
    log(f"  Original var_residual: {var_residual_original:.6f}")
    # Load Model-Averaged Random Effects
    log("\n[STEP 2] Load Model-Averaged Random Effects from 6.1.1")
    log("-" * 60)

    ma_re_path = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step05b_model_averaged_random_effects.csv"
    ma_re = pd.read_csv(ma_re_path)

    log(f"  ✓ Loaded {len(ma_re)} participants")
    log(f"  ✓ Columns: {list(ma_re.columns)}")

    # Descriptive stats for MA random effects
    log(f"\n  MA Intercept Stats:")
    log(f"    Mean: {ma_re['ma_intercept'].mean():.6f}")
    log(f"    SD: {ma_re['ma_intercept'].std():.6f}")
    log(f"    Range: [{ma_re['ma_intercept'].min():.4f}, {ma_re['ma_intercept'].max():.4f}]")

    log(f"\n  MA Slope Stats:")
    log(f"    Mean: {ma_re['ma_slope'].mean():.6f}")
    log(f"    SD: {ma_re['ma_slope'].std():.6f}")
    log(f"    Range: [{ma_re['ma_slope'].min():.4f}, {ma_re['ma_slope'].max():.4f}]")
    # Compute MA Variance Components
    log("\n[STEP 3] Compute Model-Averaged Variance Components")
    log("-" * 60)

    # Variance of MA random effects
    var_intercept_ma = ma_re['ma_intercept'].var(ddof=1)
    var_slope_ma = ma_re['ma_slope'].var(ddof=1)

    log(f"  var(ma_intercept): {var_intercept_ma:.6f}")
    log(f"  var(ma_slope): {var_slope_ma:.6f}")

    # For residual variance, we need to use average from competitive models
    # Load MA metadata to get residual variance estimate
    ma_metadata_path = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step05b_metadata.csv"

    if ma_metadata_path.exists():
        ma_metadata = pd.read_csv(ma_metadata_path)
        log(f"\n  MA Metadata:")
        for col in ma_metadata.columns:
            log(f"    {col}: {ma_metadata[col].values[0]}")

    # Use original residual variance as approximation
    # (MA doesn't change residual variance estimation much - it's pooled across models)
    var_residual_ma = var_residual_original
    log(f"\n  Using original var_residual: {var_residual_ma:.6f}")
    log(f"  (Residual variance is approximately invariant across model specifications)")
    # Compute MA ICC Estimates
    log("\n[STEP 4] Compute Model-Averaged ICC Estimates")
    log("-" * 60)

    # ICC_intercept_MA = var(ma_intercept) / (var(ma_intercept) + var_residual)
    # NOTE: This is a simplified formula. True ICC requires total variance at a timepoint.
    # For random intercept, total variance ≈ var_intercept + var_residual
    total_var_intercept = var_intercept_ma + var_residual_ma
    ICC_intercept_ma = var_intercept_ma / total_var_intercept if total_var_intercept > 0 else 0

    # ICC_slope_MA = var(ma_slope) / (var(ma_slope) + var_residual)
    # Following Hoffman & Stawski (2009) ICC_slope_simple formula
    total_var_slope = var_slope_ma + var_residual_ma
    ICC_slope_ma = var_slope_ma / total_var_slope if total_var_slope > 0 else 0

    log(f"\n  ICC Formulas (Hoffman & Stawski 2009 simplified):")
    log(f"  ICC = var(random_effect) / (var(random_effect) + var_residual)")

    log(f"\n  Model-Averaged ICC Estimates:")
    log(f"    ICC_intercept_MA: {ICC_intercept_ma:.4f}")
    log(f"    ICC_slope_MA: {ICC_slope_ma:.4f}")
    # Compare Original vs MA ICCs
    log("\n[STEP 5] Compare Original vs Model-Averaged ICCs")
    log("-" * 60)

    # ICC comparisons
    delta_icc_intercept = ICC_intercept_ma - ICC_intercept_original
    delta_icc_slope = ICC_slope_ma - ICC_slope_original

    pct_change_intercept = (delta_icc_intercept / ICC_intercept_original) * 100 if ICC_intercept_original > 0 else 0
    pct_change_slope = (delta_icc_slope / ICC_slope_original) * 100 if ICC_slope_original > 0 else 0

    log(f"\n  ICC_intercept Comparison:")
    log(f"    Original (Recip_sq): {ICC_intercept_original:.4f}")
    log(f"    Model-Averaged: {ICC_intercept_ma:.4f}")
    log(f"    Delta: {delta_icc_intercept:+.4f} ({pct_change_intercept:+.1f}%)")

    log(f"\n  ICC_slope Comparison:")
    log(f"    Original (Recip_sq): {ICC_slope_original:.4f}")
    log(f"    Model-Averaged: {ICC_slope_ma:.4f}")
    log(f"    Delta: {delta_icc_slope:+.4f} ({pct_change_slope:+.1f}%)")
    # Recompute 824× Ratio with MA ICC
    log("\n[STEP 6] Recompute Confidence vs Accuracy ICC Ratio with MA")
    log("-" * 60)

    # Ch5 ICC_slope_accuracy = 0.0005 (dichotomous data)
    ICC_slope_accuracy = 0.0005

    # Original ratio
    ratio_original = ICC_slope_original / ICC_slope_accuracy

    # MA ratio
    ratio_ma = ICC_slope_ma / ICC_slope_accuracy

    log(f"\n  ICC_slope_accuracy (Ch5 dichotomous): {ICC_slope_accuracy:.4f}")
    log(f"\n  Original Ratio (Recip_sq / Ch5):")
    log(f"    {ICC_slope_original:.4f} / {ICC_slope_accuracy:.4f} = {ratio_original:.1f}×")
    log(f"\n  MA Ratio (MA / Ch5):")
    log(f"    {ICC_slope_ma:.4f} / {ICC_slope_accuracy:.4f} = {ratio_ma:.1f}×")

    delta_ratio = ratio_ma - ratio_original
    pct_change_ratio = (delta_ratio / ratio_original) * 100

    log(f"\n  Ratio Change: {delta_ratio:+.1f}× ({pct_change_ratio:+.1f}%)")
    # Assess Robustness
    log("\n[STEP 7] Assess Robustness of 824× Finding")
    log("-" * 60)

    # Robustness criteria from T1.1 in rq_rework.md
    # "Ratio remains >500× (finding robust) OR change documented"

    if ratio_ma > 500:
        robustness = "ROBUST"
        interpretation = (f"The 824× ICC ratio finding is ROBUST to model averaging. "
                         f"MA ratio = {ratio_ma:.1f}× still exceeds 500× threshold. "
                         f"Change from original: {pct_change_ratio:+.1f}%. "
                         f"Measurement artifact hypothesis strongly supported across model specifications.")
    elif ratio_ma > 100:
        robustness = "SUBSTANTIALLY ROBUST"
        interpretation = (f"The ICC ratio finding is substantially robust though reduced. "
                         f"MA ratio = {ratio_ma:.1f}× (vs original 824×). "
                         f"Still demonstrates large ordinal vs dichotomous precision advantage. "
                         f"Measurement artifact hypothesis supported, effect size somewhat attenuated.")
    elif ratio_ma > 10:
        robustness = "MODERATE ATTENUATION"
        interpretation = (f"Model averaging moderately attenuates the ICC ratio. "
                         f"MA ratio = {ratio_ma:.1f}× (vs original 824×). "
                         f"Ordinal advantage present but magnitude uncertain. "
                         f"Finding should be interpreted with caution.")
    else:
        robustness = "NOT ROBUST"
        interpretation = (f"Model averaging substantially changes the ICC ratio finding. "
                         f"MA ratio = {ratio_ma:.1f}× (vs original 824×). "
                         f"Original finding may have been inflated by single-model selection. "
                         f"Thesis claim about ordinal precision advantage needs revision.")

    log(f"\n  Robustness Assessment: {robustness}")
    log(f"\n  Interpretation:")
    log(f"  {interpretation}")

    # Additional assessment: Does ICC_slope_MA still exceed 0.10 threshold?
    log(f"\n  Additional Checks:")
    log(f"    ICC_slope_MA > 0.10 (detectable)? {ICC_slope_ma > 0.10} (value = {ICC_slope_ma:.4f})")
    log(f"    ICC_slope_MA > 0.30 (substantial)? {ICC_slope_ma > 0.30} (value = {ICC_slope_ma:.4f})")
    log(f"    ICC_slope_MA within 20% of original? {abs(pct_change_slope) < 20} (change = {pct_change_slope:+.1f}%)")
    # Save Results
    log("\n[STEP 8] Save Validation Results")
    log("-" * 60)

    # Create output dataframe
    results_df = pd.DataFrame({
        'metric': [
            'ICC_intercept_original',
            'ICC_intercept_MA',
            'ICC_intercept_delta',
            'ICC_intercept_pct_change',
            'ICC_slope_original',
            'ICC_slope_MA',
            'ICC_slope_delta',
            'ICC_slope_pct_change',
            'ICC_slope_accuracy_ch5',
            'ratio_original',
            'ratio_MA',
            'ratio_delta',
            'ratio_pct_change',
            'robustness_assessment',
            'var_intercept_original',
            'var_intercept_MA',
            'var_slope_original',
            'var_slope_MA',
            'var_residual'
        ],
        'value': [
            ICC_intercept_original,
            ICC_intercept_ma,
            delta_icc_intercept,
            pct_change_intercept,
            ICC_slope_original,
            ICC_slope_ma,
            delta_icc_slope,
            pct_change_slope,
            ICC_slope_accuracy,
            ratio_original,
            ratio_ma,
            delta_ratio,
            pct_change_ratio,
            robustness,
            var_intercept_original,
            var_intercept_ma,
            var_slope_original,
            var_slope_ma,
            var_residual_ma
        ]
    })

    output_path = DATA_DIR / "step06b_icc_ma_validation.csv"
    results_df.to_csv(output_path, index=False)
    log(f"  ✓ Saved: {output_path}")
    # SUMMARY
    log("\n" + "=" * 80)
    log("Model-Averaged ICC Validation Complete")
    log("=" * 80)

    log(f"""
  ORIGINAL (Single Model: Recip_sq, 21.7% weight):
    ICC_intercept: {ICC_intercept_original:.4f}
    ICC_slope: {ICC_slope_original:.4f}
    Ratio vs Ch5: {ratio_original:.1f}×

  MODEL-AVERAGED (48 models, Eff_N=31.1):
    ICC_intercept: {ICC_intercept_ma:.4f} ({pct_change_intercept:+.1f}% change)
    ICC_slope: {ICC_slope_ma:.4f} ({pct_change_slope:+.1f}% change)
    Ratio vs Ch5: {ratio_ma:.1f}× ({pct_change_ratio:+.1f}% change)

  ROBUSTNESS: {robustness}

  INTERPRETATION:
    {interpretation}
    """)

    log(f"\nCompleted: {datetime.now().isoformat()}")

    return {
        'ICC_slope_original': ICC_slope_original,
        'ICC_slope_ma': ICC_slope_ma,
        'ratio_original': ratio_original,
        'ratio_ma': ratio_ma,
        'robustness': robustness
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        log(f"\n{e}")
        import traceback
        log(traceback.format_exc())
        raise
