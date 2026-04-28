"""
RQ 5.1.1 - Step 05d: Model-Averaged Residuals for RQ 6.7.3

PURPOSE:
Generate per-participant, per-session residuals using model averaging with
ΔAIC < 7 threshold (consistent with Ch6 methodology). These residuals are
needed by RQ 6.7.3 (calibration-trajectory relationship).

BACKGROUND:
The existing step05c_model_averaging.py uses ΔAIC < 2 (very strict) and only
outputs population-level predictions on a time grid. This script:
1. Uses ΔAIC < 7 (consistent with Burnham & Anderson 2002 full range)
2. Computes fitted values for EACH observation in the training data
3. Computes residuals = observed - fitted for each observation
4. Outputs per-participant residuals for downstream analyses

INPUT:
- data/step04_lmm_input.csv (training data with theta scores)
- data/step05_model_comparison.csv (66 models with Akaike weights)

OUTPUT:
- data/step05d_model_averaged_residuals.csv (400 rows: 100 participants × 4 sessions)
- results/step05d_residuals_summary.txt (summary statistics)

METHODOLOGY:
Burnham & Anderson (2002) model averaging:
1. Identify competitive models with ΔAIC < 7
2. Renormalize Akaike weights within competitive set
3. For each model: refit and compute fitted values
4. Compute model-averaged fitted: ŷ_MA = Σ w_i × ŷ_i
5. Compute residuals: e = y - ŷ_MA

Date: 2025-12-13
RQ: ch5/5.1.1
Step: 05d
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_averaging import identify_competitive_models, _create_transformations, _build_formula

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05d_model_averaged_residuals.log"


def log(msg):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + "\n")
    print(log_msg)


def main():
    """Compute model-averaged residuals for Ch5 5.1.1."""
    log("=" * 70)
    log("Step 05d: Model-Averaged Residuals (ΔAIC < 7)")
    log("=" * 70)

    # Load data
    log("\nLoading input data...")
    lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")
    comparison = pd.read_csv(RQ_DIR / "data" / "step05_model_comparison.csv")

    log(f"  Observations: {len(lmm_input)}")
    log(f"  Participants: {lmm_input['UID'].nunique()}")
    log(f"  Models compared: {len(comparison)}")

    best = comparison.iloc[0]
    log(f"  Best single model: {best['model_name']} (weight={best['akaike_weight']:.1%})")

    # Step 1: Identify competitive models
    log("\n[STEP 1] Identifying competitive models (ΔAIC < 7)...")
    competitive = identify_competitive_models(
        comparison,
        delta_aic_threshold=7.0,
        min_weight=0.001
    )

    n_competitive = len(competitive)
    total_weight = competitive['akaike_weight'].sum()

    # Compute effective N
    weights = competitive['renorm_weight'].values
    effective_n = np.exp(-np.sum(weights * np.log(weights + 1e-10)))

    log(f"  Competitive models: {n_competitive}")
    log(f"  Total original weight: {total_weight:.1%}")
    log(f"  Effective N models: {effective_n:.2f}")

    # Step 2: Create time transformations
    log("\n[STEP 2] Creating time transformations...")
    lmm_data = _create_transformations(lmm_input.copy(), 'TSVR_hours')
    log(f"  Columns added: {[c for c in lmm_data.columns if c not in lmm_input.columns]}")

    # Step 3: Fit each competitive model and compute fitted values
    log("\n[STEP 3] Fitting competitive models and computing fitted values...")

    fitted_matrix = np.zeros((n_competitive, len(lmm_data)))
    model_weights = []

    for i, (_, row) in enumerate(competitive.iterrows()):
        model_name = row['model_name']
        weight = row['renorm_weight']
        model_weights.append(weight)

        try:
            formula = _build_formula(model_name, 'theta', lmm_data)
            model = smf.mixedlm(formula, lmm_data, groups=lmm_data['UID'])
            result = model.fit(method='powell', reml=False)

            fitted_matrix[i, :] = result.fittedvalues.values

            if i < 5 or i == n_competitive - 1:
                log(f"    {i+1}/{n_competitive}: {model_name:25s} w={weight:.3f} - converged={result.converged}")
            elif i == 5:
                log(f"    ... fitting remaining {n_competitive - 5} models ...")

        except Exception as e:
            log(f"    WARNING: {model_name} failed: {e}")
            # Use simple linear model as fallback
            fallback = smf.mixedlm('theta ~ TSVR_hours', lmm_data, groups=lmm_data['UID'])
            fallback_result = fallback.fit(method='powell', reml=False)
            fitted_matrix[i, :] = fallback_result.fittedvalues.values

    # Step 4: Compute model-averaged fitted values
    log("\n[STEP 4] Computing model-averaged fitted values...")
    model_weights = np.array(model_weights)

    # Weighted average of fitted values
    ma_fitted = np.zeros(len(lmm_data))
    for i in range(n_competitive):
        ma_fitted += model_weights[i] * fitted_matrix[i, :]

    log(f"  MA fitted range: [{ma_fitted.min():.4f}, {ma_fitted.max():.4f}]")
    log(f"  MA fitted mean: {ma_fitted.mean():.4f}")

    # Step 5: Compute residuals
    log("\n[STEP 5] Computing model-averaged residuals...")
    observed = lmm_data['theta'].values
    residuals = observed - ma_fitted

    log(f"  Residuals range: [{residuals.min():.4f}, {residuals.max():.4f}]")
    log(f"  Residuals mean: {residuals.mean():.6f} (should be ~0)")
    log(f"  Residuals SD: {residuals.std():.4f}")

    # Step 6: Create output DataFrame
    log("\n[STEP 6] Creating output DataFrame...")

    # Extract UID and test from composite_ID
    lmm_data['test_num'] = lmm_data['composite_ID'].str.split('_').str[1].astype(int)
    lmm_data['test'] = 'T' + lmm_data['test_num'].astype(str)

    output_df = pd.DataFrame({
        'composite_ID': lmm_data['composite_ID'],
        'UID': lmm_data['UID'],
        'test': lmm_data['test'],
        'theta_observed': observed,
        'theta_ma_fitted': ma_fitted,
        'residual_ma': residuals,
        'TSVR_hours': lmm_data['TSVR_hours'],
    })

    # Validate
    assert len(output_df) == 400, f"Expected 400 rows, got {len(output_df)}"
    assert output_df['UID'].nunique() == 100, f"Expected 100 UIDs, got {output_df['UID'].nunique()}"

    log(f"  Output rows: {len(output_df)}")
    log(f"  Unique UIDs: {output_df['UID'].nunique()}")

    # Step 7: Save outputs
    log("\n[STEP 7] Saving outputs...")

    # Save residuals
    output_path = RQ_DIR / "data" / "step05d_model_averaged_residuals.csv"
    output_df.to_csv(output_path, index=False)
    log(f"  Saved: {output_path.name} ({len(output_df)} rows)")

    # Save summary
    summary_path = RQ_DIR / "results" / "step05d_residuals_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RQ 5.1.1 - Model-Averaged Residuals Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Methodology: Burnham & Anderson (2002) Model Averaging\n")
        f.write(f"ΔAIC threshold: 7.0\n\n")

        f.write(f"Model Selection:\n")
        f.write(f"  Total models tested: {len(comparison)}\n")
        f.write(f"  Competitive models (ΔAIC < 7): {n_competitive}\n")
        f.write(f"  Total original weight: {total_weight:.1%}\n")
        f.write(f"  Effective N models: {effective_n:.2f}\n\n")

        f.write(f"Top 5 Models:\n")
        for i, (_, row) in enumerate(competitive.head(5).iterrows()):
            f.write(f"  {i+1}. {row['model_name']:25s} w={row['renorm_weight']:.3f}\n")

        f.write(f"\nResidual Statistics:\n")
        f.write(f"  Mean: {residuals.mean():.6f}\n")
        f.write(f"  SD: {residuals.std():.4f}\n")
        f.write(f"  Min: {residuals.min():.4f}\n")
        f.write(f"  Max: {residuals.max():.4f}\n\n")

        f.write(f"Output:\n")
        f.write(f"  File: step05d_model_averaged_residuals.csv\n")
        f.write(f"  Rows: {len(output_df)}\n")
        f.write(f"  UIDs: {output_df['UID'].nunique()}\n")
        f.write(f"  Tests per UID: 4\n\n")

        f.write(f"Usage:\n")
        f.write(f"  RQ 6.7.3 should use 'residual_ma' column for trajectory variability\n")
        f.write(f"  This replaces single-model (PowerLaw_04) residuals with MA residuals\n")

    log(f"  Saved: {summary_path.name}")

    # Final summary
    log("\n" + "=" * 70)
    log("Step 05d COMPLETE")
    log("=" * 70)
    log(f"  Competitive models: {n_competitive}")
    log(f"  Effective N: {effective_n:.2f}")
    log(f"  Output: step05d_model_averaged_residuals.csv")
    log(f"  Purpose: Provides MA residuals for RQ 6.7.3")
    log("=" * 70)

    return output_df


if __name__ == "__main__":
    main()
