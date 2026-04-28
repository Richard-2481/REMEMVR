#!/usr/bin/env python3
"""
Step 07: Random Slopes Comparison (MANDATORY VALIDATION REQUIREMENT)

PURPOSE:
Test whether random slopes improve model fit over intercepts-only models.
This resolves the validation blocker identified by validation process (2025-12-31).

CRITICAL CONTEXT:
- Current analysis uses random intercepts + slopes: `~ 1 + Days | UID`
- var_slope = 0.098, ICC_slope = 21.6% interpretation REQUIRES demonstrating slopes needed
- Taxonomy Section 4.4 (added 2025-12-11) mandates random slopes testing

APPROACH:
1. Load existing model comparison results (10 competitive models, intercepts+slopes)
2. Refit 10 competitive models with intercepts-only: `~ 1 | UID`
3. Compare AIC: ΔAIC = AIC(intercepts_only) - AIC(intercepts+slopes)
4. Decision: If ΔAIC > 2.0 → Slopes improve fit → BLOCKER RESOLVED

EXPECTED INPUTS:
  1. ../5.1.1/data/step04_lmm_input.csv (LMM input data)
  2. data/step06_competitive_models_metadata.yaml (list of 10 competitive models)
  3. data/step06_model_comparison.csv (existing AIC values with slopes)

EXPECTED OUTPUTS:
  1. data/step07_random_slopes_comparison.csv
     Columns: [model_name, AIC_with_slopes, AIC_intercepts_only, delta_AIC, slopes_improve_fit]
     Purpose: Comparison table for random slopes justification

  2. logs/step07_random_slopes_comparison.log
     Purpose: Execution log with decision criteria

VALIDATION CRITERIA:
  1. All 10 competitive models converge (both with slopes and intercepts-only)
  2. ΔAIC > 2.0 for MAJORITY of models → Slopes justified
  3. No NaN values in comparison table
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
from datetime import datetime
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_selection import compare_lmm_models_kitchen_sink

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_random_slopes_comparison.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Input paths
LMM_INPUT = RQ_DIR / "../5.1.1/data/step04_lmm_input.csv"
COMPETITIVE_MODELS_YAML = RQ_DIR / "data" / "step06_competitive_models_metadata.yaml"
MODEL_COMPARISON_CSV = RQ_DIR / "data" / "step06_model_comparison.csv"

# Output paths
COMPARISON_CSV = RQ_DIR / "data" / "step07_random_slopes_comparison.csv"

COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)

# Logging Function

def log(msg: str) -> None:
    """Write message to both console and log file (UTF-8 encoding)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted_msg + "\n")


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Random Slopes Comparison")
        log("=" * 70)
        log("validation blocker resolution - testing if random slopes needed")
        log("Section 4.4 (added 2025-12-11) mandates random slopes testing")
        log("")
        # Load Inputs
        log("Loading LMM input data from RQ 5.1.1...")
        if not LMM_INPUT.exists():
            log(f"LMM input file not found: {LMM_INPUT}")
            sys.exit(1)

        lmm_input = pd.read_csv(LMM_INPUT, encoding='utf-8')
        log(f"{len(lmm_input)} rows, {lmm_input['UID'].nunique()} participants")

        log("Loading competitive models metadata...")
        if not COMPETITIVE_MODELS_YAML.exists():
            log(f"Competitive models YAML not found: {COMPETITIVE_MODELS_YAML}")
            sys.exit(1)

        with open(COMPETITIVE_MODELS_YAML, 'r', encoding='utf-8') as f:
            competitive_meta = yaml.safe_load(f)

        competitive_models = competitive_meta['competitive_models']
        log(f"{len(competitive_models)} competitive models")
        for model_name in competitive_models:
            log(f"  - {model_name}")

        log("Loading existing model comparison (with slopes)...")
        if not MODEL_COMPARISON_CSV.exists():
            log(f"Model comparison CSV not found: {MODEL_COMPARISON_CSV}")
            sys.exit(1)

        model_comparison = pd.read_csv(MODEL_COMPARISON_CSV, encoding='utf-8')
        log(f"{len(model_comparison)} models in comparison table")
        log("")
        # Extract Existing AICs (With Slopes)
        log("Extracting AIC values for competitive models (with slopes)...")

        aic_with_slopes = {}
        for model_name in competitive_models:
            model_row = model_comparison[model_comparison['model_name'] == model_name]
            if len(model_row) == 0:
                log(f"Model {model_name} not found in comparison table")
                continue
            aic_with_slopes[model_name] = model_row.iloc[0]['AIC']
            log(f"  {model_name}: AIC = {aic_with_slopes[model_name]:.2f}")

        log(f"Extracted {len(aic_with_slopes)} AIC values")
        log("")
        # Fit Intercepts-Only Models Using Kitchen Sink
        log("Fitting intercepts-only models (re_formula='~1')...")
        log("This will take ~2-3 minutes...")
        log("")

        # Use kitchen sink with intercepts-only
        try:
            intercepts_only_results = compare_lmm_models_kitchen_sink(
                data=lmm_input,
                outcome_var='theta',
                tsvr_var='TSVR_hours',
                groups_var='UID',
                re_formula='~1',  # Intercepts only
                reml=False,       # Use ML for AIC comparison
                return_models=False,
                save_dir=None,
                log_file=None,
            )
        except Exception as e:
            log(f"Kitchen sink (intercepts-only) failed: {e}")
            sys.exit(1)

        comparison_intercepts = intercepts_only_results['comparison']
        log(f"{len(comparison_intercepts)} intercepts-only models fitted")
        log("")
        # Extract AICs for Competitive Models (Intercepts-Only)
        log("Extracting AIC values (intercepts-only)...")

        aic_intercepts_only = {}
        for model_name in competitive_models:
            model_row = comparison_intercepts[comparison_intercepts['model_name'] == model_name]
            if len(model_row) == 0:
                log(f"Model {model_name} failed to converge (intercepts-only)")
                aic_intercepts_only[model_name] = np.nan
                continue
            aic_intercepts_only[model_name] = model_row.iloc[0]['AIC']
            log(f"  {model_name}: AIC = {aic_intercepts_only[model_name]:.2f}")

        log(f"Extracted {sum(~pd.isna(list(aic_intercepts_only.values())))}/{len(competitive_models)} AIC values")
        log("")
        # Compute ΔAIC and Decision
        log("Computing ΔAIC = AIC(intercepts_only) - AIC(with_slopes)...")
        log("")

        comparison_results = []
        for model_name in competitive_models:
            if model_name not in aic_with_slopes or model_name not in aic_intercepts_only:
                log(f"Skipping {model_name} (missing AIC values)")
                continue

            aic_slopes = aic_with_slopes[model_name]
            aic_intcpt = aic_intercepts_only[model_name]

            if pd.isna(aic_intcpt):
                log(f"Skipping {model_name} (intercepts-only failed)")
                continue

            delta_aic = aic_intcpt - aic_slopes
            slopes_improve = delta_aic > 2.0

            comparison_results.append({
                'model_name': model_name,
                'AIC_with_slopes': aic_slopes,
                'AIC_intercepts_only': aic_intcpt,
                'delta_AIC': delta_aic,
                'slopes_improve_fit': slopes_improve,
            })

            symbol = "✓" if slopes_improve else "✗"
            log(f"  {symbol} {model_name}:")
            log(f"      AIC (slopes):    {aic_slopes:.2f}")
            log(f"      AIC (int-only):  {aic_intcpt:.2f}")
            log(f"      ΔAIC:            {delta_aic:+.2f} ({'slopes IMPROVE' if slopes_improve else 'slopes NOT needed'})")

        comparison_df = pd.DataFrame(comparison_results)

        log("")
        log("=" * 70)
        log("Random Slopes Comparison Results")
        log("=" * 70)

        # Overall statistics
        n_models = len(comparison_df)
        n_improved = comparison_df['slopes_improve_fit'].sum()
        pct_improved = (n_improved / n_models) * 100 if n_models > 0 else 0
        median_delta_aic = comparison_df['delta_AIC'].median()
        mean_delta_aic = comparison_df['delta_AIC'].mean()

        log(f"  Models tested: {n_models}")
        log(f"  Models where slopes improve fit (ΔAIC > 2.0): {n_improved}/{n_models} ({pct_improved:.1f}%)")
        log(f"  Median ΔAIC: {median_delta_aic:+.2f}")
        log(f"  Mean ΔAIC:   {mean_delta_aic:+.2f}")
        log("")

        # Decision criteria
        if n_improved >= n_models * 0.7:  # 70% threshold
            log("✓ RANDOM SLOPES JUSTIFIED")
            log(f"  Rationale: {n_improved}/{n_models} models ({pct_improved:.1f}%) show ΔAIC > 2.0")
            log(f"  Interpretation: Individual heterogeneity in forgetting rates EXISTS")
            log(f"  var_slope = 0.098 and ICC_slope = 21.6% interpretations VALID")
            log("")
            log("✓ RESOLVED - validation criterion met")
            decision = "SLOPES_JUSTIFIED"
        else:
            log("✗ RANDOM SLOPES NOT JUSTIFIED")
            log(f"  Rationale: Only {n_improved}/{n_models} models ({pct_improved:.1f}%) show ΔAIC > 2.0")
            log(f"  Interpretation: Individual differences in slopes may be artifact")
            log(f"  var_slope = 0.098 interpretation QUESTIONABLE")
            log("")
            log("✗ NOT RESOLVED - Need to reconsider ICC interpretation")
            decision = "SLOPES_NOT_JUSTIFIED"

        log("=" * 70)
        log("")
        # Save Results
        log("Saving comparison results to CSV...")

        comparison_df.to_csv(COMPARISON_CSV, index=False, encoding='utf-8')
        log(f"{COMPARISON_CSV}")
        log("")
        # VALIDATION
        log("Running output validation...")

        if comparison_df.isna().sum().sum() > 0:
            log(f"Comparison table contains NaN values")
            sys.exit(1)
        log(f"No NaN values in comparison table")

        if len(comparison_df) < 10:
            log(f"Expected 10 models, got {len(comparison_df)}")
        else:
            log(f"All 10 competitive models compared")

        log("")
        log("=" * 70)
        log("Step 07 complete: Random slopes comparison")
        log(f"  Decision: {decision}")
        log(f"  Models improved by slopes: {n_improved}/{n_models} ({pct_improved:.1f}%)")
        log(f"  Median ΔAIC: {median_delta_aic:+.2f}")
        log(f"  Output: {COMPARISON_CSV}")
        log("=" * 70)

        # Exit with code based on decision
        if decision == "SLOPES_JUSTIFIED":
            sys.exit(0)  # Success
        else:
            sys.exit(2)  # Slopes not justified (non-zero exit for caller awareness)

    except Exception as e:
        log(f"Unexpected error: {str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
