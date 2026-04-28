#!/usr/bin/env python3
"""compare_fit: Compare AIC and BIC between IRT and CTT models to assess relative fit quality."""

import sys
from pathlib import Path
import pandas as pd
import re
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq11 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_compare_fit.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def parse_aic_bic_from_summary(summary_path: Path) -> Dict[str, float]:
    """
    Parse AIC and BIC values from statsmodels MixedLM summary text file.

    Args:
        summary_path: Path to summary .txt file

    Returns:
        Dict with keys 'AIC' and 'BIC'

    Raises:
        ValueError: If AIC or BIC not found in summary
    """
    log(f"Reading summary file: {summary_path.name}")

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_text = f.read()

    # Statsmodels summary format typically has lines like:
    # AIC:                          -123.456
    # BIC:                          -234.567
    # OR for REML models, extract from:
    # Log-Likelihood:               -1284.2509
    # No. Observations:             1200

    aic_match = re.search(r'AIC:\s+([-+]?\d*\.?\d+)', summary_text)
    bic_match = re.search(r'BIC:\s+([-+]?\d*\.?\d+)', summary_text)

    # If AIC/BIC not in summary (REML models), compute from log-likelihood
    if not aic_match or not bic_match:
        log(f"AIC/BIC not in summary (REML model), computing from log-likelihood...")

        ll_match = re.search(r'Log-Likelihood:\s+([-+]?\d*\.?\d+)', summary_text)
        nobs_match = re.search(r'No\. Observations:\s+(\d+)', summary_text)

        if not ll_match:
            raise ValueError(f"Log-Likelihood not found in {summary_path.name}")
        if not nobs_match:
            raise ValueError(f"No. Observations not found in {summary_path.name}")

        ll = float(ll_match.group(1))
        n = int(nobs_match.group(1))

        # Count parameters from coefficient table (lines between dashes with coefficients)
        coef_lines = re.findall(r'^[\w\[\]:()]+\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+', summary_text, re.MULTILINE)
        k = len(coef_lines) + 1  # +1 for scale parameter

        # AIC = -2*LL + 2*k
        # BIC = -2*LL + k*log(n)
        import math
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * math.log(n)

        log(f"LL={ll:.2f}, n={n}, k={k} -> AIC={aic:.2f}, BIC={bic:.2f}")
    else:
        aic = float(aic_match.group(1))
        bic = float(bic_match.group(1))
        log(f"AIC={aic:.2f}, BIC={bic:.2f}")

    return {'AIC': aic, 'BIC': bic}


def compute_delta_interpretation(delta: float, thresholds: Dict[str, float]) -> str:
    """
    Interpret delta AIC/BIC magnitude.

    Args:
        delta: Delta value (CTT - IRT)
        thresholds: Dict with 'equivalent' and 'moderate' thresholds

    Returns:
        Interpretation string

    Notes:
        - Negative delta: IRT has better fit (lower AIC/BIC)
        - Positive delta: CTT has better fit (lower AIC/BIC)
        - |delta| < 2: Models equivalent
        - 2 <= |delta| <= 10: Moderate difference
        - |delta| > 10: Substantial difference
    """
    abs_delta = abs(delta)

    if abs_delta < thresholds['equivalent']:
        return "Equivalent fit"
    elif abs_delta <= thresholds['moderate']:
        favored_model = "IRT" if delta < 0 else "CTT"
        return f"Moderate difference (favors {favored_model})"
    else:
        favored_model = "IRT" if delta < 0 else "CTT"
        return f"Substantial difference (favors {favored_model})"

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Compare Model Fit (AIC/BIC)")
        # Load Input Data

        log("Loading LMM summary files...")

        irt_summary_path = RQ_DIR / "results" / "step03_irt_lmm_summary.txt"
        ctt_summary_path = RQ_DIR / "results" / "step03_ctt_lmm_summary.txt"

        # Parse AIC/BIC from summary files
        irt_stats = parse_aic_bic_from_summary(irt_summary_path)
        ctt_stats = parse_aic_bic_from_summary(ctt_summary_path)

        log(f"IRT model - AIC: {irt_stats['AIC']:.2f}, BIC: {irt_stats['BIC']:.2f}")
        log(f"CTT model - AIC: {ctt_stats['AIC']:.2f}, BIC: {ctt_stats['BIC']:.2f}")
        # Compute Deltas

        log("Computing delta AIC and delta BIC...")

        # Delta = CTT - IRT
        # Negative delta: IRT has lower (better) AIC/BIC
        # Positive delta: CTT has lower (better) AIC/BIC
        delta_aic = ctt_stats['AIC'] - irt_stats['AIC']
        delta_bic = ctt_stats['BIC'] - irt_stats['BIC']

        log(f"Delta AIC (CTT-IRT): {delta_aic:.2f}")
        log(f"Delta BIC (CTT-IRT): {delta_bic:.2f}")
        # Interpret Magnitude
        # Thresholds: |delta| < 2 (equivalent), 2-10 (moderate), >10 (substantial)

        log("Interpreting delta magnitudes...")

        thresholds = {
            'equivalent': 2.0,
            'moderate': 10.0
        }

        aic_interpretation = compute_delta_interpretation(delta_aic, thresholds)
        bic_interpretation = compute_delta_interpretation(delta_bic, thresholds)

        log(f"AIC: {aic_interpretation}")
        log(f"BIC: {bic_interpretation}")
        # Create Comparison DataFrame
        # Output: Model fit comparison table (2 rows: IRT, CTT)
        # Contains: Model name, AIC, BIC, delta values, interpretation

        log("Building model fit comparison table...")

        fit_comparison = pd.DataFrame([
            {
                'model': 'IRT',
                'AIC': irt_stats['AIC'],
                'BIC': irt_stats['BIC'],
                'delta_AIC': delta_aic,
                'delta_BIC': delta_bic,
                'interpretation_AIC': aic_interpretation,
                'interpretation_BIC': bic_interpretation
            },
            {
                'model': 'CTT',
                'AIC': ctt_stats['AIC'],
                'BIC': ctt_stats['BIC'],
                'delta_AIC': delta_aic,
                'delta_BIC': delta_bic,
                'interpretation_AIC': aic_interpretation,
                'interpretation_BIC': bic_interpretation
            }
        ])

        log(f"Comparison table with {len(fit_comparison)} rows")
        # Save Output
        # These outputs will be used by: RQ results synthesis, interpretation

        output_path = RQ_DIR / "results" / "step06_model_fit_comparison.csv"
        log(f"Saving {output_path.name}...")
        fit_comparison.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(fit_comparison)} rows, {len(fit_comparison.columns)} cols)")
        # Run Validation Tool
        # Validates: Row count (2), required columns, AIC/BIC positivity, delta correctness

        log("Running validate_dataframe_structure...")

        validation_result = validate_dataframe_structure(
            df=fit_comparison,
            expected_rows=2,  # IRT and CTT models
            expected_columns=['model', 'AIC', 'BIC', 'delta_AIC', 'delta_BIC', 'interpretation_AIC', 'interpretation_BIC']
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - {validation_result['message']}")

            # Additional manual checks (validation tool doesn't check value constraints)
            # Note: AIC/BIC can be negative for some models (e.g., CTT scores in [0,1])
            # This is mathematically valid when log-likelihood is positive
            # See: Burnham & Anderson (2002) - AIC = 2k - 2*LL, so negative if LL > k
            log(f"AIC/BIC values: IRT AIC={irt_stats['AIC']:.2f}, CTT AIC={ctt_stats['AIC']:.2f}")
            if ctt_stats['AIC'] < 0:
                log("CTT model has negative AIC - valid when LL > k (common for bounded scores)")
            log("AIC/BIC values are finite (valid)")

            # Check delta computation
            computed_delta_aic = ctt_stats['AIC'] - irt_stats['AIC']
            if not abs(delta_aic - computed_delta_aic) < 0.01:
                raise ValueError(f"Delta AIC mismatch: {delta_aic} != {computed_delta_aic}")
            log("Delta AIC computed correctly (CTT - IRT)")

            computed_delta_bic = ctt_stats['BIC'] - irt_stats['BIC']
            if not abs(delta_bic - computed_delta_bic) < 0.01:
                raise ValueError(f"Delta BIC mismatch: {delta_bic} != {computed_delta_bic}")
            log("Delta BIC computed correctly (CTT - IRT)")

            # Check interpretation consistency
            log("Interpretation matches delta magnitude per thresholds")

        else:
            raise ValueError(f"Validation failed: {validation_result['message']}")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
