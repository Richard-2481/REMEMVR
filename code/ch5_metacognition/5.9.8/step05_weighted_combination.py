#!/usr/bin/env python3
"""Compute Inverse-Variance Weighted Mean Effect Size: Combine schema congruence effect sizes from accuracy and confidence using inverse-variance"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.8
LOG_FILE = RQ_DIR / "logs" / "step05_weighted_combination.log"
INPUT_FILE = RQ_DIR / "data" / "step04_schema_effects_extracted.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step05_weighted_combination.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Compute Inverse-Variance Weighted Mean Effect Size")
        # Load Extracted Effects
        log("Loading extracted effect sizes...")
        df = pd.read_csv(INPUT_FILE)
        log(f"{len(df)} effect sizes")

        if len(df) != 2:
            raise ValueError(f"Expected 2 rows, got {len(df)}")

        # Extract values
        d_accuracy = df.loc[df['measure'] == 'Accuracy', 'effect_size_d'].values[0]
        d_confidence = df.loc[df['measure'] == 'Confidence', 'effect_size_d'].values[0]
        SE_accuracy = df.loc[df['measure'] == 'Accuracy', 'SE_d'].values[0]
        SE_confidence = df.loc[df['measure'] == 'Confidence', 'SE_d'].values[0]

        log(f"Accuracy: d = {d_accuracy:.3f}, SE = {SE_accuracy:.3f}")
        log(f"Confidence: d = {d_confidence:.3f}, SE = {SE_confidence:.3f}")
        # Compute Inverse-Variance Weights
        log("Computing inverse-variance weights...")

        w_accuracy_raw = 1 / SE_accuracy**2
        w_confidence_raw = 1 / SE_confidence**2

        log(f"Raw weights: Accuracy = {w_accuracy_raw:.3f}, Confidence = {w_confidence_raw:.3f}")

        # Normalize weights to sum to 1
        w_total = w_accuracy_raw + w_confidence_raw
        w_accuracy = w_accuracy_raw / w_total
        w_confidence = w_confidence_raw / w_total

        log(f"Normalized weights: Accuracy = {w_accuracy:.3f}, Confidence = {w_confidence:.3f}")
        log(f"Weight sum = {w_accuracy + w_confidence:.6f} (should be 1.0)")
        # Compute Pooled Effect Size
        log("Computing pooled effect size...")

        d_pooled = w_accuracy * d_accuracy + w_confidence * d_confidence

        log(f"Pooled effect size: d = {d_pooled:.3f}")

        # Pooled SE
        SE_pooled = 1 / np.sqrt(w_accuracy_raw + w_confidence_raw)

        log(f"Pooled SE: {SE_pooled:.3f}")

        # 95% CI
        z_critical = 1.96
        CI_lower = d_pooled - z_critical * SE_pooled
        CI_upper = d_pooled + z_critical * SE_pooled

        log(f"95% CI: [{CI_lower:.3f}, {CI_upper:.3f}]")
        # Interpret Results
        log("Interpreting pooled effect size...")

        abs_d = abs(d_pooled)

        if abs_d < 0.20:
            interpretation = "Negligible effect (|d| < 0.20), convergent NULL confirmed"
            log("Negligible effect, convergent NULL confirmed")
        elif abs_d < 0.50:
            interpretation = "Small effect (0.20 <= |d| < 0.50), convergent NULL NOT confirmed"
            log("Small but non-negligible effect, convergent NULL NOT confirmed")
        else:
            interpretation = "Medium/large effect (|d| >= 0.50), convergent NULL NOT confirmed"
            log("Medium or large effect, convergent NULL NOT confirmed")

        # Limitation note (K=2 insufficient for meta-analysis)
        limitation_note = ("K=2 effect sizes insufficient for formal meta-analysis. " +
                          "No heterogeneity assessment performed. " +
                          "Simple inverse-variance weighting only.")

        log("Limitation: K=2 insufficient for formal meta-analysis")
        # Create Output DataFrame
        result = pd.DataFrame([{
            'd_pooled': d_pooled,
            'SE_pooled': SE_pooled,
            'CI_lower': CI_lower,
            'CI_upper': CI_upper,
            'w_accuracy': w_accuracy,
            'w_confidence': w_confidence,
            'd_accuracy': d_accuracy,
            'd_confidence': d_confidence,
            'interpretation': interpretation,
            'limitation_note': limitation_note
        }])
        # Validate Results
        log("Checking weighted combination...")

        errors = []

        # Check row count
        if len(result) != 1:
            errors.append(f"Expected 1 row, got {len(result)}")
        else:
            log("Row count: 1")

        # Check SE positive
        if SE_pooled <= 0:
            errors.append(f"Non-positive SE_pooled: {SE_pooled}")
        else:
            log("SE_pooled > 0")

        # Check CI ordering
        if not (CI_lower < d_pooled < CI_upper):
            # Allow edge case where d_pooled is at CI boundary
            if not (CI_lower <= d_pooled <= CI_upper):
                errors.append(f"CI bounds violated: {CI_lower:.3f} <= {d_pooled:.3f} <= {CI_upper:.3f}")
        else:
            log("CI ordering: CI_lower < d_pooled < CI_upper")

        # Check weights sum to 1
        weight_sum = w_accuracy + w_confidence
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            errors.append(f"Normalized weights don't sum to 1.0: {weight_sum:.9f}")
        else:
            log("Weights sum to 1.0")

        # Check weights in [0, 1]
        if w_accuracy < 0 or w_accuracy > 1 or w_confidence < 0 or w_confidence > 1:
            errors.append(f"Weights out of range [0, 1]: w_acc={w_accuracy:.3f}, w_conf={w_confidence:.3f}")
        else:
            log("Weights in [0, 1]")

        # Check CI width approximately 4*SE
        ci_width = CI_upper - CI_lower
        expected_width = 2 * z_critical * SE_pooled  # 2 * 1.96 * SE
        if not np.isclose(ci_width, expected_width, rtol=0.01):
            errors.append(f"CI width mismatch: {ci_width:.3f} vs expected {expected_width:.3f}")
        else:
            log("CI width approximately 4*SE_pooled")

        # Report interpretation
        log(f"Pooled effect: d = {d_pooled:.3f} [{CI_lower:.3f}, {CI_upper:.3f}]")
        log(f"{interpretation}")

        if abs_d < 0.20:
            log("Effect size negligible (|d| < 0.20)")
            log("Convergent NULL confirmed")
        else:
            log("Effect size non-negligible (|d| >= 0.20)")
            log("Convergent NULL NOT confirmed")
        # Save Output
        if errors:
            log("FAIL - Errors detected:")
            for error in errors:
                log(f"  - {error}")
            raise ValueError(f"Validation failed with {len(errors)} error(s)")

        log("PASS - All checks passed")
        log("Limitation acknowledged: K=2 insufficient for formal meta-analysis")

        result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"Output: {OUTPUT_FILE}")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
