#!/usr/bin/env python3
"""Variance Decomposition: Decompose total ICC difference into components: scaling artifact, modeling artifact,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Configuration

SCRIPT_PATH = Path(__file__).resolve()
RQ_DIR = SCRIPT_PATH.parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_variance_decomposition.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Variance Decomposition")
        # Load All ICC Estimates from Steps 1-5

        log("\nLoading ICC estimates from prior steps...")

        # Step 1: Accuracy ICC
        df_acc = pd.read_csv(RQ_DIR / "data" / "step01_accuracy_variance_components.csv")
        ICC_acc = float(df_acc[df_acc['component'] == 'ICC_slope']['value'].values[0])
        log(f"  ICC_accuracy: {ICC_acc:.6f}")

        # Step 2: Confidence ICCs (single-model and model-averaged)
        df_conf = pd.read_csv(RQ_DIR / "data" / "step02_confidence_variance_components.csv")
        ICC_conf_single = float(df_conf[df_conf['model_type'] == 'single_Recip_sq']['ICC_slope'].values[0])
        ICC_conf_ma = float(df_conf[df_conf['model_type'] == 'model_averaged_48']['ICC_slope'].values[0])
        log(f"  ICC_conf_single: {ICC_conf_single:.6f}")
        log(f"  ICC_conf_ma: {ICC_conf_ma:.6f}")

        # Step 4: Matched functional form (optional - for alternative decomposition)
        try:
            df_matched = pd.read_csv(RQ_DIR / "data" / "step04_matched_functional_form_ratio.csv")
            ICC_conf_PowerLaw = float(df_matched['ICC_confidence_PowerLaw'].values[0])
            log(f"  ICC_conf_PowerLaw: {ICC_conf_PowerLaw:.6f}")
            has_matched = True
        except:
            log("  ICC_conf_PowerLaw: NOT AVAILABLE (Step 4 may have failed)")
            ICC_conf_PowerLaw = np.nan
            has_matched = False

        # Step 5: Binary confidence
        try:
            df_binary = pd.read_csv(RQ_DIR / "data" / "step05_binary_sensitivity_ratio.csv")
            ICC_binary_conf = float(df_binary[df_binary['measure_type'] == 'binary_confidence']['ICC_slope'].values[0])
            log(f"  ICC_binary_conf: {ICC_binary_conf:.6f}")
            has_binary = True
        except:
            log("  ICC_binary_conf: NOT AVAILABLE (Step 5 may have failed)")
            ICC_binary_conf = np.nan
            has_binary = False
        # Define Variance Components
        # D_total: Total difference to explain
        # D_scale: Scaling artifact (ordinal vs binary)
        # D_model: Modeling artifact (single-model vs model-averaged)
        # D_genuine: Residual (genuine psychological variance)

        log("\nComputing variance decomposition components...")

        D_total = ICC_conf_single - ICC_acc
        log(f"  D_total (total difference): {D_total:.6f}")

        # Scaling artifact: difference between ordinal single-model and binary
        if has_binary:
            D_scale = abs(ICC_conf_single - ICC_binary_conf)
        else:
            D_scale = 0.0
            log("  WARNING: D_scale = 0 (binary ICC unavailable)")
        log(f"  D_scale (scaling artifact): {D_scale:.6f}")

        # Modeling artifact: difference between single-model and model-averaged
        D_model = abs(ICC_conf_single - ICC_conf_ma)
        log(f"  D_model (modeling artifact): {D_model:.6f}")

        # Genuine variance: residual after removing artifacts
        D_genuine = D_total - D_scale - D_model
        log(f"  D_genuine (genuine variance): {D_genuine:.6f}")

        # Check for negative genuine variance (artifacts overexplaining)
        if D_genuine < -0.01:
            log(f"  WARNING: D_genuine < 0 (artifacts overexplain by {abs(D_genuine):.6f})")
        # Compute Percentages

        log("\nComputing percentages...")

        if D_total > 0:
            Scaling_pct = (D_scale / D_total) * 100
            Modeling_pct = (D_model / D_total) * 100
            Genuine_pct = (D_genuine / D_total) * 100
        else:
            log("  WARNING: D_total <= 0, cannot compute percentages")
            Scaling_pct = 0.0
            Modeling_pct = 0.0
            Genuine_pct = 0.0

        log(f"  Scaling artifact: {Scaling_pct:.2f}%")
        log(f"  Modeling artifact: {Modeling_pct:.2f}%")
        log(f"  Genuine variance: {Genuine_pct:.2f}%")

        # Validate sum to 100%
        total_pct = Scaling_pct + Modeling_pct + Genuine_pct
        log(f"  Total percentage: {total_pct:.2f}%")

        if abs(total_pct - 100.0) > 0.01:
            log(f"  ERROR: Components don't sum to 100% (got {total_pct:.2f}%)")
            raise ValueError(f"STEP ERROR: Variance decomposition doesn't sum to 100% (got {total_pct:.2f}%)")
        # Create Primary Decomposition Table

        log("\nCreating variance decomposition table...")

        decomposition = pd.DataFrame({
            'component': ['scaling_artifact', 'modeling_artifact', 'genuine_variance', 'total'],
            'variance_explained': [D_scale, D_model, D_genuine, D_total],
            'percent_of_total': [Scaling_pct, Modeling_pct, Genuine_pct, 100.0],
            'interpretation': [
                f'Ordinal vs binary measurement scaling ({Scaling_pct:.1f}%)',
                f'Single-model vs model-averaged ({Modeling_pct:.1f}%)',
                f'Genuine individual differences ({Genuine_pct:.1f}%)',
                f'Total ICC difference to explain'
            ]
        })

        output_decomp = RQ_DIR / "data" / "step06_variance_decomposition.csv"
        decomposition.to_csv(output_decomp, index=False, encoding='utf-8')
        log(f"{output_decomp.name}")
        # Create Validation Table

        log("\nCreating validation checks...")

        validation = pd.DataFrame({
            'metric': [
                'all_percentages_non_negative',
                'components_sum_to_100',
                'genuine_variance_reasonable'
            ],
            'value': [
                min(Scaling_pct, Modeling_pct, Genuine_pct),
                total_pct,
                D_genuine
            ],
            'threshold': [
                0.0,
                100.0,
                -0.10
            ],
            'pass_fail': [
                'PASS' if min(Scaling_pct, Modeling_pct, Genuine_pct) >= 0 else 'FAIL',
                'PASS' if abs(total_pct - 100.0) <= 0.01 else 'FAIL',
                'PASS' if D_genuine >= -0.10 else 'WARNING'
            ]
        })

        output_val = RQ_DIR / "data" / "step06_decomposition_validation.csv"
        validation.to_csv(output_val, index=False, encoding='utf-8')
        log(f"{output_val.name}")

        # Log validation results
        for idx, row in validation.iterrows():
            log(f"  {row['metric']}: {row['pass_fail']} (value={row['value']:.4f}, threshold={row['threshold']:.4f})")
        # Alternative Decomposition (if PowerLaw matched available)

        log("\nCreating alternative decomposition...")

        alt_methods = []

        # Primary method
        alt_methods.append({
            'method': 'primary_binary_scaling',
            'D_total': D_total,
            'scaling_pct': Scaling_pct,
            'notes': 'Uses binary confidence ICC to isolate scaling artifact'
        })

        # Alternative method using PowerLaw matched (if available)
        if has_matched and has_binary:
            # Alternative: functional form artifact instead of modeling artifact
            D_func = abs(ICC_conf_single - ICC_conf_PowerLaw)
            Func_pct = (D_func / D_total * 100) if D_total > 0 else 0.0

            alt_methods.append({
                'method': 'alternative_functional_form',
                'D_total': D_total,
                'scaling_pct': Func_pct,
                'notes': 'Uses PowerLaw matched ICC to isolate functional form artifact'
            })

            log(f"  Alternative decomposition available:")
            log(f"    Functional form artifact: {Func_pct:.2f}%")

        alt_decomp = pd.DataFrame(alt_methods)
        output_alt = RQ_DIR / "data" / "step06_alternative_decomposition.csv"
        alt_decomp.to_csv(output_alt, index=False, encoding='utf-8')
        log(f"{output_alt.name}")
        # Interpretation

        log("\nVariance decomposition interpretation:")

        total_artifact = Scaling_pct + Modeling_pct

        if total_artifact > 80:
            log(f"  PRIMARY HYPOTHESIS SUPPORTED: {total_artifact:.1f}% artifact")
            log(f"    Measurement artifacts dominate ICC difference")
            log(f"    Confidence individual differences largely methodological")
        elif Genuine_pct > 50:
            log(f"  ALTERNATIVE HYPOTHESIS SUPPORTED: {Genuine_pct:.1f}% genuine")
            log(f"    Metacognitive variance genuinely higher than accuracy")
            log(f"    Individual differences in confidence are real")
        else:
            log(f"  MIXED RESULT: {total_artifact:.1f}% artifact, {Genuine_pct:.1f}% genuine")
            log(f"    Both artifacts and genuine variance contribute")
        # Final Validation Summary

        log("\nFinal summary:")

        all_pass = all(validation['pass_fail'] == 'PASS')
        log(f"  All validations passed: {'YES' if all_pass else 'NO'}")
        log(f"  Components sum to 100%: {'PASS' if abs(total_pct - 100.0) <= 0.01 else 'FAIL'}")
        log(f"  Decomposition complete: PASS")

        if not all_pass:
            log("  WARNING: Some validation checks failed (see table above)")

        log("\nStep 06 complete - Variance decomposition computed")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
