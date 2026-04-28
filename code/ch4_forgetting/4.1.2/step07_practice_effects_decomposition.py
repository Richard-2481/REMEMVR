#!/usr/bin/env python3
"""
Step 07: Practice Effects Decomposition for RQ 5.1.2

PURPOSE:
Decompose T1→T2 trajectory into:
1. Practice gains (retrieval practice strengthens traces)
2. Genuine forgetting (memory decay despite practice)

CRITICAL INSIGHT:
Repeated testing creates CONFOUND:
- T1→T2: First retest may show IMPROVEMENT (practice effect)
- T2→T4: Subsequent tests show genuine DECLINE (forgetting > practice)

If we treat T1→T4 as single forgetting curve, we may:
- Underestimate early forgetting rate (practice masks true decline)
- Misinterpret deceleration (could be practice saturation, not consolidation)

METHODOLOGY:
Dual-phase model:
- Phase 1 (T1→T2): Practice gain + forgetting
- Phase 2 (T2→T4): Forgetting only (practice saturated)

Test: Does separating phases change interpretation of two-phase forgetting?

EXPECTED INPUTS:
  - data/step01_time_transformed.csv (theta, TSVR, UID, test)

EXPECTED OUTPUTS:
  - data/step07_practice_decomp_summary.txt (practice vs forgetting estimates)
  - data/step07_practice_effect_by_phase.csv (T1→T2 vs T2→T4 trajectories)
  - logs/step07_practice_effects_decomposition.log

VALIDATION:
  - Phase 1 slope significantly different from Phase 2 (practice effect exists)
  - If Phase 1 slope > 0: Practice DOMINATES (improvement despite forgetting)
  - If Phase 1 slope < 0 but |Phase1| < |Phase2|: Practice PARTIALLY masks forgetting
  - If Phase 1 ≈ Phase 2: No practice effect (pure forgetting)

INTERPRETATION:
Results will inform RQ 5.1.3 age analysis:
- If practice effects large: Must decompose before testing Age × Time
- If practice effects negligible: Original analysis sufficient
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_practice_effects_decomposition.log"

# Logging

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Initialize log
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("")

# Main Analysis

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("STEP 07: PRACTICE EFFECTS DECOMPOSITION")
        log("=" * 80)
        log(f"Date: {pd.Timestamp.now()}")
        log("")
        # Load Data and Define Phases

        log("[STEP 1] Loading data and defining practice vs forgetting phases...")

        time_data = pd.read_csv(RQ_DIR / "data" / "step01_time_transformed.csv", encoding='utf-8')
        log(f"  Loaded: {len(time_data)} observations, {time_data['UID'].nunique()} participants")

        # Define phases based on test number
        # Phase 1 (Practice): T1→T2 (tests 1 and 2)
        # Phase 2 (Forgetting): T2→T4 (tests 2, 3, and 4)

        # Create Phase variable
        time_data['Phase'] = time_data['test'].map({
            1: 'Practice',
            2: 'Practice',  # T2 appears in both phases (transition point)
            3: 'Forgetting',
            4: 'Forgetting'
        })

        # Create phase-specific time variable (reset to 0 at phase start)
        time_data['Time_within_phase'] = 0.0

        for uid in time_data['UID'].unique():
            uid_mask = time_data['UID'] == uid

            # Practice phase: Time from T1 (baseline = 0)
            practice_mask = uid_mask & (time_data['Phase'] == 'Practice')
            if practice_mask.sum() > 0:
                t1_time = time_data.loc[uid_mask & (time_data['test'] == 1), 'Time'].values[0]
                time_data.loc[practice_mask, 'Time_within_phase'] = time_data.loc[practice_mask, 'Time'] - t1_time

            # Forgetting phase: Time from T2 (reset to 0 at T2)
            forgetting_mask = uid_mask & (time_data['Phase'] == 'Forgetting')
            if forgetting_mask.sum() > 0:
                t2_time = time_data.loc[uid_mask & (time_data['test'] == 2), 'Time'].values[0]
                time_data.loc[forgetting_mask, 'Time_within_phase'] = time_data.loc[forgetting_mask, 'Time'] - t2_time

        log(f"  Practice phase: {len(time_data[time_data['Phase']=='Practice'])} observations (T1→T2)")
        log(f"  Forgetting phase: {len(time_data[time_data['Phase']=='Forgetting'])} observations (T2→T4)")
        log("")
        # Fit Dual-Phase Model

        log("[STEP 2] Fitting dual-phase model: theta ~ Time_within_phase * Phase...")
        log("  This tests if Practice phase slope differs from Forgetting phase slope")
        log("")

        # Dual-phase LMM with interaction
        formula = "theta ~ Time_within_phase * Phase"

        practice_model = smf.mixedlm(
            formula,
            data=time_data,
            groups=time_data['UID'],
            re_formula="~1"  # Intercept-only (matching previous models)
        ).fit(method='lbfgs', reml=False)

        log(f"  Converged: {practice_model.converged}")
        log(f"  AIC: {practice_model.aic:.2f}")
        log("")
        # Extract Phase-Specific Slopes

        log("[STEP 3] Extracting practice vs forgetting slopes...")

        fe = practice_model.fe_params
        bse = practice_model.bse
        pvalues = practice_model.pvalues

        # Practice phase slope (reference category)
        practice_slope = fe['Time_within_phase']
        practice_se = bse['Time_within_phase']
        practice_p = pvalues['Time_within_phase']

        log(f"  Practice phase (T1→T2) slope:")
        log(f"    β = {practice_slope:.6f} ± {practice_se:.6f}")
        log(f"    p = {practice_p:.6f}")

        # Forgetting phase slope (Practice + interaction)
        interaction_term = "Time_within_phase:Phase[T.Practice]"
        if interaction_term in fe.index:
            forgetting_slope = practice_slope + fe[interaction_term]
            # SE for sum (approximate, assumes no covariance)
            forgetting_se = np.sqrt(practice_se**2 + bse[interaction_term]**2)
            interaction_p = pvalues[interaction_term]
        else:
            log("  Interaction term not found - phases may not differ")
            forgetting_slope = practice_slope
            forgetting_se = practice_se
            interaction_p = 1.0

        log(f"  Forgetting phase (T2→T4) slope:")
        log(f"    β = {forgetting_slope:.6f} ± {forgetting_se:.6f}")
        log(f"  Interaction p-value: {interaction_p:.6f}")
        log("")
        # Test for Practice Effect

        log("[STEP 4] Testing for practice effect...")

        # Compute effect size (difference in slopes)
        slope_difference = practice_slope - forgetting_slope
        slope_diff_se = np.sqrt(practice_se**2 + forgetting_se**2)
        slope_diff_z = slope_difference / slope_diff_se
        slope_diff_p = 2 * (1 - stats.norm.cdf(np.abs(slope_diff_z)))

        log(f"  Slope difference (Practice - Forgetting):")
        log(f"    Δβ = {slope_difference:.6f} ± {slope_diff_se:.6f}")
        log(f"    z = {slope_diff_z:.3f}")
        log(f"    p = {slope_diff_p:.6f}")
        log("")

        # Interpret practice effect
        bonferroni_alpha = 0.0033

        if slope_diff_p < bonferroni_alpha:
            log(f"  SIGNIFICANT practice effect detected (p < {bonferroni_alpha})")

            if practice_slope > 0:
                log("  INTERPRETATION: Practice DOMINATES - memory IMPROVES T1→T2 despite forgetting")
                log("                  (Retrieval practice > decay)")
            elif abs(practice_slope) < abs(forgetting_slope):
                log("  INTERPRETATION: Practice PARTIALLY masks forgetting")
                log("                  (T1→T2 decline less steep than T2→T4)")
            else:
                log("  INTERPRETATION: Practice AMPLIFIES forgetting")
                log("                  (Unexpected pattern - investigate further)")
        else:
            log(f"  NO significant practice effect (p >= {bonferroni_alpha})")
            log("  INTERPRETATION: Forgetting rate similar across phases")
            log("                  (Practice effects negligible or saturated by T1)")
        log("")
        # Compare to Original Two-Phase Model

        log("[STEP 5] Comparing practice decomposition to original piecewise model...")

        # Original piecewise model used 48h inflection (Segment: Early vs Late)
        # Practice decomposition uses T1→T2 vs T2→T4 (aligned with test sessions)

        log("  Original piecewise model:")
        log("    - Inflection at 48 hours (arbitrary time-based cutoff)")
        log("    - Early/Late segments may mix practice and forgetting")
        log("")
        log("  Practice decomposition model:")
        log("    - Inflection at T2 (first retest, ~24h)")
        log("    - Phase 1 (T1→T2): Practice + forgetting confounded")
        log("    - Phase 2 (T2→T4): Pure forgetting (practice saturated)")
        log("")

        if slope_diff_p < bonferroni_alpha:
            log("  CONCLUSION: Practice effects significant")
            log("              Original piecewise model may conflate practice with consolidation")
            log("              Two-phase pattern could reflect:")
            log("                1. Consolidation (biological stabilization)")
            log("                2. Practice saturation (retrieval strengthening)")
            log("                3. Both processes operating simultaneously")
        else:
            log("  CONCLUSION: Practice effects negligible")
            log("              Original piecewise model interpretation valid")
            log("              Two-phase pattern likely reflects genuine consolidation")
        log("")
        # Save Outputs

        log("[STEP 6] Saving practice decomposition results...")

        # Save phase-specific estimates
        phase_estimates = pd.DataFrame({
            'phase': ['Practice (T1→T2)', 'Forgetting (T2→T4)', 'Difference'],
            'slope': [practice_slope, forgetting_slope, slope_difference],
            'se': [practice_se, forgetting_se, slope_diff_se],
            'p_value': [practice_p, interaction_p, slope_diff_p]
        })

        phase_estimates.to_csv(RQ_DIR / "data" / "step07_practice_effect_by_phase.csv",
                                index=False, encoding='utf-8')
        log("  Saved: step07_practice_effect_by_phase.csv")

        # Save summary text
        summary_path = RQ_DIR / "data" / "step07_practice_decomp_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PRACTICE EFFECTS DECOMPOSITION SUMMARY - RQ 5.1.2\n")
            f.write("=" * 80 + "\n\n")
            f.write("RESEARCH QUESTION:\n")
            f.write("  Does T1→T2 trajectory reflect practice gains or genuine forgetting?\n\n")
            f.write("METHODOLOGY:\n")
            f.write("  Dual-phase model: theta ~ Time_within_phase * Phase + (1 | UID)\n")
            f.write("  Phase 1 (Practice): T1→T2 (first retest)\n")
            f.write("  Phase 2 (Forgetting): T2→T4 (subsequent tests)\n\n")
            f.write("-" * 80 + "\n")
            f.write("RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Practice phase slope (T1→T2):   β = {practice_slope:.6f} ± {practice_se:.6f}, p = {practice_p:.6f}\n")
            f.write(f"Forgetting phase slope (T2→T4): β = {forgetting_slope:.6f} ± {forgetting_se:.6f}\n")
            f.write(f"Slope difference:                Δβ = {slope_difference:.6f}, p = {slope_diff_p:.6f}\n\n")

            if slope_diff_p < bonferroni_alpha:
                f.write(f"CONCLUSION: Significant practice effect detected (p < {bonferroni_alpha})\n")
                if practice_slope > 0:
                    f.write("  Practice DOMINATES: Memory improves T1→T2 despite forgetting\n")
                else:
                    f.write("  Practice PARTIALLY masks forgetting: T1→T2 decline less steep than T2→T4\n")
            else:
                f.write(f"CONCLUSION: No significant practice effect (p >= {bonferroni_alpha})\n")
                f.write("  Forgetting rate similar across phases\n")

            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("IMPLICATIONS FOR RQ 5.1.3 (AGE EFFECTS):\n")
            f.write("-" * 80 + "\n")
            if slope_diff_p < bonferroni_alpha:
                f.write("  CRITICAL: Age × Time analysis must account for practice effects\n")
                f.write("  Younger adults may benefit MORE from practice (processing speed advantage)\n")
                f.write("  This could create artifact of 'older adults better forgetting rate'\n")
                f.write("  Recommendation: Test Age × Phase interaction before interpreting Age × Time\n")
            else:
                f.write("  Practice effects negligible - original Age × Time analysis sufficient\n")

        log(f"  Saved: {summary_path.name}")
        log("")
        # Summary

        log("=" * 80)
        log("Practice effects decomposition complete!")
        log("=" * 80)
        log("")
        log("KEY FINDINGS:")
        log(f"  1. Practice slope: β = {practice_slope:.6f} (p = {practice_p:.6f})")
        log(f"  2. Forgetting slope: β = {forgetting_slope:.6f}")
        log(f"  3. Difference: Δβ = {slope_difference:.6f} (p = {slope_diff_p:.6f})")
        log("")

        if slope_diff_p < bonferroni_alpha:
            log("NEXT STEPS:")
            log("  1. Apply this methodology to RQ 5.1.3 (test Age × Phase interaction)")
            log("  2. Update RQ 5.1.2 summary.md with practice effect findings")
            log("  3. Revise two-phase interpretation (consolidation vs practice saturation)")
        else:
            log("CONCLUSION:")
            log("  Practice effects negligible - original two-phase interpretation stands")
            log("  No need to revise Age × Time analysis in RQ 5.1.3")

    except Exception as e:
        log("")
        log("=" * 80)
        log("Practice decomposition failed!")
        log("=" * 80)
        log(f"Error: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
