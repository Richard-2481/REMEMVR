#!/usr/bin/env python3
"""
Practice Effects Decomposition for RQ 5.1.3

PURPOSE:
Decompose T1→T2 practice gains from T2→T4 genuine forgetting to address:
1. Wrong-direction age interactions (older adults showing less decline)
2. Practice confound: Repeated testing may benefit younger adults more
3. Test age-invariance assumption: Do all ages benefit equally from practice?

METHODOLOGY:
1. Separate T1→T2 trajectory (practice phase) from T2→T4 (forgetting phase)
2. Fit dual-phase model: Practice gain (T1→T2) + Forgetting (T2→T4)
3. Test Age × Phase interaction: Does practice benefit vary with age?
4. Compare to original model: Does practice decomposition resolve wrong-direction?

THEORETICAL BACKGROUND:
- Practice effects: ~0.60 SD gain on retest (BMC Neuroscience 2010)
- Age-dependent practice: Younger adults may benefit more (processing speed)
- Confound with forgetting: If younger adults improve T1→T2 while older stable,
  this creates artifact of "older adults better forgetting rate"

INPUTS:
  - data/step01_lmm_input_prepared.csv (theta, Age_c, Time, UID, TEST)

OUTPUTS:
  - data/step03_practice_decomp_input.csv (with Phase variable)
  - data/step03_practice_phase_estimates.csv (T1→T2 practice gains by age)
  - data/step03_forgetting_phase_estimates.csv (T2→T4 forgetting by age)
  - data/step03_age_practice_interaction.csv (Age × Phase test)
  - logs/step03_practice_decomposition.log
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

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_practice_decomposition.log"

# =============================================================================
# Logging
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Initialize log
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("")

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("="*70)
        log("RQ 5.1.3 - PRACTICE EFFECTS DECOMPOSITION")
        log("="*70)
        log(f"Date: {pd.Timestamp.now()}")
        log("")

        # =====================================================================
        # STEP 1: Load Data and Create Phase Variable
        # =====================================================================

        log("[STEP 1] Loading data and defining practice vs forgetting phases...")
        input_path = RQ_DIR / "data" / "step01_lmm_input_prepared.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input data missing: {input_path}")

        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"  Loaded: {len(df)} observations, {df['UID'].nunique()} participants")
        log("")

        # Define phases based on TEST (FIX: TEST is int64, not string)
        # Practice phase: T1→T2 (first exposure to retest)
        # Forgetting phase: T2→T4 (post-practice forgetting)
        df['Phase'] = df['TEST'].map({
            1: 'Practice',  # T1 baseline
            2: 'Practice',  # T2 immediate retest (practice gain expected)
            3: 'Forgetting',  # T3 delayed (forgetting from T2)
            4: 'Forgetting'   # T4 final (continued forgetting)
        })

        # Create phase-specific time variables
        # Practice phase: Time relative to T1 (0 hours at T1)
        # Forgetting phase: Time relative to T2 (0 hours at T2, reset)
        df['Time_within_phase'] = 0.0

        for uid in df['UID'].unique():
            mask = df['UID'] == uid

            # Practice phase: Time from T1
            practice_mask = mask & (df['Phase'] == 'Practice')
            if practice_mask.sum() > 0:
                t1_time = df.loc[mask & (df['TEST'] == 1), 'Time'].values[0]
                df.loc[practice_mask, 'Time_within_phase'] = df.loc[practice_mask, 'Time'] - t1_time

            # Forgetting phase: Time from T2 (reset to 0 at T2)
            forgetting_mask = mask & (df['Phase'] == 'Forgetting')
            if forgetting_mask.sum() > 0:
                t2_time = df.loc[mask & (df['TEST'] == 2), 'Time'].values[0]
                df.loc[forgetting_mask, 'Time_within_phase'] = df.loc[forgetting_mask, 'Time'] - t2_time

        # Log transform of within-phase time
        df['Time_within_phase_log'] = np.log(df['Time_within_phase'] + 1)

        log("  Phase definitions:")
        log(f"    Practice: T1→T2 (N={len(df[df['Phase']=='Practice'])} obs)")
        log(f"    Forgetting: T3→T4 (N={len(df[df['Phase']=='Forgetting'])} obs)")
        log("")

        # =====================================================================
        # STEP 2: Fit Dual-Phase Model with Age Interactions
        # =====================================================================

        log("[STEP 2] Fitting dual-phase model: theta ~ Phase × Time × Age_c...")
        log("  Formula: theta ~ (Time_within_phase_log * Phase) * Age_c")
        log("  Random effects: ~Time_within_phase_log | UID")
        log("")

        # Dual-phase LMM
        formula = "theta ~ (Time_within_phase_log * Phase) * Age_c"
        md = smf.mixedlm(formula, data=df, groups=df['UID'],
                          re_formula="~Time_within_phase_log")
        mdf = md.fit(method='lbfgs', reml=False)

        log(f"  Model converged: {mdf.converged}")
        log(f"  AIC: {mdf.aic:.2f}")
        log(f"  N observations: {len(df)}")
        log("")

        # =====================================================================
        # STEP 3: Extract Key Effects
        # =====================================================================

        log("[STEP 3] Extracting practice vs forgetting age effects...")
        log("")

        # Key parameters of interest:
        # 1. Phase[T.Practice]:Age_c → Age effect on practice phase intercept
        # 2. Time_within_phase_log:Phase[T.Practice]:Age_c → Age × Practice slope
        # 3. Time_within_phase_log:Age_c → Age × Forgetting slope (reference)

        params = mdf.params
        pvalues = mdf.pvalues
        bse = mdf.bse

        effects_of_interest = [
            'Age_c',  # Age effect on baseline (T1)
            'Time_within_phase_log:Age_c',  # Age × forgetting slope (reference=Forgetting)
            'Phase[T.Practice]:Age_c',  # Age effect on practice phase
            'Time_within_phase_log:Phase[T.Practice]:Age_c'  # Age × practice slope
        ]

        log("  Key Age Effects:")
        log("  " + "-"*60)
        for effect in effects_of_interest:
            if effect in params.index:
                coef = params[effect]
                se = bse[effect]
                z = coef / se
                p = pvalues[effect]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                log(f"  {effect:50s}: β={coef:7.4f} ± {se:.4f}, p={p:.4f} {sig}")
            else:
                log(f"  {effect:50s}: NOT IN MODEL")
        log("")

        # =====================================================================
        # STEP 4: Test Age-Invariance of Practice Effects
        # =====================================================================

        log("[STEP 4] Testing age-invariance of practice effects...")

        # Null hypothesis: Practice effects are age-invariant
        # H0: Time_within_phase_log:Phase[T.Practice]:Age_c = 0
        # (no Age × Practice slope interaction)

        if 'Time_within_phase_log:Phase[T.Practice]:Age_c' in params.index:
            age_practice_coef = params['Time_within_phase_log:Phase[T.Practice]:Age_c']
            age_practice_se = bse['Time_within_phase_log:Phase[T.Practice]:Age_c']
            age_practice_z = age_practice_coef / age_practice_se
            age_practice_p = pvalues['Time_within_phase_log:Phase[T.Practice]:Age_c']

            log(f"  Age × Practice Slope Interaction:")
            log(f"    Coefficient: {age_practice_coef:.5f} ± {age_practice_se:.5f}")
            log(f"    Z-statistic: {age_practice_z:.3f}")
            log(f"    P-value: {age_practice_p:.4f}")
            log("")

            if age_practice_p < 0.05:
                log("  RESULT: Practice effects are AGE-DEPENDENT (p<0.05)")
                log("  Interpretation: Younger and older adults show different practice gains")
                if age_practice_coef > 0:
                    log("  Direction: Older adults benefit MORE from practice (unexpected)")
                else:
                    log("  Direction: Younger adults benefit MORE from practice (expected)")
            else:
                log("  RESULT: Practice effects are AGE-INVARIANT (p≥0.05)")
                log("  Interpretation: All ages benefit equally from repeated testing")
        else:
            log("  WARNING: Age × Practice interaction not in model")
        log("")

        # =====================================================================
        # STEP 5: Compare Practice vs Forgetting Age Effects
        # =====================================================================

        log("[STEP 5] Comparing practice phase vs forgetting phase age effects...")

        # Extract phase-specific slopes
        # Forgetting slope (reference): Time_within_phase_log:Age_c
        # Practice slope: Time_within_phase_log:Age_c + Time_within_phase_log:Phase[T.Practice]:Age_c

        if 'Time_within_phase_log:Age_c' in params.index:
            forgetting_age_slope = params['Time_within_phase_log:Age_c']
            forgetting_age_se = bse['Time_within_phase_log:Age_c']

            if 'Time_within_phase_log:Phase[T.Practice]:Age_c' in params.index:
                practice_age_slope = forgetting_age_slope + params['Time_within_phase_log:Phase[T.Practice]:Age_c']
                practice_age_se = np.sqrt(forgetting_age_se**2 +
                                          bse['Time_within_phase_log:Phase[T.Practice]:Age_c']**2)
            else:
                practice_age_slope = forgetting_age_slope
                practice_age_se = forgetting_age_se

            log("  Phase-Specific Age × Slope Effects:")
            log(f"    Practice phase (T1→T2):   β={practice_age_slope:7.4f} ± {practice_age_se:.4f}")
            log(f"    Forgetting phase (T2→T4): β={forgetting_age_slope:7.4f} ± {forgetting_age_se:.4f}")
            log("")

            # Interpretation
            if practice_age_slope > 0 and forgetting_age_slope < 0:
                log("  PATTERN: Practice confound detected!")
                log("    - Practice phase: Older adults IMPROVE more (positive slope)")
                log("    - Forgetting phase: Older adults FORGET faster (negative slope)")
                log("    - Original wrong-direction artifact explained by practice")
            elif abs(practice_age_slope) < abs(forgetting_age_slope):
                log("  PATTERN: Age effects stronger in forgetting than practice")
                log("    - Consistent with hippocampal aging hypothesis")
            else:
                log("  PATTERN: Age effects similar across phases")
        log("")

        # =====================================================================
        # STEP 6: Save Outputs
        # =====================================================================

        log("[STEP 6] Saving outputs...")

        # Save data with phase variable
        output_path = RQ_DIR / "data" / "step03_practice_decomp_input.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"  Saved: {output_path.name}")

        # Save practice phase estimates (aggregate by age tertile for visualization)
        practice_df = df[df['Phase'] == 'Practice'].copy()
        practice_df['Age_tertile'] = pd.qcut(practice_df['age'], q=3,
                                               labels=['Young', 'Middle', 'Older'])

        practice_summary = practice_df.groupby(['Age_tertile', 'TEST']).agg({
            'theta': ['mean', 'std', 'count'],
            'Time_within_phase': 'mean',
            'Age_c': 'mean'
        }).reset_index()
        practice_summary.columns = ['_'.join(col).strip('_') for col in practice_summary.columns.values]

        practice_path = RQ_DIR / "data" / "step03_practice_phase_estimates.csv"
        practice_summary.to_csv(practice_path, index=False, encoding='utf-8')
        log(f"  Saved: {practice_path.name}")

        # Save forgetting phase estimates
        forgetting_df = df[df['Phase'] == 'Forgetting'].copy()
        forgetting_df['Age_tertile'] = pd.qcut(forgetting_df['age'], q=3,
                                                 labels=['Young', 'Middle', 'Older'])

        forgetting_summary = forgetting_df.groupby(['Age_tertile', 'TEST']).agg({
            'theta': ['mean', 'std', 'count'],
            'Time_within_phase': 'mean',
            'Age_c': 'mean'
        }).reset_index()
        forgetting_summary.columns = ['_'.join(col).strip('_') for col in forgetting_summary.columns.values]

        forgetting_path = RQ_DIR / "data" / "step03_forgetting_phase_estimates.csv"
        forgetting_summary.to_csv(forgetting_path, index=False, encoding='utf-8')
        log(f"  Saved: {forgetting_path.name}")

        # Save age × phase interaction test
        interaction_data = {
            'effect': ['Age_c', 'Time_within_phase_log:Age_c',
                       'Phase[T.Practice]:Age_c',
                       'Time_within_phase_log:Phase[T.Practice]:Age_c'],
            'coef': [params.get(e, np.nan) for e in
                     ['Age_c', 'Time_within_phase_log:Age_c',
                      'Phase[T.Practice]:Age_c',
                      'Time_within_phase_log:Phase[T.Practice]:Age_c']],
            'se': [bse.get(e, np.nan) for e in
                   ['Age_c', 'Time_within_phase_log:Age_c',
                    'Phase[T.Practice]:Age_c',
                    'Time_within_phase_log:Phase[T.Practice]:Age_c']],
            'p': [pvalues.get(e, np.nan) for e in
                  ['Age_c', 'Time_within_phase_log:Age_c',
                   'Phase[T.Practice]:Age_c',
                   'Time_within_phase_log:Phase[T.Practice]:Age_c']]
        }
        interaction_df = pd.DataFrame(interaction_data)
        interaction_path = RQ_DIR / "data" / "step03_age_practice_interaction.csv"
        interaction_df.to_csv(interaction_path, index=False, encoding='utf-8')
        log(f"  Saved: {interaction_path.name}")

        log("")
        log("="*70)
        log("[SUCCESS] Practice effects decomposition complete!")
        log("="*70)
        log("Key findings:")
        if 'Time_within_phase_log:Phase[T.Practice]:Age_c' in params.index:
            if pvalues['Time_within_phase_log:Phase[T.Practice]:Age_c'] < 0.05:
                log("  - Practice effects are AGE-DEPENDENT")
                log("  - This explains wrong-direction age interactions in original model")
            else:
                log("  - Practice effects are AGE-INVARIANT")
                log("  - Wrong-direction findings NOT explained by practice confound")
        log("")
        log("Next: Examine step03_age_practice_interaction.csv for detailed results")

    except Exception as e:
        log("")
        log("="*70)
        log("[ERROR] Practice decomposition failed!")
        log("="*70)
        log(f"Error: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
