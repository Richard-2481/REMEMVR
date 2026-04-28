"""
RQ 6.3.4 - ICC by Domain Analysis (Steps 01-06)

PURPOSE:
Decompose variance in domain-specific confidence trajectories (What/Where/When) into trait-like
(intercept: baseline confidence) and state-like (residual: within-person fluctuation) components.
Tests whether ICC_slope differs by memory domain, and whether 5-level confidence data reveals
domain-specific trait variance that dichotomous accuracy data (Ch5 5.2.6) could not detect.

KEY QUESTION:
Is confidence decline more trait-like (individual difference) for some memory domains than others?

INPUT:
- results/ch6/6.3.1/data/step03_theta_confidence.csv (400 rows: wide format with theta by domain)
- results/ch6/6.3.1/data/step00_tsvr_mapping.csv (TSVR hours mapping)

OUTPUT:
- data/step01_variance_components_by_domain.csv (3 domains × 4 variance components)
- data/step02_variance_components.csv (with total_variance added)
- data/step03_icc_estimates.csv (3 domains × 3 ICC types)
- data/step04_random_effects.csv (300 rows: 100 participants × 3 domains)
- data/step05_domain_icc_comparison.csv (domain ranking) + step05_pairwise_icc_differences.csv
- data/step06_ch5_comparison.csv (confidence vs accuracy ICC comparison)

METHODOLOGY:
- Fit separate LMMs per domain with random intercept + slope on TSVR_hours
- Extract variance components from cov_re matrix per domain
- Compute ICCs following Hoffman & Stawski (2009)
- Compare ICC_slope across domains and to Ch5 5.2.6 accuracy ICC

Date: 2025-12-11
RQ: ch6/6.3.4
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

LOG_FILE = RQ_DIR / "logs" / "steps_01_to_06.log"
DATA_DIR = RQ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
(RQ_DIR / "logs").mkdir(exist_ok=True)

DOMAINS = ["What", "Where", "When"]

def log(msg):
    """Log message to file and stdout with flush"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def step01_fit_domain_lmms():
    """
    Step 01: Fit domain-stratified LMMs with random slopes

    Fit separate random-effects LMMs per domain (What, Where, When) to estimate
    variance components (var_intercept, var_slope, cov_int_slope, var_residual).
    """
    import statsmodels.formula.api as smf

    log("=" * 80)
    log("[STEP 01] Fit Domain-Stratified LMMs with Random Slopes")
    log("=" * 80)

    # Load theta data from RQ 6.3.1 (wide format: composite_ID, theta_What, theta_Where, theta_When)
    theta_path = PROJECT_ROOT / "results" / "ch6" / "6.3.1" / "data" / "step03_theta_confidence.csv"
    tsvr_path = PROJECT_ROOT / "results" / "ch6" / "6.3.1" / "data" / "step00_tsvr_mapping.csv"

    log(f"\nLoading data from RQ 6.3.1:")
    log(f"  Theta file: {theta_path}")
    log(f"  TSVR file: {tsvr_path}")

    theta_df = pd.read_csv(theta_path)
    tsvr_df = pd.read_csv(tsvr_path)

    log(f"  ✓ Theta data: {len(theta_df)} rows × {len(theta_df.columns)} columns")
    log(f"  ✓ TSVR data: {len(tsvr_df)} rows")

    # Merge theta with TSVR
    df = theta_df.merge(tsvr_df, on='composite_ID', how='inner')
    log(f"  ✓ Merged: {len(df)} rows")

    # Parse UID from composite_ID (e.g., "A010_T1" -> "A010")
    df['UID'] = df['composite_ID'].str.split('_').str[0]
    log(f"  ✓ N participants: {df['UID'].nunique()}")
    log(f"  ✓ TSVR range: [{df['TSVR_hours'].min():.2f}, {df['TSVR_hours'].max():.2f}] hours")

    # Store fitted models and variance components
    models = {}
    variance_components_list = []

    # Fit LMM per domain
    for domain in DOMAINS:
        log(f"\n[DOMAIN: {domain}]")

        # Get theta column name
        theta_col = f'theta_{domain}'

        # Check column exists
        if theta_col not in df.columns:
            log(f"  Column {theta_col} not found!")
            raise ValueError(f"Column {theta_col} not found in data")

        # Prepare domain-specific data (long format for this domain)
        domain_df = df[['UID', 'composite_ID', 'test', 'TSVR_hours', theta_col]].copy()
        domain_df = domain_df.rename(columns={theta_col: 'theta_confidence'})

        # Check for missing values
        n_missing = domain_df['theta_confidence'].isna().sum()
        if n_missing > 0:
            log(f"  {n_missing} missing theta values - dropping")
            domain_df = domain_df.dropna(subset=['theta_confidence'])

        log(f"  N observations: {len(domain_df)}")
        log(f"  Theta range: [{domain_df['theta_confidence'].min():.3f}, {domain_df['theta_confidence'].max():.3f}]")

        # Fit LMM with random intercept + slope on TSVR_hours
        formula = "theta_confidence ~ TSVR_hours"
        re_formula = "~TSVR_hours"

        log(f"  Fitting LMM: {formula} + ({re_formula} | UID)")

        model = smf.mixedlm(formula, domain_df, groups=domain_df["UID"], re_formula=re_formula)
        result = model.fit(reml=False)

        log(f"  ✓ Converged: {result.converged}")
        log(f"  ✓ AIC: {result.aic:.4f}")

        # Save model summary
        summary_path = DATA_DIR / f"step01_lmm_{domain.lower()}_model_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"RQ 6.3.4 - LMM Summary for {domain} Domain\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Random effects: ({re_formula} | UID)\n\n")
            f.write(str(result.summary()) + "\n")
        log(f"  ✓ Saved summary: {summary_path.name}")

        # Extract variance components
        cov_re = result.cov_re
        var_intercept = cov_re.iloc[0, 0]
        var_slope = cov_re.iloc[1, 1] if cov_re.shape[0] > 1 else 0.0
        cov_int_slope = cov_re.iloc[0, 1] if cov_re.shape[0] > 1 else 0.0
        var_residual = result.scale

        log(f"  var_intercept: {var_intercept:.6f}")
        log(f"  var_slope: {var_slope:.9f}")
        log(f"  cov_int_slope: {cov_int_slope:.9f}")
        log(f"  var_residual: {var_residual:.6f}")

        # Validate variance components
        if var_intercept < 0 or var_slope < 0 or var_residual < 0:
            log(f"  Negative variance component detected - boundary estimate")

        # Store model and variance components
        models[domain] = result
        variance_components_list.append({
            'domain': domain,
            'var_intercept': var_intercept,
            'var_slope': var_slope,
            'cov_int_slope': cov_int_slope,
            'var_residual': var_residual
        })

    # Save variance components
    var_df = pd.DataFrame(variance_components_list)
    output_path = DATA_DIR / "step01_variance_components_by_domain.csv"
    var_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    log(f"\n:")
    log(f"  ✓ All 3 domain LMMs fitted successfully")
    log(f"  ✓ Variance components extracted for all domains")

    return models, var_df, df


def step02_extract_variance_components(var_df):
    """
    Step 02: Create variance components table with total variance computed
    """
    log("\n" + "=" * 80)
    log("[STEP 02] Extract Variance Components Per Domain")
    log("=" * 80)

    # Compute total variance (excluding covariance from sum)
    var_df['total_variance'] = (var_df['var_intercept'] +
                                 var_df['var_slope'] +
                                 var_df['var_residual'])

    log(f"\n[TOTAL VARIANCE BY DOMAIN]:")
    for _, row in var_df.iterrows():
        log(f"  {row['domain']}: {row['total_variance']:.6f}")

    # Save
    output_path = DATA_DIR / "step02_variance_components.csv"
    var_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return var_df


def step03_compute_icc_estimates(var_df, df):
    """
    Step 03: Compute three ICC estimates per domain

    ICCs:
    - ICC_intercept: Proportion of variance attributable to stable baseline differences
    - ICC_slope_simple: Proportion of slope variance relative to slope + residual
    - ICC_slope_conditional: Slope variance at final timepoint (Day 6)
    """
    log("\n" + "=" * 80)
    log("[STEP 03] Compute ICC Per Domain")
    log("=" * 80)

    # Get time parameters
    max_time = df['TSVR_hours'].max()  # Day 6 ~144-150 hours
    log(f"\n:")
    log(f"  Max TSVR_hours (Day 6): {max_time:.2f}")

    icc_list = []

    for _, row in var_df.iterrows():
        domain = row['domain']
        var_int = row['var_intercept']
        var_slope = row['var_slope']
        cov_int_slope = row['cov_int_slope']
        var_res = row['var_residual']

        log(f"\n[DOMAIN: {domain}]")

        # ICC_intercept: var_intercept / (var_intercept + var_residual)
        ICC_intercept = var_int / (var_int + var_res) if (var_int + var_res) > 0 else 0

        # ICC_slope_simple: var_slope / (var_slope + var_residual)
        ICC_slope_simple = var_slope / (var_slope + var_res) if (var_slope + var_res) > 0 else 0

        # ICC_slope_conditional at Day 6
        # Total variance at Day 6 = var_int + var_slope*t^2 + 2*cov*t + var_res
        total_var_day6 = (var_int +
                         var_slope * max_time**2 +
                         2 * cov_int_slope * max_time +
                         var_res)
        ICC_slope_conditional = var_slope * max_time**2 / total_var_day6 if total_var_day6 > 0 else 0

        log(f"  ICC_intercept: {ICC_intercept:.4f}")
        log(f"  ICC_slope_simple: {ICC_slope_simple:.6f}")
        log(f"  ICC_slope_conditional: {ICC_slope_conditional:.6f}")

        # Interpret ICCs
        def interpret_icc(icc):
            if icc < 0.05:
                return "negligible"
            elif icc < 0.10:
                return "small"
            elif icc < 0.20:
                return "moderate"
            elif icc < 0.40:
                return "substantial"
            else:
                return "high"

        icc_list.append({
            'domain': domain,
            'ICC_intercept': ICC_intercept,
            'ICC_slope_simple': ICC_slope_simple,
            'ICC_slope_conditional': ICC_slope_conditional,
            'interpretation_intercept': interpret_icc(ICC_intercept),
            'interpretation_slope': interpret_icc(ICC_slope_simple)
        })

    icc_df = pd.DataFrame(icc_list)

    # Validate ICC bounds
    for col in ['ICC_intercept', 'ICC_slope_simple']:
        if (icc_df[col] < 0).any():
            log(f"Negative {col} detected - computation error")
            raise ValueError(f"Negative {col} detected")
        if (icc_df[col] > 1).any():
            log(f"{col} > 1.0 detected - boundary estimate")

    # Save
    output_path = DATA_DIR / "step03_icc_estimates.csv"
    icc_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return icc_df


def step04_extract_random_effects(models, df):
    """
    Step 04: Extract participant-specific random effects per domain

    Output: 300 rows (100 participants × 3 domains) with random_intercept and random_slope
    """
    log("\n" + "=" * 80)
    log("[STEP 04] Extract Random Effects Per Domain")
    log("=" * 80)

    re_list = []

    for domain in DOMAINS:
        log(f"\n[DOMAIN: {domain}]")

        result = models[domain]
        random_effects = result.random_effects

        log(f"  N groups: {len(random_effects)}")

        for uid, re_array in random_effects.items():
            re_dict = {
                'UID': str(uid),
                'domain': domain
            }

            if hasattr(re_array, 'iloc'):
                re_dict['random_intercept'] = re_array.iloc[0]
                re_dict['random_slope'] = re_array.iloc[1] if len(re_array) > 1 else 0.0
            else:
                re_dict['random_intercept'] = re_array[0]
                re_dict['random_slope'] = re_array[1] if len(re_array) > 1 else 0.0

            re_list.append(re_dict)

    re_df = pd.DataFrame(re_list)

    log(f"\n:")
    log(f"  ✓ Total rows: {len(re_df)} (expected 300)")
    log(f"  ✓ N unique UIDs: {re_df['UID'].nunique()}")
    log(f"  ✓ N unique domains: {re_df['domain'].nunique()}")

    # Check for complete factorial design
    expected_rows = 100 * 3  # 100 participants × 3 domains
    if len(re_df) != expected_rows:
        log(f"  Expected {expected_rows} rows, got {len(re_df)}")

    # Check for duplicates
    n_duplicates = re_df.duplicated(subset=['UID', 'domain']).sum()
    if n_duplicates > 0:
        log(f"  {n_duplicates} duplicate UID×domain combinations")
        raise ValueError("Duplicate UID×domain combinations detected")

    # Save
    output_path = DATA_DIR / "step04_random_effects.csv"
    re_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return re_df


def step05_compare_icc_across_domains(icc_df):
    """
    Step 05: Compare ICC_slope across domains and rank by trait-like variance
    """
    log("\n" + "=" * 80)
    log("[STEP 05] Compare ICC_slope Across Domains")
    log("=" * 80)

    # Rank domains by ICC_slope_simple
    icc_df_sorted = icc_df.sort_values('ICC_slope_simple', ascending=False).reset_index(drop=True)
    icc_df_sorted['rank'] = range(1, len(icc_df_sorted) + 1)

    # Interpret
    def interpret_trait(icc):
        if icc > 0.10:
            return "High trait variance"
        elif icc > 0.05:
            return "Moderate trait variance"
        else:
            return "Low trait variance"

    icc_df_sorted['trait_interpretation'] = icc_df_sorted['ICC_slope_simple'].apply(interpret_trait)

    log(f"\n[DOMAIN RANKING BY ICC_slope]:")
    for _, row in icc_df_sorted.iterrows():
        log(f"  Rank {row['rank']}: {row['domain']} - ICC_slope={row['ICC_slope_simple']:.6f} ({row['trait_interpretation']})")

    # Save comparison table
    comparison_df = icc_df_sorted[['domain', 'ICC_slope_simple', 'rank', 'trait_interpretation']]
    comparison_df = comparison_df.rename(columns={'trait_interpretation': 'interpretation'})
    output_path = DATA_DIR / "step05_domain_icc_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    # Compute pairwise differences
    log(f"\n[PAIRWISE DIFFERENCES]:")
    pairwise_list = []

    icc_by_domain = icc_df.set_index('domain')['ICC_slope_simple']

    comparisons = [
        ("What vs Where", "What", "Where"),
        ("What vs When", "What", "When"),
        ("Where vs When", "Where", "When")
    ]

    for comparison_name, d1, d2 in comparisons:
        delta = icc_by_domain[d1] - icc_by_domain[d2]

        if abs(delta) > 0.05:
            interp = "Meaningful difference"
        else:
            interp = "Negligible difference"

        log(f"  {comparison_name}: Δ = {delta:.6f} ({interp})")

        pairwise_list.append({
            'comparison': comparison_name,
            'delta_ICC': delta,
            'interpretation': interp
        })

    pairwise_df = pd.DataFrame(pairwise_list)
    output_path = DATA_DIR / "step05_pairwise_icc_differences.csv"
    pairwise_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return comparison_df, pairwise_df


def step06_compare_to_ch5(icc_df):
    """
    Step 06: Compare confidence ICC (this RQ) to accuracy ICC (Ch5 5.2.6)

    Tests measurement richness hypothesis: Does 5-level confidence data reveal
    domain-specific trait variance that dichotomous accuracy data missed?
    """
    log("\n" + "=" * 80)
    log("[STEP 06] Compare to Ch5 5.2.6 Accuracy ICC")
    log("=" * 80)

    # Try to load Ch5 5.2.6 ICC estimates
    ch5_path = PROJECT_ROOT / "results" / "ch5" / "5.2.6" / "data" / "step03_icc_estimates.csv"

    if ch5_path.exists():
        log(f"\nFound Ch5 5.2.6 ICC estimates: {ch5_path}")
        ch5_df = pd.read_csv(ch5_path)
        log(f"  ✓ Loaded {len(ch5_df)} rows")

        # Ch5 5.2.6 format: domain, icc_type, icc_value, interpretation, threshold_used
        # Pivot to get ICC_slope_simple per domain
        ch5_slope = ch5_df[ch5_df['icc_type'] == 'slope_simple'][['domain', 'icc_value']]
        ch5_slope = ch5_slope.rename(columns={'icc_value': 'ICC_slope_accuracy'})
        ch5_slope = ch5_slope.set_index('domain')

        log(f"\n[Ch5 5.2.6 ICC_slope_accuracy by domain]:")
        for domain in DOMAINS:
            if domain in ch5_slope.index:
                log(f"  {domain}: {ch5_slope.loc[domain, 'ICC_slope_accuracy']:.6f}")
            else:
                log(f"  {domain}: NOT FOUND")

        # Merge with confidence ICC
        comparison_list = []
        for _, row in icc_df.iterrows():
            domain = row['domain']
            icc_conf = row['ICC_slope_simple']

            if domain in ch5_slope.index:
                icc_acc = ch5_slope.loc[domain, 'ICC_slope_accuracy']
            else:
                icc_acc = np.nan

            delta = icc_conf - icc_acc if not np.isnan(icc_acc) else np.nan

            # Interpret
            if np.isnan(delta):
                interp = "comparison pending (accuracy data missing)"
            elif delta > 0.05:
                interp = "Confidence reveals MORE trait variance"
            elif delta < -0.05:
                interp = "Accuracy reveals MORE trait variance (unexpected)"
            else:
                interp = "Equivalent trait variance"

            comparison_list.append({
                'domain': domain,
                'ICC_slope_confidence': icc_conf,
                'ICC_slope_accuracy': icc_acc,
                'delta_ICC': delta,
                'interpretation': interp
            })

        comparison_df = pd.DataFrame(comparison_list)

        log(f"\n[COMPARISON RESULTS]:")
        for _, row in comparison_df.iterrows():
            log(f"  {row['domain']}:")
            log(f"    Confidence ICC: {row['ICC_slope_confidence']:.6f}")
            log(f"    Accuracy ICC:   {row['ICC_slope_accuracy']:.6f}")
            log(f"    Delta:          {row['delta_ICC']:.6f}")
            log(f"    → {row['interpretation']}")

    else:
        log(f"\nCh5 5.2.6 file not found: {ch5_path}")
        log("  Creating placeholder comparison with 'pending' status")

        comparison_list = []
        for _, row in icc_df.iterrows():
            comparison_list.append({
                'domain': row['domain'],
                'ICC_slope_confidence': row['ICC_slope_simple'],
                'ICC_slope_accuracy': np.nan,
                'delta_ICC': np.nan,
                'interpretation': 'comparison pending (Ch5 5.2.6 not executed)'
            })

        comparison_df = pd.DataFrame(comparison_list)

    # Save
    output_path = DATA_DIR / "step06_ch5_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return comparison_df


if __name__ == "__main__":
    try:
        log("=" * 80)
        log(f"RQ 6.3.4 - ICC by Domain Analysis")
        log(f"Started: {datetime.now().isoformat()}")
        log("=" * 80)

        # Step 1: Fit domain-stratified LMMs
        models, var_df, df = step01_fit_domain_lmms()

        # Step 2: Extract variance components with total
        var_df = step02_extract_variance_components(var_df)

        # Step 3: Compute ICC estimates per domain
        icc_df = step03_compute_icc_estimates(var_df, df)

        # Step 4: Extract random effects per domain
        re_df = step04_extract_random_effects(models, df)

        # Step 5: Compare ICC across domains
        comparison_df, pairwise_df = step05_compare_icc_across_domains(icc_df)

        # Step 6: Compare to Ch5 5.2.6
        ch5_comparison = step06_compare_to_ch5(icc_df)

        log("\n" + "=" * 80)
        log("RQ 6.3.4 Complete")
        log("=" * 80)

        log(f"\n[SUMMARY - ICC by Domain]:")
        for _, row in icc_df.iterrows():
            log(f"  {row['domain']}: ICC_int={row['ICC_intercept']:.4f}, ICC_slope={row['ICC_slope_simple']:.6f} ({row['interpretation_slope']})")

        log(f"\n[KEY FINDING]:")
        # Determine if any domain shows substantial ICC_slope
        max_slope = icc_df['ICC_slope_simple'].max()
        if max_slope > 0.10:
            log(f"  ✓ TRAIT VARIANCE DETECTED: Max ICC_slope = {max_slope:.4f}")
            log(f"  → Some domains show trait-like confidence decline patterns")
        else:
            log(f"  ✓ LOW TRAIT VARIANCE: All ICC_slope < 0.10")
            log(f"  → Parallels Ch5 5.2.6 pattern (state-like, not trait-like)")

        log(f"\n  Completed: {datetime.now().isoformat()}")

    except Exception as e:
        log(f"\n{e}")
        import traceback
        log(traceback.format_exc())
        raise
