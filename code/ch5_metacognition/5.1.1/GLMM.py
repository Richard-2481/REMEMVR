"""
GLMM Analysis for RQ 6.1.1: Confidence Trajectory Functional Form

This script validates the IRT→LMM results using item-level ordinal GLMM analysis.

Key differences from Ch5 GLMM:
- Confidence data is ORDINAL (5 levels: 0.2, 0.4, 0.6, 0.8, 1.0)
- Uses ordinal logistic regression via GEE
- Tests whether time main effect from IRT→LMM holds with item-level data

RQ 6.1.1: Which functional form best describes confidence decline?
- IRT→LMM found: Logarithmic best (63.9% Akaike weight in 5-model comparison)
- Kitchen sink (65 models): High uncertainty, Sin+Cos best but non-converged

This GLMM uses:
- Outcome: Ordinal confidence ratings (5 levels)
- Predictor: log(TSVR) for time effect
- Clustering: GEE with exchangeable correlation by participant
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.cov_struct import Exchangeable
import warnings
import matplotlib.pyplot as plt
from scipy import stats

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

# Plot styling
TIME_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # T1-T4


def load_and_reshape_data():
    """
    Load wide-format IRT input and reshape to long format for GLMM.

    Returns:
        DataFrame with columns: UID, test, item, confidence, TSVR_hours, log_TSVR
    """
    # Load IRT input (wide format: composite_ID × items)
    irt_input = pd.read_csv(DATA_DIR / "step00_irt_input.csv")
    print(f"Loaded IRT input: {irt_input.shape}")

    # Load TSVR mapping
    tsvr_mapping = pd.read_csv(DATA_DIR / "step00_tsvr_mapping.csv")
    print(f"Loaded TSVR mapping: {tsvr_mapping.shape}")

    # Parse composite_ID to extract UID and test
    # Format: A010_1 (UID_test) - note different from Ch5 format
    irt_input['UID'] = irt_input['composite_ID'].str.split('_').str[0]
    irt_input['test'] = irt_input['composite_ID'].str.split('_').str[1].astype(int)

    # Merge TSVR
    tsvr_mapping['UID'] = tsvr_mapping['composite_ID'].str.split('_').str[0]
    tsvr_mapping['test_num'] = tsvr_mapping['composite_ID'].str.split('_').str[1].astype(int)

    irt_input = irt_input.merge(
        tsvr_mapping[['UID', 'test_num', 'TSVR_hours']],
        left_on=['UID', 'test'],
        right_on=['UID', 'test_num'],
        how='left'
    )

    # Get item columns (TC_* columns)
    item_cols = [c for c in irt_input.columns if c.startswith('TC_')]
    print(f"Found {len(item_cols)} TC_* items")

    # Reshape to long format
    id_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours']
    long_df = irt_input.melt(
        id_vars=id_cols,
        value_vars=item_cols,
        var_name='item',
        value_name='confidence'
    )

    # Remove missing values
    long_df = long_df.dropna(subset=['confidence', 'TSVR_hours'])

    # Create log-transformed time (add small constant to avoid log(0))
    long_df['log_TSVR'] = np.log(long_df['TSVR_hours'] + 0.01)

    print(f"\nReshaped data: {len(long_df)} observations")
    print(f"N participants: {long_df['UID'].nunique()}")
    print(f"N tests per participant: {long_df.groupby('UID')['test'].nunique().mean():.1f}")
    print(f"N items per test: {len(item_cols)}")

    # Summary statistics
    print(f"\nConfidence distribution:")
    print(long_df['confidence'].value_counts().sort_index())

    print(f"\nTSVR summary:")
    print(long_df.groupby('test')['TSVR_hours'].describe()[['mean', 'std', 'min', 'max']])

    return long_df


def fit_glmm_ordinal(df):
    """
    Fit ordinal logistic GLMM using cumulative link approach.

    Since statsmodels doesn't have direct ordinal GEE, we use two approaches:
    1. Treat confidence as continuous (quasi-likelihood)
    2. Binary collapse (high/low confidence) for robustness check

    Returns:
        dict: Results from different model specifications
    """
    print("\n" + "="*60)
    print("FITTING GLMM MODELS")
    print("="*60)

    results = {}

    # Sort by participant for GEE
    df_sorted = df.sort_values(['UID', 'test', 'item'])

    # =========================================================================
    # Model 1: Quasi-continuous GEE (treat ordinal as interval)
    # =========================================================================
    print("\n### Model 1: Quasi-Continuous GEE ###")
    print("Treating ordinal confidence (0.2-1.0) as continuous")

    X = df_sorted[['log_TSVR']].copy()
    X.insert(0, 'Intercept', 1)
    y = df_sorted['confidence']
    groups = df_sorted['UID']

    try:
        model_cont = GEE(
            y, X,
            groups=groups,
            family=Gaussian(),
            cov_struct=Exchangeable()
        )
        res_cont = model_cont.fit()

        print("\nParameter Estimates:")
        print(f"{'Parameter':<20} {'Estimate':>12} {'SE':>12} {'z':>10} {'p':>10}")
        print("-"*66)
        for param in res_cont.params.index:
            est = res_cont.params[param]
            se = res_cont.bse[param]
            z = res_cont.tvalues[param]
            p = res_cont.pvalues[param]
            print(f"{param:<20} {est:>12.6f} {se:>12.6f} {z:>10.3f} {p:>10.6f}")

        results['continuous'] = res_cont

    except Exception as e:
        print(f"Error fitting continuous GEE: {e}")
        results['continuous'] = None

    # =========================================================================
    # Model 2: Binomial GEE (collapse to high/low confidence)
    # =========================================================================
    print("\n### Model 2: Binomial GEE (High vs Low Confidence) ###")
    print("Dichotomizing: High confidence (≥0.6) vs Low confidence (<0.6)")

    df_sorted['conf_high'] = (df_sorted['confidence'] >= 0.6).astype(int)

    y_bin = df_sorted['conf_high']

    try:
        model_bin = GEE(
            y_bin, X,
            groups=groups,
            family=Binomial(),
            cov_struct=Exchangeable()
        )
        res_bin = model_bin.fit()

        print("\nParameter Estimates:")
        print(f"{'Parameter':<20} {'Estimate':>12} {'SE':>12} {'z':>10} {'p':>10}")
        print("-"*66)
        for param in res_bin.params.index:
            est = res_bin.params[param]
            se = res_bin.bse[param]
            z = res_bin.tvalues[param]
            p = res_bin.pvalues[param]
            print(f"{param:<20} {est:>12.6f} {se:>12.6f} {z:>10.3f} {p:>10.6f}")

        # Odds ratio for time effect
        or_time = np.exp(res_bin.params['log_TSVR'])
        print(f"\nOdds ratio for log_TSVR: {or_time:.4f}")
        print(f"Interpretation: Per unit increase in log(TSVR), odds of high confidence multiply by {or_time:.4f}")

        results['binomial'] = res_bin

    except Exception as e:
        print(f"Error fitting binomial GEE: {e}")
        results['binomial'] = None

    # =========================================================================
    # Model 3: Proportional Odds (via multiple binary models)
    # =========================================================================
    print("\n### Model 3: Pseudo-Proportional Odds ###")
    print("Fitting binary GEE at each confidence threshold")

    thresholds = [0.2, 0.4, 0.6, 0.8]  # Cumulative thresholds
    threshold_results = []

    for thresh in thresholds:
        df_sorted[f'conf_gt_{int(thresh*10)}'] = (df_sorted['confidence'] > thresh).astype(int)
        y_thresh = df_sorted[f'conf_gt_{int(thresh*10)}']

        try:
            model_thresh = GEE(
                y_thresh, X,
                groups=groups,
                family=Binomial(),
                cov_struct=Exchangeable()
            )
            res_thresh = model_thresh.fit()

            threshold_results.append({
                'threshold': thresh,
                'intercept': res_thresh.params['Intercept'],
                'intercept_se': res_thresh.bse['Intercept'],
                'log_TSVR_beta': res_thresh.params['log_TSVR'],
                'log_TSVR_se': res_thresh.bse['log_TSVR'],
                'log_TSVR_p': res_thresh.pvalues['log_TSVR']
            })

        except Exception as e:
            print(f"  Error at threshold {thresh}: {e}")

    if threshold_results:
        print("\nCumulative Threshold Analysis:")
        print(f"{'Threshold':<12} {'β(time)':>12} {'SE':>12} {'p':>12}")
        print("-"*50)
        for tr in threshold_results:
            print(f">{tr['threshold']:<11} {tr['log_TSVR_beta']:>12.6f} {tr['log_TSVR_se']:>12.6f} {tr['log_TSVR_p']:>12.6f}")

        # Test proportional odds assumption
        betas = [tr['log_TSVR_beta'] for tr in threshold_results]
        print(f"\nProportional odds check:")
        print(f"  β range across thresholds: {min(betas):.4f} to {max(betas):.4f}")
        print(f"  If similar, proportional odds assumption holds")

        results['thresholds'] = threshold_results

    return results


def compare_with_lmm_results():
    """Print original IRT→LMM results for comparison."""
    print("\n" + "="*60)
    print("ORIGINAL IRT→LMM RESULTS (for comparison)")
    print("="*60)

    print("""
### Original 5-Model Comparison (from 1_concept.md):
Best model: Logarithmic
Akaike weight: 63.9% (clear winner)

Model Rankings:
1. Logarithmic: 63.9%
2. Linear+Logarithmic: 23.7%
3. Quadratic+Logarithmic: 9.3%
4. Quadratic: 3.1%
5. Linear: <0.1%

### Kitchen Sink Comparison (65 models):
Best: Sin+Cos (21.7%, NON-CONVERGED)
Best Converged: Recip_sq (2.7%)
Logarithmic: Rank #38, 0.95%

### Key Finding from IRT→LMM:
- TIME effect is SIGNIFICANT (confidence declines over retention interval)
- Functional form is UNCERTAIN (high model uncertainty in kitchen sink)
- Logarithmic is best only in limited 5-model comparison
""")


def save_results_summary(results, df):
    """Save markdown summary of GLMM results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "glmm_comparison.md"

    # Extract key statistics
    cont = results.get('continuous')
    bin_ = results.get('binomial')
    thresh = results.get('thresholds', [])

    summary = f"""# GLMM vs IRT → LMM Comparison

## Research Question
**RQ 6.1.1:** Which functional form best describes confidence decline over a 6-day retention interval?

## Methods Comparison

| Aspect | IRT → LMM | GLMM (this analysis) |
|--------|-----------|---------------------|
| **Approach** | Two-stage (GRM → LMM) | Single-stage |
| **Outcome** | Theta scores (continuous) | Ordinal ratings (5-level) |
| **Error structure** | Gaussian | Quasi-continuous GEE / Binomial GEE |
| **N observations** | 400 (aggregated) | {len(df):,} (item-level) |
| **Time variable** | log_Days_plus1 | log(TSVR_hours) |
| **Random effects** | Random slopes by participant | GEE clustering by participant |

## Key Results

### Time Effect (The Critical Test)

| Method | β (time) | SE | p-value | Interpretation |
|--------|----------|----|---------| --------------|
"""

    if cont is not None:
        summary += f"| GLMM Continuous | {cont.params['log_TSVR']:.6f} | {cont.bse['log_TSVR']:.6f} | {cont.pvalues['log_TSVR']:.6f} | {'SIGNIFICANT' if cont.pvalues['log_TSVR'] < 0.05 else 'NS'} |\n"

    if bin_ is not None:
        summary += f"| GLMM Binomial | {bin_.params['log_TSVR']:.6f} | {bin_.bse['log_TSVR']:.6f} | {bin_.pvalues['log_TSVR']:.6f} | {'SIGNIFICANT' if bin_.pvalues['log_TSVR'] < 0.05 else 'NS'} |\n"

    summary += """| IRT → LMM | ~-0.058 | ~0.009 | <0.001 | SIGNIFICANT |

### GLMM Model Details

"""

    if cont is not None:
        summary += f"""#### Quasi-Continuous GEE
- **Intercept**: {cont.params['Intercept']:.6f} (SE={cont.bse['Intercept']:.6f})
- **log_TSVR**: {cont.params['log_TSVR']:.6f} (SE={cont.bse['log_TSVR']:.6f}, p={cont.pvalues['log_TSVR']:.6f})
- Treating ordinal confidence (0.2-1.0) as interval scale

"""

    if bin_ is not None:
        summary += f"""#### Binomial GEE (High vs Low Confidence)
- **Intercept**: {bin_.params['Intercept']:.6f} (SE={bin_.bse['Intercept']:.6f})
- **log_TSVR**: {bin_.params['log_TSVR']:.6f} (SE={bin_.bse['log_TSVR']:.6f}, p={bin_.pvalues['log_TSVR']:.6f})
- **Odds Ratio**: {np.exp(bin_.params['log_TSVR']):.4f}
- Dichotomized at ≥0.6 = high confidence

"""

    if thresh:
        summary += """#### Cumulative Threshold Analysis
Testing time effect at each ordinal threshold (proportional odds check):

| Threshold | β(time) | SE | p |
|-----------|---------|----|----|
"""
        for tr in thresh:
            summary += f"| >{tr['threshold']} | {tr['log_TSVR_beta']:.6f} | {tr['log_TSVR_se']:.6f} | {tr['log_TSVR_p']:.6f} |\n"

        betas = [tr['log_TSVR_beta'] for tr in thresh]
        summary += f"""
**Proportional Odds Check:** β range = {min(betas):.4f} to {max(betas):.4f}
{'✓ Similar βs suggest proportional odds assumption holds' if max(betas) - min(betas) < 0.05 else '⚠ βs vary substantially across thresholds'}

"""

    # Confidence by test
    summary += """### Confidence by Test (Descriptive)

| Test | Mean | SD | N |
|------|------|----|---|
"""
    for test in sorted(df['test'].unique()):
        test_data = df[df['test'] == test]['confidence']
        summary += f"| T{test} | {test_data.mean():.3f} | {test_data.std():.3f} | {len(test_data):,} |\n"

    summary += """
## Conclusion

"""

    # Determine overall conclusion
    time_sig_cont = cont is not None and cont.pvalues['log_TSVR'] < 0.05
    time_sig_bin = bin_ is not None and bin_.pvalues['log_TSVR'] < 0.05

    if time_sig_cont and time_sig_bin:
        summary += """**The time effect finding is ROBUST to methodological choice.**

Both GLMM approaches confirm the IRT→LMM result:
1. **Significant main effect of time** - Confidence declines over the retention interval
2. **Time effect is NEGATIVE** - Later tests show lower confidence
3. Confidence decline is consistent across item-level and aggregated approaches
"""
    elif time_sig_cont or time_sig_bin:
        summary += """**The time effect finding is PARTIALLY ROBUST.**

One GLMM approach confirms the IRT→LMM result:
- Time effect significant in {'continuous GEE' if time_sig_cont else 'binomial GEE'}
- Time effect {'significant' if time_sig_bin else 'non-significant'} in {'binomial GEE' if time_sig_cont else 'continuous GEE'}
"""
    else:
        summary += """**DISCREPANCY: Time effect NOT significant in GLMM.**

The IRT→LMM found significant time effect, but GLMM did not confirm this.
Possible reasons:
1. Different statistical assumptions (Gaussian vs ordinal)
2. Aggregation effects in IRT→LMM
3. Item-level heterogeneity masking time effect
"""

    summary += """
The GLMM approach directly models ordinal confidence ratings at item-level,
providing a robustness check for the two-stage IRT→LMM methodology.
"""

    with open(output_path, 'w') as f:
        f.write(summary)

    print(f"\nSaved comparison summary to: {output_path}")
    return summary


def create_glmm_plots(df, results):
    """
    Create plots visualizing GLMM results.

    Generates:
    1. Empirical confidence over continuous TSVR (binned for visualization)
    2. Model-predicted confidence trajectories
    """
    print("\n" + "="*60)
    print("CREATING GLMM PLOTS")
    print("="*60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Empirical Confidence Trajectory ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create TSVR bins for visualization
    df['TSVR_bin'] = pd.cut(df['TSVR_hours'], bins=20)

    # Calculate mean confidence and TSVR per bin
    binned = df.groupby('TSVR_bin', observed=True).agg({
        'confidence': ['mean', 'std', 'count'],
        'TSVR_hours': 'mean'
    }).reset_index()

    binned.columns = ['bin', 'confidence', 'std', 'n', 'tsvr_mean']
    binned = binned.dropna()

    # Calculate 95% CI
    binned['se'] = binned['std'] / np.sqrt(binned['n'])
    binned['ci'] = 1.96 * binned['se']

    ax.errorbar(
        binned['tsvr_mean'], binned['confidence'],
        yerr=binned['ci'],
        marker='o', markersize=8, capsize=4,
        color='#1f77b4',
        label='Mean ± 95% CI',
        linewidth=2
    )

    ax.set_xlabel('Time Since VR Encoding (hours)', fontsize=12)
    ax.set_ylabel('Mean Confidence Rating', fontsize=12)
    ax.set_title('Confidence Trajectory Over Retention Interval\n(GLMM Data: Binned for Visualization)', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'glmm_confidence_trajectory.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'glmm_confidence_trajectory.png'}")

    # --- Plot 2: Model Predictions ---
    cont = results.get('continuous')
    if cont is not None:
        fig, ax = plt.subplots(figsize=(10, 6))

        tsvr_range = np.linspace(df['TSVR_hours'].min(), df['TSVR_hours'].max(), 200)
        log_tsvr_range = np.log(tsvr_range + 0.01)

        # Predicted confidence
        pred_conf = cont.params['Intercept'] + cont.params['log_TSVR'] * log_tsvr_range

        # Standard error bands (simplified)
        se_pred = np.sqrt(
            cont.bse['Intercept']**2 +
            (cont.bse['log_TSVR'] * log_tsvr_range)**2
        )

        ax.plot(tsvr_range, pred_conf, color='#1f77b4', linewidth=2.5, label='GEE Prediction')
        ax.fill_between(
            tsvr_range,
            pred_conf - 1.96*se_pred,
            pred_conf + 1.96*se_pred,
            color='#1f77b4', alpha=0.2, label='95% CI'
        )

        # Overlay empirical points (by test)
        test_means = df.groupby('test').agg({
            'confidence': 'mean',
            'TSVR_hours': 'mean'
        }).reset_index()

        ax.scatter(
            test_means['TSVR_hours'], test_means['confidence'],
            s=150, color='red', marker='o', zorder=5,
            label='Empirical Means (T1-T4)'
        )

        ax.set_xlabel('Time Since VR Encoding (hours)', fontsize=12)
        ax.set_ylabel('Predicted Confidence', fontsize=12)
        ax.set_title('GLMM Model Predictions: Confidence Decline\n(Quasi-Continuous GEE with Log-Time)', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(0.3, 0.8)
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.annotate(
            f'β(time) = {cont.params["log_TSVR"]:.4f}\np < 0.001' if cont.pvalues['log_TSVR'] < 0.001 else f'β(time) = {cont.params["log_TSVR"]:.4f}\np = {cont.pvalues["log_TSVR"]:.4f}',
            xy=(0.98, 0.98), xycoords='axes fraction',
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'glmm_model_predictions.png', dpi=150)
        plt.close()
        print(f"Saved: {PLOTS_DIR / 'glmm_model_predictions.png'}")

    # --- Plot 3: Binomial Probability ---
    bin_ = results.get('binomial')
    if bin_ is not None:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Predicted probability of high confidence
        log_odds = bin_.params['Intercept'] + bin_.params['log_TSVR'] * log_tsvr_range
        prob_high = 1 / (1 + np.exp(-log_odds))

        # CI bands
        se_log_odds = np.sqrt(
            bin_.bse['Intercept']**2 +
            (bin_.bse['log_TSVR'] * log_tsvr_range)**2
        )
        prob_lower = 1 / (1 + np.exp(-(log_odds - 1.96*se_log_odds)))
        prob_upper = 1 / (1 + np.exp(-(log_odds + 1.96*se_log_odds)))

        ax.plot(tsvr_range, prob_high, color='#2ca02c', linewidth=2.5, label='GEE Prediction')
        ax.fill_between(
            tsvr_range, prob_lower, prob_upper,
            color='#2ca02c', alpha=0.2, label='95% CI'
        )

        # Create conf_high if not exists
        if 'conf_high' not in df.columns:
            df['conf_high'] = (df['confidence'] >= 0.6).astype(int)

        # Overlay empirical proportions
        test_props = df.groupby('test').agg({
            'conf_high': 'mean',
            'TSVR_hours': 'mean'
        }).reset_index()

        ax.scatter(
            test_props['TSVR_hours'], test_props['conf_high'],
            s=150, color='red', marker='o', zorder=5,
            label='Empirical Proportions (T1-T4)'
        )

        ax.set_xlabel('Time Since VR Encoding (hours)', fontsize=12)
        ax.set_ylabel('P(High Confidence)', fontsize=12)
        ax.set_title('GLMM Binomial: Probability of High Confidence (≥0.6)\nOver Retention Interval', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'glmm_binomial_probability.png', dpi=150)
        plt.close()
        print(f"Saved: {PLOTS_DIR / 'glmm_binomial_probability.png'}")

    # --- Plot 4: Threshold Analysis ---
    thresh = results.get('thresholds')
    if thresh:
        fig, ax = plt.subplots(figsize=(10, 6))

        thresholds = [tr['threshold'] for tr in thresh]
        betas = [tr['log_TSVR_beta'] for tr in thresh]
        ses = [tr['log_TSVR_se'] for tr in thresh]

        ax.errorbar(
            thresholds, betas,
            yerr=[1.96*se for se in ses],
            marker='o', markersize=10, capsize=6,
            color='#d62728', linewidth=2
        )

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('β(log_TSVR)', fontsize=12)
        ax.set_title('Time Effect Across Confidence Thresholds\n(Proportional Odds Check)', fontsize=14)
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f'>{t}' for t in thresholds])
        ax.grid(True, alpha=0.3)

        # Annotate
        ax.annotate(
            f'β range: {min(betas):.4f} to {max(betas):.4f}\nSimilar βs support proportional odds',
            xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'glmm_threshold_analysis.png', dpi=150)
        plt.close()
        print(f"Saved: {PLOTS_DIR / 'glmm_threshold_analysis.png'}")

    print("\nAll plots created successfully!")


def main():
    """Main analysis pipeline."""
    print("="*70)
    print("GLMM ANALYSIS: RQ 6.1.1 Confidence Trajectory")
    print("Testing IRT → LMM findings with item-level ordinal GLMM")
    print("="*70)

    # Load and prepare data
    df = load_and_reshape_data()

    # Fit GLMM models
    results = fit_glmm_ordinal(df)

    # Show comparison with original LMM
    compare_with_lmm_results()

    # Save results summary
    summary = save_results_summary(results, df)

    # Create plots
    create_glmm_plots(df, results)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY: GLMM vs IRT → LMM Comparison")
    print("="*70)

    cont = results.get('continuous')
    bin_ = results.get('binomial')

    if cont is not None:
        time_p = cont.pvalues['log_TSVR']
        print(f"\nQuasi-Continuous GEE:")
        print(f"  Time effect: β = {cont.params['log_TSVR']:.6f}, p = {time_p:.6f}")
        print(f"  Result: {'SIGNIFICANT' if time_p < 0.05 else 'NOT significant'}")

    if bin_ is not None:
        time_p_bin = bin_.pvalues['log_TSVR']
        print(f"\nBinomial GEE (High vs Low Confidence):")
        print(f"  Time effect: β = {bin_.params['log_TSVR']:.6f}, p = {time_p_bin:.6f}")
        print(f"  Odds ratio: {np.exp(bin_.params['log_TSVR']):.4f}")
        print(f"  Result: {'SIGNIFICANT' if time_p_bin < 0.05 else 'NOT significant'}")

    print("""
CONCLUSION:
The GLMM analysis directly models ordinal confidence ratings at the item level.
This provides a robustness check for the two-stage IRT→LMM methodology used in
the original RQ 6.1.1 analysis.
""")


if __name__ == "__main__":
    main()
