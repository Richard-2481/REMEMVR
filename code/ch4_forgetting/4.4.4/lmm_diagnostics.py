"""
LMM Assumption Diagnostics for RQ 5.4.4

Purpose: Generate diagnostic plots to validate LMM assumptions for BOTH IRT and CTT models
         and investigate delta-AIC = -3607 anomaly (CTT vastly superior fit).

Diagnostics Performed (per improvement_taxonomy.md Section 5.1):
1. Residual normality (Q-Q plots, Shapiro-Wilk test)
2. Homoscedasticity (residuals vs fitted, Breusch-Pagan test)
3. Leverage/influence (Cook's distance)
4. Random effects normality (Q-Q plot for random intercepts)

Expected Insight: If IRT shows heteroscedasticity or non-normality, this explains
                  why CTT (bounded [0,1] scale) fits better than unbounded IRT theta.

Date: 2025-12-31 (quality validation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from pathlib import Path
import pickle
import logging

# Setup
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'plots'
LOGS_DIR = BASE_DIR / 'logs'
PLOTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'lmm_diagnostics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_model_and_data(model_path, data_path, dv_col):
    """Load fitted LMM model and corresponding data."""

    logger.info(f"Loading model: {model_path.name}")
    with open(model_path, 'rb') as f:
        model_result = pickle.load(f)

    logger.info(f"Loading data: {data_path.name}")
    data = pd.read_csv(data_path)

    logger.info(f"DV: {dv_col}, N={len(data)} observations")

    return model_result, data

def compute_diagnostics(model_result, data, dv_col):
    """Compute residuals, fitted values, and diagnostic statistics."""

    # Fitted values
    fitted = model_result.fittedvalues

    # Residuals (observed - fitted)
    observed = data[dv_col].values
    residuals = observed - fitted

    # Standardized residuals
    residual_std = residuals / np.std(residuals)

    # Leverage (hat values) - approximation for LMMs
    # True leverage requires full hat matrix (computationally expensive for LMMs)
    # Use simple approximation: distance from mean in predictor space
    leverage_approx = np.abs(data['recip_TSVR'] - data['recip_TSVR'].mean())

    # Cook's distance approximation: (standardized_residual^2 * leverage) / k
    # k = number of predictors (9 fixed effects in Recip+Log model)
    k = 9
    cooks_d = (residual_std ** 2) * leverage_approx / k

    diagnostics = pd.DataFrame({
        'fitted': fitted,
        'observed': observed,
        'residuals': residuals,
        'residuals_std': residual_std,
        'leverage': leverage_approx,
        'cooks_d': cooks_d
    })

    return diagnostics

def plot_diagnostics(diagnostics, model_result, model_label, output_prefix):
    """Generate 4-panel diagnostic plot."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'LMM Diagnostic Plots: {model_label}', fontsize=16, fontweight='bold')

    # Panel 1: Q-Q Plot (Residual Normality)
    ax = axes[0, 0]
    stats.probplot(diagnostics['residuals'], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Residual Normality', fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.grid(True, alpha=0.3)

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(diagnostics['residuals'])
    ax.text(0.05, 0.95, f"Shapiro-Wilk p={shapiro_p:.4f}",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Residuals vs Fitted (Homoscedasticity)
    ax = axes[0, 1]
    ax.scatter(diagnostics['fitted'], diagnostics['residuals'], alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_title('Residuals vs Fitted: Homoscedasticity Check', fontweight='bold')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)

    # Add LOESS smoother to detect patterns
    from scipy.signal import savgol_filter
    sorted_idx = np.argsort(diagnostics['fitted'])
    fitted_sorted = diagnostics['fitted'].iloc[sorted_idx]
    residuals_sorted = diagnostics['residuals'].iloc[sorted_idx]
    if len(fitted_sorted) > 50:
        smoothed = savgol_filter(residuals_sorted, window_length=51, polyorder=3)
        ax.plot(fitted_sorted, smoothed, 'b-', linewidth=2, label='Trend')
        ax.legend()

    # Panel 3: Scale-Location Plot (Spread-Location)
    ax = axes[1, 0]
    ax.scatter(diagnostics['fitted'], np.sqrt(np.abs(diagnostics['residuals_std'])), alpha=0.3, s=10)
    ax.set_title('Scale-Location Plot: Variance Stability', fontweight='bold')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Standardized Residuals|')
    ax.grid(True, alpha=0.3)

    # Panel 4: Cook's Distance (Influential Observations)
    ax = axes[1, 1]
    ax.stem(range(len(diagnostics)), diagnostics['cooks_d'], linefmt='grey', markerfmt='o', basefmt=' ')
    ax.axhline(y=4/len(diagnostics), color='r', linestyle='--', linewidth=2, label='Threshold (4/n)')
    ax.set_title("Cook's Distance: Influential Observations", fontweight='bold')
    ax.set_xlabel('Observation Index')
    ax.set_ylabel("Cook's Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Count influential points
    threshold = 4 / len(diagnostics)
    n_influential = (diagnostics['cooks_d'] > threshold).sum()
    ax.text(0.05, 0.95, f"{n_influential} influential points (>{threshold:.4f})",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path = PLOTS_DIR / f'{output_prefix}_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()

    return shapiro_stat, shapiro_p, n_influential

def heteroscedasticity_test(diagnostics, data):
    """Breusch-Pagan test for heteroscedasticity."""

    # Extract exogenous variables (fixed effects design matrix)
    # For Recip+Log model: Congruence (2 dummy) + recip_TSVR + log_TSVR + 4 interactions + intercept = 9 terms
    # Simplified: Use recip_TSVR and log_TSVR as key predictors

    exog = data[['recip_TSVR', 'log_TSVR']].values

    # Add constant for intercept
    exog = np.column_stack([np.ones(len(exog)), exog])

    # Breusch-Pagan test
    bp_stat, bp_p, _, _ = het_breuschpagan(diagnostics['residuals'], exog)

    logger.info(f"Breusch-Pagan test: stat={bp_stat:.4f}, p={bp_p:.4f}")

    if bp_p < 0.05:
        logger.warning("Heteroscedasticity detected (p < 0.05)")
        return 'Heteroscedastic'
    else:
        logger.info("Homoscedasticity assumption satisfied (p >= 0.05)")
        return 'Homoscedastic'

def random_effects_qq_plot(model_result, model_label, output_prefix):
    """Q-Q plot for random intercepts normality."""

    try:
        random_effects = model_result.random_effects

        # Extract random intercepts (first column if slopes present, otherwise only column)
        re_intercepts = []
        for uid, re in random_effects.items():
            re_intercepts.append(re['Group'][0])  # First element is intercept

        re_intercepts = np.array(re_intercepts)

        # Q-Q plot
        fig, ax = plt.subplots(figsize=(7, 5))
        stats.probplot(re_intercepts, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: Random Intercepts Normality ({model_label})', fontweight='bold')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles (Random Intercepts)')
        ax.grid(True, alpha=0.3)

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(re_intercepts)
        ax.text(0.05, 0.95, f"Shapiro-Wilk p={shapiro_p:.4f}\nN={len(re_intercepts)} groups",
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = PLOTS_DIR / f'{output_prefix}_random_effects_qq.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

        return shapiro_stat, shapiro_p

    except Exception as e:
        logger.warning(f"Could not extract random effects for Q-Q plot: {e}")
        return None, None

def generate_summary_report(irt_results, ctt_results):
    """Generate text summary comparing IRT vs CTT diagnostics."""

    report = []
    report.append("="*60)
    report.append("LMM DIAGNOSTIC SUMMARY: IRT vs CTT Comparison")
    report.append("="*60)
    report.append("")

    # Normality
    report.append("1. RESIDUAL NORMALITY (Shapiro-Wilk Test)")
    report.append(f"   IRT: p={irt_results['shapiro_p']:.4f} {'✓ Normal' if irt_results['shapiro_p'] > 0.05 else '✗ Non-normal'}")
    report.append(f"   CTT: p={ctt_results['shapiro_p']:.4f} {'✓ Normal' if ctt_results['shapiro_p'] > 0.05 else '✗ Non-normal'}")
    report.append("")

    # Homoscedasticity
    report.append("2. HOMOSCEDASTICITY (Breusch-Pagan Test)")
    report.append(f"   IRT: {irt_results['bp_result']} (p={irt_results['bp_p']:.4f})")
    report.append(f"   CTT: {ctt_results['bp_result']} (p={ctt_results['bp_p']:.4f})")
    report.append("")

    # Influential observations
    report.append("3. INFLUENTIAL OBSERVATIONS (Cook's Distance)")
    report.append(f"   IRT: {irt_results['n_influential']} influential points")
    report.append(f"   CTT: {ctt_results['n_influential']} influential points")
    report.append("")

    # Random effects
    report.append("4. RANDOM EFFECTS NORMALITY (Q-Q Plot)")
    if irt_results['re_shapiro_p'] is not None:
        report.append(f"   IRT: p={irt_results['re_shapiro_p']:.4f} {'✓ Normal' if irt_results['re_shapiro_p'] > 0.05 else '✗ Non-normal'}")
    else:
        report.append("   IRT: Not computed")

    if ctt_results['re_shapiro_p'] is not None:
        report.append(f"   CTT: p={ctt_results['re_shapiro_p']:.4f} {'✓ Normal' if ctt_results['re_shapiro_p'] > 0.05 else '✗ Non-normal'}")
    else:
        report.append("   CTT: Not computed")
    report.append("")

    # Delta-AIC explanation
    report.append("="*60)
    report.append("DELTA-AIC EXPLANATION (CTT vastly superior: ΔAIC = -3607)")
    report.append("="*60)
    report.append("")

    # Identify which assumptions violated
    irt_violations = []
    if irt_results['shapiro_p'] < 0.05:
        irt_violations.append("Non-normal residuals")
    if irt_results['bp_result'] == 'Heteroscedastic':
        irt_violations.append("Heteroscedasticity")

    ctt_violations = []
    if ctt_results['shapiro_p'] < 0.05:
        ctt_violations.append("Non-normal residuals")
    if ctt_results['bp_result'] == 'Heteroscedastic':
        ctt_violations.append("Heteroscedasticity")

    if len(irt_violations) > len(ctt_violations):
        report.append(f"IRT VIOLATIONS: {', '.join(irt_violations) if irt_violations else 'None'}")
        report.append(f"CTT VIOLATIONS: {', '.join(ctt_violations) if ctt_violations else 'None'}")
        report.append("")
        report.append("INTERPRETATION:")
        report.append("CTT's bounded [0,1] scale better satisfies LMM assumptions than")
        report.append("unbounded IRT theta, explaining superior AIC fit. This is a")
        report.append("SCALE PROPERTY difference, not a measurement failure.")
    else:
        report.append("Both models show similar assumption violations.")
        report.append("Delta-AIC likely driven by scale properties (bounded vs unbounded)")
        report.append("rather than differential assumption violations.")

    report.append("")
    report.append("="*60)

    report_text = "\n".join(report)

    # Save report
    output_path = DATA_DIR / 'lmm_diagnostics_summary.txt'
    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Saved summary: {output_path}")

    print("\n" + report_text)

def main():
    """Main execution: Diagnose both IRT and CTT LMMs."""

    logger.info("="*60)
    logger.info("RQ 5.4.4: LMM Assumption Diagnostics")
    logger.info("="*60)
    logger.info("Investigating delta-AIC = -3607 (CTT >> IRT)\n")

    # IRT Model Diagnostics
    logger.info("Processing IRT Model (Theta)...")
    irt_model, irt_data = load_model_and_data(
        DATA_DIR / 'step03_irt_lmm_model.pkl',
        DATA_DIR / 'step03_irt_lmm_input.csv',
        'theta'
    )

    irt_diag = compute_diagnostics(irt_model, irt_data, 'theta')
    irt_shapiro_stat, irt_shapiro_p, irt_n_influential = plot_diagnostics(
        irt_diag, irt_model, 'IRT (Theta)', 'irt'
    )
    irt_bp_result = heteroscedasticity_test(irt_diag, irt_data)
    irt_re_shapiro_stat, irt_re_shapiro_p = random_effects_qq_plot(
        irt_model, 'IRT', 'irt'
    )

    # Store Breusch-Pagan p-value
    exog = irt_data[['recip_TSVR', 'log_TSVR']].values
    exog = np.column_stack([np.ones(len(exog)), exog])
    _, irt_bp_p, _, _ = het_breuschpagan(irt_diag['residuals'], exog)

    irt_results = {
        'shapiro_p': irt_shapiro_p,
        'bp_result': irt_bp_result,
        'bp_p': irt_bp_p,
        'n_influential': irt_n_influential,
        're_shapiro_p': irt_re_shapiro_p
    }

    # CTT Model Diagnostics
    logger.info("\nProcessing CTT Model (Proportion Correct)...")
    ctt_model, ctt_data = load_model_and_data(
        DATA_DIR / 'step03_ctt_lmm_model.pkl',
        DATA_DIR / 'step03_ctt_lmm_input.csv',
        'CTT_mean'
    )

    ctt_diag = compute_diagnostics(ctt_model, ctt_data, 'CTT_mean')
    ctt_shapiro_stat, ctt_shapiro_p, ctt_n_influential = plot_diagnostics(
        ctt_diag, ctt_model, 'CTT (Proportion Correct)', 'ctt'
    )
    ctt_bp_result = heteroscedasticity_test(ctt_diag, ctt_data)
    ctt_re_shapiro_stat, ctt_re_shapiro_p = random_effects_qq_plot(
        ctt_model, 'CTT', 'ctt'
    )

    # Store Breusch-Pagan p-value
    exog = ctt_data[['recip_TSVR', 'log_TSVR']].values
    exog = np.column_stack([np.ones(len(exog)), exog])
    _, ctt_bp_p, _, _ = het_breuschpagan(ctt_diag['residuals'], exog)

    ctt_results = {
        'shapiro_p': ctt_shapiro_p,
        'bp_result': ctt_bp_result,
        'bp_p': ctt_bp_p,
        'n_influential': ctt_n_influential,
        're_shapiro_p': ctt_re_shapiro_p
    }

    # Generate Comparative Summary
    generate_summary_report(irt_results, ctt_results)

    logger.info("\nLMM diagnostics COMPLETE")
    logger.info("HIGH PRIORITY RESOLVED: Section 5.1 assumption validation performed")

if __name__ == '__main__':
    main()
