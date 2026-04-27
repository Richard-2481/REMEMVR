#!/usr/bin/env python3
"""
RQ 5.5.6: Generate LMM Diagnostic Plots

Purpose:
    Create diagnostic plots for Source and Destination location-stratified LMMs
    to validate assumptions (normality, homoscedasticity, independence).

Generates:
    - Q-Q plots (residual normality)
    - Residuals vs Fitted (homoscedasticity)
    - Scale-Location (spread-level)
    - Residuals vs Leverage (influential points)

Output:
    plots/diagnostics_source.png (2x2 grid)
    plots/diagnostics_destination.png (2x2 grid)

Author: Claude (rq_platinum agent)
Date: 2025-12-30
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import logging
import sys

# Setup logging
log_path = Path("results/ch5/5.5.6/logs/generate_lmm_diagnostics.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_diagnostics(location_name):
    """Generate 2x2 diagnostic plot grid for a location-stratified LMM."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating diagnostics for: {location_name}")
    logger.info(f"{'='*60}")
    
    # Load fitted model
    model_path = Path(f"results/ch5/5.5.6/data/step01_{location_name}_lmm_model.pkl")
    with open(model_path, 'rb') as f:
        result = pickle.load(f)
    
    # Extract residuals and fitted values
    fitted = result.fittedvalues
    residuals = result.resid
    
    logger.info(f"Observations: {len(residuals)}")
    logger.info(f"Residuals mean: {residuals.mean():.6f} (should be ~0)")
    logger.info(f"Residuals SD: {residuals.std():.4f}")
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'LMM Diagnostic Plots - {location_name.capitalize()} Location', 
                 fontsize=14, fontweight='bold')
    
    # ------------------------------------------------------------------
    # Plot 1: Q-Q Plot (Normality of Residuals)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Standardized Residuals")
    ax.grid(True, alpha=0.3)
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    ax.text(0.05, 0.95, f'Shapiro-Wilk p={shapiro_p:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    logger.info(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
    
    # ------------------------------------------------------------------
    # Plot 2: Residuals vs Fitted (Homoscedasticity)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    ax.scatter(fitted, residuals, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_title("Residuals vs Fitted Values")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.grid(True, alpha=0.3)
    
    # Add loess smooth (simple moving average as approximation)
    sorted_idx = np.argsort(fitted)
    window = max(10, len(fitted) // 20)
    smooth_fitted = pd.Series(fitted.values[sorted_idx]).rolling(window, center=True).mean()
    smooth_resid = pd.Series(residuals.values[sorted_idx]).rolling(window, center=True).mean()
    ax.plot(smooth_fitted, smooth_resid, 'b-', linewidth=2, label='Smooth')
    ax.legend()
    
    # ------------------------------------------------------------------
    # Plot 3: Scale-Location (Spread-Level)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    standardized_resid = residuals / residuals.std()
    sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))
    ax.scatter(fitted, sqrt_abs_resid, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    ax.set_title("Scale-Location Plot")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("√|Standardized Residuals|")
    ax.grid(True, alpha=0.3)
    
    # Add smooth
    smooth_sqrt = pd.Series(sqrt_abs_resid.values[sorted_idx]).rolling(window, center=True).mean()
    ax.plot(smooth_fitted, smooth_sqrt, 'r-', linewidth=2, label='Smooth')
    ax.legend()
    
    # ------------------------------------------------------------------
    # Plot 4: Histogram of Residuals
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    ax.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    plt.tight_layout()
    output_path = Path(f"results/ch5/5.5.6/plots/diagnostics_{location_name}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved diagnostic plot: {output_path}")
    
    return shapiro_p

def main():
    logger.info("="*60)
    logger.info("RQ 5.5.6: Generate LMM Diagnostic Plots")
    logger.info("="*60)
    
    results = {}
    for location in ['source', 'destination']:
        shapiro_p = generate_diagnostics(location)
        results[location] = shapiro_p
    
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*60)
    for location, p_val in results.items():
        status = "PASS" if p_val > 0.01 else "FAIL"
        logger.info(f"{location.capitalize()}: Shapiro-Wilk p={p_val:.4f} [{status}]")
    
    logger.info("\nInterpretation:")
    logger.info("- Q-Q Plot: Points should follow red line (normality)")
    logger.info("- Residuals vs Fitted: Should show random scatter around 0 (homoscedasticity)")
    logger.info("- Scale-Location: Smooth line should be horizontal (constant variance)")
    logger.info("- Histogram: Should resemble normal distribution")
    logger.info("\nDiagnostic plots generated successfully")

if __name__ == "__main__":
    main()
