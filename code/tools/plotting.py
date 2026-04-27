"""
Generic plotting functions for REMEMVR analysis pipeline.

Provides reusable, configurable plotting functions with consistent styling
for trajectory plots, diagnostics, histograms, and IRT probability conversions.

Key Features:
- Loads styling from config/plotting.yaml
- Publication-ready defaults (300 DPI, clear fonts)
- Saves both PNG and CSV data for reproducibility
- Supports grouped visualizations by domain/factor

Functions:
    set_plot_style_defaults() - Apply consistent matplotlib/seaborn styling
    plot_trajectory() - Trajectory with fitted curves + observed errorhars
    plot_diagnostics() - 2x2 diagnostic grid for regression validation
    plot_histogram_by_group() - Grouped histograms
    plot_comparison_bars() - Grouped bar plots with CI for comparisons
    convert_theta_to_probability() - IRT response function
    save_plot_with_data() - Save PNG + CSV simultaneously

Author: Claude (REMEMVR Project)
Date: 2025-01-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import List, Optional, Dict, Tuple, Any

from tools.config import load_config_from_file


# =============================================================================
# Style Setup
# =============================================================================

def set_plot_style_defaults(config_path: Optional[Path] = None) -> None:
    """
    Apply consistent matplotlib and seaborn styling from config.

    Loads plotting parameters from config/plotting.yaml and applies them
    to matplotlib rcParams. If config not found, uses sensible defaults.

    Args:
        config_path: Optional path to plotting.yaml. If None, uses default.

    Raises:
        None - gracefully falls back to defaults if config missing

    Example:
        >>> set_plot_style_defaults()  # Uses config/plotting.yaml
        >>> plt.plot([0, 1], [0, 1])  # Will use configured style
    """
    # Set seaborn style first (base style)
    sns.set_style("whitegrid")

    # Try to load config
    try:
        if config_path is None:
            config = load_config_from_file('plotting')
        else:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    except (FileNotFoundError, KeyError, Exception):
        # Use defaults if config not found
        config = {
            'dpi': 300,
            'font_size': 11,
            'axes_labelsize': 12,
            'axes_titlesize': 13,
            'legend_fontsize': 10,
            'line_width': 2.5,
            'marker_size': 8
        }

    # Apply matplotlib rcParams
    plt.rcParams['figure.dpi'] = config.get('dpi', 300)
    plt.rcParams['font.size'] = config.get('font_size', 11)
    plt.rcParams['axes.labelsize'] = config.get('axes_labelsize', 12)
    plt.rcParams['axes.titlesize'] = config.get('axes_titlesize', 13)
    plt.rcParams['legend.fontsize'] = config.get('legend_fontsize', 10)


# =============================================================================
# Trajectory Plotting
# =============================================================================

def plot_trajectory(
    time_pred: np.ndarray,
    fitted_curves: Dict[str, np.ndarray],
    observed_data: pd.DataFrame,
    time_col: str = 'Time',
    value_col: str = 'Value',
    group_col: str = 'Group',
    xlabel: str = 'Time',
    ylabel: str = 'Value',
    title: str = 'Trajectory Plot',
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path] = None,
    show_errorbar: bool = True,
    annotation: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create trajectory plot with fitted curves and observed data points.

    Plots smooth fitted trajectories (lines) alongside observed data points
    with error bars (mean ± SEM). Useful for visualizing forgetting curves,
    growth curves, or any longitudinal data.

    Args:
        time_pred: Time points for fitted curve (1D array, e.g., np.linspace(0, 6, 100))
        fitted_curves: Dict mapping group names to fitted values {group: array}
        observed_data: DataFrame with observed data (long format)
        time_col: Column name for time variable in observed_data
        value_col: Column name for observed values
        group_col: Column name for grouping variable (e.g., 'Domain', 'Factor')
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        colors: Optional dict mapping groups to colors {group: hex_color}
        figsize: Figure size (width, height) in inches
        output_path: Optional path to save plot
        show_errorbar: Whether to show error bars (default True)
        annotation: Optional text annotation (placed bottom-left)

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axes

    Example:
        >>> time_pred = np.linspace(0, 6, 100)
        >>> fitted = {'What': 0.5 - 0.1*time_pred, 'Where': 0.4 - 0.08*time_pred}
        >>> observed = pd.DataFrame({'Time': [0,1,3,6]*2, 'Value': [...], 'Group': [...]})
        >>> fig, ax = plot_trajectory(time_pred, fitted, observed)
    """
    # Calculate observed statistics (mean ± SEM per group × time)
    observed_stats = observed_data.groupby([group_col, time_col]).agg(
        mean=( value_col, 'mean'),
        sem=(value_col, lambda x: x.std() / np.sqrt(len(x))),
        n=(value_col, 'count')
    ).reset_index()

    # Get default colors if not provided
    if colors is None:
        try:
            config = load_config_from_file('plotting')
            colors = config.get('colors', {
                'What': '#E74C3C',
                'Where': '#3498DB',
                'When': '#2ECC71'
            })
        except:
            # Fallback colors
            colors = {
                'What': '#E74C3C',
                'Where': '#3498DB',
                'When': '#2ECC71'
            }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each group
    for group, fitted_values in fitted_curves.items():
        color = colors.get(group, None)  # None uses default matplotlib color cycle

        # Plot fitted trajectory (smooth curve)
        ax.plot(time_pred, fitted_values,
                color=color, linewidth=2.5, label=f'{group} (fitted)',
                alpha=0.9)

        # Plot observed data points (mean ± SEM)
        group_data = observed_stats[observed_stats[group_col] == group]

        if show_errorbar and len(group_data) > 0:
            ax.errorbar(group_data[time_col], group_data['mean'],
                        yerr=group_data['sem'],
                        fmt='o', color=color, markersize=8,
                        capsize=5, capthick=2, alpha=0.7,
                        label=f'{group} (observed)')

    # Formatting
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Add annotation if provided
    if annotation:
        ax.text(0.02, 0.02, annotation,
                transform=ax.transAxes,
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                verticalalignment='bottom')

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


# =============================================================================
# Diagnostic Plotting
# =============================================================================

def plot_diagnostics(
    df: pd.DataFrame,
    fitted_col: str = 'fitted',
    residuals_col: str = 'residuals',
    group_col: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create 2x2 diagnostic plot grid for regression model validation.

    Creates four diagnostic plots:
    - (A) Residuals vs Fitted: Check linearity and homoscedasticity
    - (B) Q-Q Plot: Check normality of residuals
    - (C) Scale-Location: Check homoscedasticity with sqrt(|residuals|)
    - (D) Residuals by Group: Check group-level distributions

    Args:
        df: DataFrame with fitted values and residuals
        fitted_col: Column name for fitted values
        residuals_col: Column name for residuals
        group_col: Optional column name for grouping (e.g., 'Domain')
        figsize: Figure size (width, height) in inches
        output_path: Optional path to save plot

    Returns:
        fig: Matplotlib figure
        axes: 2x2 array of axes

    Example:
        >>> df = pd.DataFrame({'fitted': [...], 'residuals': [...], 'group': [...]})
        >>> fig, axes = plot_diagnostics(df)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # (A) Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(df[fitted_col], df[residuals_col], alpha=0.4, s=10)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('(A) Residuals vs Fitted', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add lowess smooth
    try:
        sorted_idx = np.argsort(df[fitted_col])
        fitted_sorted = df[fitted_col].iloc[sorted_idx]
        resid_sorted = df[residuals_col].iloc[sorted_idx]

        window = min(51, len(fitted_sorted) // 10 * 2 + 1)  # Ensure odd
        if window >= 3:
            smooth = pd.Series(resid_sorted).rolling(window=window, center=True).mean()
            ax.plot(fitted_sorted, smooth, color='blue', linewidth=2, alpha=0.8)
    except Exception:
        pass  # Skip smooth if it fails

    # (B) Q-Q Plot (Normality of residuals)
    ax = axes[0, 1]
    stats.probplot(df[residuals_col], dist="norm", plot=ax)
    ax.set_title('(B) Normal Q-Q Plot', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (C) Scale-Location (Homoscedasticity)
    ax = axes[1, 0]
    sqrt_abs_resid = np.sqrt(np.abs(df[residuals_col]))
    ax.scatter(df[fitted_col], sqrt_abs_resid, alpha=0.4, s=10)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Residuals|')
    ax.set_title('(C) Scale-Location', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add smooth
    try:
        window = min(51, len(fitted_sorted) // 10 * 2 + 1)
        if window >= 3:
            sorted_idx = np.argsort(df[fitted_col])
            fitted_sorted = df[fitted_col].iloc[sorted_idx]
            sqrt_resid_sorted = sqrt_abs_resid.iloc[sorted_idx]
            smooth = pd.Series(sqrt_resid_sorted).rolling(window=window, center=True).mean()
            ax.plot(fitted_sorted, smooth, color='red', linewidth=2, alpha=0.8)
    except Exception:
        pass

    # (D) Residuals by Group (if group_col provided)
    ax = axes[1, 1]
    if group_col and group_col in df.columns:
        # Get colors
        try:
            config = load_config_from_file('plotting')
            colors = config.get('colors', {})
        except:
            colors = {}

        for group in df[group_col].unique():
            group_resid = df[df[group_col] == group][residuals_col]
            color = colors.get(group, None)
            ax.hist(group_resid, alpha=0.5, bins=30, label=str(group), color=color)

        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('(D) Residuals Distribution by Group', fontweight='bold')
        ax.legend()
    else:
        # Overall histogram if no groups
        ax.hist(df[residuals_col], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('(D) Residuals Distribution', fontweight='bold')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, axes


# =============================================================================
# Histogram Plotting
# =============================================================================

def plot_histogram_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    xlabel: str = 'Value',
    ylabel: str = 'Frequency',
    title: str = 'Histogram by Group',
    bins: int = 20,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path] = None,
    vline: Optional[float] = None,
    vline_label: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create grouped histogram with overlapping distributions.

    Args:
        df: DataFrame with values and groups
        value_col: Column name for values to histogram
        group_col: Column name for grouping variable
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        bins: Number of histogram bins
        colors: Optional dict mapping groups to colors
        figsize: Figure size (width, height)
        output_path: Optional path to save plot
        vline: Optional x-coordinate for vertical reference line
        vline_label: Optional label for vertical line

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axes

    Example:
        >>> df = pd.DataFrame({'Value': [...], 'Group': ['What', 'Where', 'When']})
        >>> fig, ax = plot_histogram_by_group(df, 'Value', 'Group')
    """
    # Get default colors if not provided
    if colors is None:
        try:
            config = load_config_from_file('plotting')
            colors = config.get('colors', {})
        except:
            colors = {}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram for each group
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][value_col]
        color = colors.get(group, None)

        ax.hist(group_data, alpha=0.6, bins=bins,
                label=f'{group} (n={len(group_data)})',
                color=color)

    # Formatting
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical line if specified
    if vline is not None:
        ax.axvline(x=vline, color='black', linestyle='--', linewidth=1, alpha=0.5)
        if vline_label:
            ax.text(vline + 0.1, ax.get_ylim()[1] * 0.95, vline_label,
                    fontsize=9, style='italic')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


# =============================================================================
# IRT Utilities
# =============================================================================

def convert_theta_to_probability(
    theta: np.ndarray,
    discrimination: float,
    difficulty: float
) -> np.ndarray:
    """
    Convert IRT ability (theta) to probability of correct response.

    Uses the 2-parameter logistic (2PL) IRT response function:
        P(correct) = 1 / (1 + exp(-(a * (θ - b))))

    where:
        θ (theta) = ability
        a = discrimination (how well item differentiates ability levels)
        b = difficulty (ability level at which P=0.5)

    Args:
        theta: Ability parameter (scalar or array)
        discrimination: Item discrimination parameter (a > 0)
        difficulty: Item difficulty parameter (b, can be any real number)

    Returns:
        prob: Probability of correct response (0 to 1)

    Example:
        >>> theta = np.array([-2, -1, 0, 1, 2])
        >>> prob = convert_theta_to_probability(theta, discrimination=1.5, difficulty=0.0)
        >>> print(prob)  # [0.047, 0.18, 0.5, 0.82, 0.953]

    Note:
        - When θ = b, P ≈ 0.5
        - Higher discrimination = steeper response curve
        - This function is vectorized (works with numpy arrays)
    """
    # Ensure numpy array for vectorization
    theta = np.asarray(theta)

    # IRT 2PL response function
    prob = 1 / (1 + np.exp(-(discrimination * (theta - difficulty))))

    return prob


# =============================================================================
# Save Utilities
# =============================================================================

def save_plot_with_data(
    fig: plt.Figure,
    output_path: Path,
    data: Optional[pd.DataFrame] = None,
    dpi: int = 300
) -> None:
    """
    Save plot as PNG and optionally save associated data as CSV.

    Saves matplotlib figure and corresponding data for reproducibility.
    CSV is saved with same name as PNG but .csv extension.

    Args:
        fig: Matplotlib figure to save
        output_path: Path for PNG file (e.g., "trajectory.png")
        data: Optional DataFrame to save as CSV
        dpi: DPI for PNG output (default 300 for publication quality)

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([0, 1, 2], [0, 1, 4])
        >>> data = pd.DataFrame({'x': [0,1,2], 'y': [0,1,4]})
        >>> save_plot_with_data(fig, Path("plot.png"), data)
        # Saves: plot.png and plot.csv
    """
    # Ensure Path object
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    # Save data if provided
    if data is not None:
        csv_path = output_path.with_suffix('.csv')
        data.to_csv(csv_path, index=False)


def plot_trajectory_probability(
    df_thetas: pd.DataFrame,
    item_parameters_path: Path,
    time_var: str = 'test',
    factors: List[str] = None,
    title: str = "Memory Trajectory (Probability Scale)",
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
    show_errorbar: bool = True
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot trajectory with theta transformed to probability scale (Decision D069).

    Implements dual-scale trajectory plotting:
    - Theta scale: Statistical rigor, psychometrician-interpretable
    - Probability scale: General audience interpretable, reviewer-friendly

    Uses IRT 2PL transformation: P = 1 / (1 + exp(-(a * (theta - b))))
    where:
    - a = mean discrimination from Pass 2 item parameters
    - b = 0 (reference difficulty)
    - theta = theta scores

    Args:
        df_thetas: DataFrame with theta scores (wide format with UID rows)
                   Must contain: UID, {time_var}, {factor}_Theta columns
        item_parameters_path: Path to item_parameters.csv (Pass 2 output)
        time_var: Time variable column name (default: 'test')
        factors: List of factor names to plot (default: infer from columns)
        title: Plot title
        figsize: Figure size (width, height) in inches
        colors: Optional dict mapping factors to colors
        output_path: Optional path to save plot
        show_errorbar: Whether to show error bars (default True)

    Returns:
        Tuple of:
        - fig: Matplotlib figure
        - ax: Matplotlib axes
        - prob_data: DataFrame with probability-transformed scores

    Example:
        fig, ax, prob_data = plot_trajectory_probability(
            df_thetas=df_thetas,
            item_parameters_path=Path("data/item_parameters.csv"),
            time_var='Days',
            factors=['What', 'Where', 'When']
        )

    Decision D069 Context:
        Dual-scale reporting enhances interpretability without sacrificing rigor.
        Theta scale preserves statistical properties (IRT estimates), while probability
        scale provides intuitive metric (0-100% correct) for general audience.
    """
    print("\n" + "=" * 60)
    print("PROBABILITY-SCALE TRAJECTORY PLOT (Decision D069)")
    print("=" * 60)

    # Read item parameters to get mean discrimination
    df_items = pd.read_csv(item_parameters_path)

    # Calculate mean discrimination across all items
    mean_a = df_items['a'].mean()
    print(f"Mean item discrimination: {mean_a:.3f}")

    # Infer factors if not provided
    if factors is None:
        factors = [col.replace('_Theta', '') for col in df_thetas.columns if col.endswith('_Theta')]

    print(f"Transforming factors: {factors}")

    # Transform theta to probability
    prob_data = df_thetas.copy()

    for factor in factors:
        theta_col = f'{factor}_Theta'
        prob_col = f'{factor}_Probability'

        if theta_col not in df_thetas.columns:
            print(f"  Warning: {theta_col} not found in data. Skipping.")
            continue

        # IRT 2PL transformation: P = 1 / (1 + exp(-(a * (theta - b))))
        # Using b=0 (reference difficulty)
        theta = df_thetas[theta_col]
        probability = 1 / (1 + np.exp(-(mean_a * theta)))

        # Convert to percentage scale (0-100%)
        prob_data[prob_col] = probability * 100

        print(f"  {factor}: theta range [{theta.min():.2f}, {theta.max():.2f}] -> P% range [{prob_data[prob_col].min():.1f}%, {prob_data[prob_col].max():.1f}%]")

    # Get default colors if not provided
    if colors is None:
        try:
            config = load_config_from_file('plotting')
            colors = config.get('colors', {})
        except Exception:
            colors = {}

    # Default color cycle if no config colors
    default_colors = plt.cm.tab10.colors
    factor_colors = {}
    for i, factor in enumerate(factors):
        factor_colors[factor] = colors.get(factor, default_colors[i % len(default_colors)])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each factor
    for factor in factors:
        prob_col = f'{factor}_Probability'

        if prob_col not in prob_data.columns:
            continue

        # Calculate mean and SEM per time point
        grouped = prob_data.groupby(time_var)[prob_col].agg(['mean', 'sem', 'count']).reset_index()

        time_points = grouped[time_var].values
        means = grouped['mean'].values
        sems = grouped['sem'].values

        color = factor_colors.get(factor, None)

        # Plot line with markers
        ax.plot(time_points, means, 'o-', label=factor, color=color, linewidth=2.5, markersize=8)

        # Add error bars if requested
        if show_errorbar:
            ax.errorbar(time_points, means, yerr=sems, fmt='none', color=color, capsize=4, alpha=0.7)

    # Formatting
    ax.set_xlabel(time_var.replace('_', ' ').title())
    ax.set_ylabel("Probability Correct (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(title="Factor")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    print("=" * 60 + "\n")

    return fig, ax, prob_data


def prepare_piecewise_plot_data(
    df_input: pd.DataFrame,
    lmm_result,
    segment_col: str,
    factor_col: str,
    segment_values: List[str],
    factor_values: List[str],
    days_within_col: str = 'Days_within',
    theta_col: str = 'theta',
    early_grid_points: int = 20,
    late_grid_points: int = 60,
    ci_level: float = 0.95
) -> Dict[str, pd.DataFrame]:
    """
    Prepare piecewise trajectory plot data with observed means and model predictions.

    Aggregates observed theta scores by segment and factor, computes 95% CI, and
    generates model predictions on a grid of Days_within values for smooth trajectory
    lines. Designed for piecewise LMM plots with separate Early and Late panels.

    Args:
        df_input: Input DataFrame with piecewise LMM data
                  Required columns: segment_col, factor_col, days_within_col, theta_col
        lmm_result: Fitted LMM model object (statsmodels MixedLMResults)
        segment_col: Column name for segment variable (e.g., 'Segment')
        factor_col: Column name for factor variable (e.g., 'Congruence', 'domain')
        segment_values: List of segment names, first is Early (e.g., ['Early', 'Late'])
        factor_values: List of factor levels (e.g., ['Common', 'Congruent', 'Incongruent'])
        days_within_col: Column name for within-segment time (default: 'Days_within')
        theta_col: Column name for theta scores (default: 'theta')
        early_grid_points: Number of prediction points for Early segment (default: 20)
        late_grid_points: Number of prediction points for Late segment (default: 60)
        ci_level: Confidence interval level (default: 0.95)

    Returns:
        Dict with keys 'early' and 'late', each containing a DataFrame with columns:
        - Days_within: Time within segment
        - {factor_col}: Factor level
        - theta_observed: Observed mean theta
        - CI_lower_observed: Lower CI bound
        - CI_upper_observed: Upper CI bound
        - theta_predicted: Model predicted theta
        - Data_Type: 'observed' or 'predicted'

    Example:
        >>> plot_data = prepare_piecewise_plot_data(
        ...     df_input=df_piecewise,
        ...     lmm_result=lmm_model,
        ...     segment_col='Segment',
        ...     factor_col='Congruence',
        ...     segment_values=['Early', 'Late'],
        ...     factor_values=['Common', 'Congruent', 'Incongruent']
        ... )
        >>> df_early = plot_data['early']  # Early segment plot data
        >>> df_late = plot_data['late']    # Late segment plot data

    Reference:
        Based on RQ 5.2 step05_prepare_piecewise_plot_data.py but generalized
        for any segment/factor variable names.
    """
    # Determine Early vs Late segment names
    early_segment = segment_values[0]
    late_segment = segment_values[1] if len(segment_values) > 1 else segment_values[0]

    # Calculate z-score for CI
    from scipy import stats as scipy_stats
    z_score = scipy_stats.norm.ppf((1 + ci_level) / 2)

    # Prepare output storage
    result = {}

    # Process each segment separately
    for segment_name, segment_value, grid_points, max_days in [
        ('early', early_segment, early_grid_points, 1.0),
        ('late', late_segment, late_grid_points, 6.0)
    ]:
        # Filter data for this segment
        df_segment = df_input[df_input[segment_col] == segment_value].copy()

        if len(df_segment) == 0:
            # Create empty dataframe if no data for this segment
            result[segment_name] = pd.DataFrame(columns=[
                days_within_col, factor_col, 'theta_observed',
                'CI_lower_observed', 'CI_upper_observed',
                'theta_predicted', 'Data_Type'
            ])
            continue

        # Aggregate observed data by factor ONLY (not by days_within since it varies)
        # We'll compute overall observed mean + CI for each factor level
        grouped = df_segment.groupby([factor_col])[theta_col].agg(
            theta_mean=('mean'),
            theta_sem=('sem'),
            n=('count')
        ).reset_index()

        # Compute representative Days_within (median for this factor-segment combo)
        days_median = df_segment.groupby([factor_col])[days_within_col].median().reset_index()
        grouped = grouped.merge(days_median, on=factor_col)

        # Compute 95% CI
        grouped['CI_lower_observed'] = grouped['theta_mean'] - z_score * grouped['theta_sem']
        grouped['CI_upper_observed'] = grouped['theta_mean'] + z_score * grouped['theta_sem']
        grouped['Data_Type'] = 'observed'

        # Rename for output consistency
        grouped = grouped.rename(columns={'theta_mean': 'theta_observed'})

        # Generate prediction grid for smooth trajectories
        pred_data = []
        days_grid = np.linspace(0, max_days, grid_points)

        for factor in factor_values:
            for days_val in days_grid:
                # Create row for prediction
                pred_row = pd.DataFrame({
                    segment_col: [segment_value],
                    factor_col: [factor],
                    days_within_col: [days_val]
                })

                try:
                    # Get model prediction (population-level only, no random effects)
                    pred_theta = lmm_result.predict(pred_row)
                    theta_pred_val = pred_theta.values[0] if hasattr(pred_theta, 'values') else pred_theta[0]
                except Exception as e:
                    # Fallback to NaN if prediction fails
                    theta_pred_val = np.nan

                pred_data.append({
                    days_within_col: days_val,
                    factor_col: factor,
                    'theta_observed': np.nan,
                    'CI_lower_observed': np.nan,
                    'CI_upper_observed': np.nan,
                    'theta_predicted': theta_pred_val,
                    'Data_Type': 'predicted'
                })

        df_predictions = pd.DataFrame(pred_data)

        # For observed data, get corresponding predictions at the median Days_within
        for idx, row in grouped.iterrows():
            pred_row = pd.DataFrame({
                segment_col: [segment_value],
                factor_col: [row[factor_col]],
                days_within_col: [row[days_within_col]]
            })

            try:
                pred_theta = lmm_result.predict(pred_row)
                grouped.loc[idx, 'theta_predicted'] = pred_theta.values[0] if hasattr(pred_theta, 'values') else pred_theta[0]
            except Exception:
                grouped.loc[idx, 'theta_predicted'] = np.nan

        # Combine observed and predictions
        # Select and reorder columns
        observed_cols = [days_within_col, factor_col, 'theta_observed',
                        'CI_lower_observed', 'CI_upper_observed',
                        'theta_predicted', 'Data_Type']

        # Ensure all columns exist
        for col in observed_cols:
            if col not in grouped.columns:
                grouped[col] = np.nan

        grouped_clean = grouped[observed_cols]

        df_combined = pd.concat([grouped_clean, df_predictions], ignore_index=True)
        df_combined = df_combined.sort_values([factor_col, days_within_col])

        result[segment_name] = df_combined

    return result


# =============================================================================
# Piecewise Trajectory Plotting
# =============================================================================

def plot_comparison_bars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    facet_col: Optional[str] = None,
    ci_lower_col: Optional[str] = None,
    ci_upper_col: Optional[str] = None,
    annotation_col: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    orientation: str = 'vertical',
    output_path: Optional[Path] = None,
    bar_width: float = 0.35,
    show_legend: bool = True
) -> Tuple[plt.Figure, Any]:
    """
    Create grouped bar plot with confidence intervals for comparison visualizations.

    Supports grouped bars (multiple bars per x-category), optional faceting,
    confidence interval error bars, and custom annotations. Designed for
    model comparison plots (e.g., IRT vs CTT) and correlation comparisons.

    Args:
        df: DataFrame with data to plot
        x_col: Column name for x-axis categories
        y_col: Column name for y-axis values
        group_col: Optional column name for grouping (creates grouped bars)
        facet_col: Optional column name for faceting (creates subplots)
        ci_lower_col: Optional column name for CI lower bound
        ci_upper_col: Optional column name for CI upper bound
        annotation_col: Optional column name for annotations (e.g., significance stars)
        colors: Optional dict mapping group names to colors
        xlabel: X-axis label (default: x_col with underscores replaced)
        ylabel: Y-axis label (default: y_col with underscores replaced)
        title: Plot title
        figsize: Figure size (width, height) in inches
        orientation: Bar orientation - 'vertical' or 'horizontal'
        output_path: Optional path to save plot
        bar_width: Width of individual bars (default 0.35)
        show_legend: Whether to show legend (default True)

    Returns:
        If facet_col is None:
            Tuple of (fig, ax) where ax is single Axes
        If facet_col is provided:
            Tuple of (fig, axes) where axes is array of Axes

    Raises:
        ValueError: If DataFrame is empty
        KeyError: If required column not in DataFrame

    Example:
        >>> # Simple bar plot
        >>> df = pd.DataFrame({'model': ['A', 'B'], 'AIC': [100, 110]})
        >>> fig, ax = plot_comparison_bars(df, 'model', 'AIC')

        >>> # Grouped bars with CI
        >>> df = pd.DataFrame({
        ...     'location': ['source', 'source', 'dest', 'dest'],
        ...     'version': ['full', 'purified'] * 2,
        ...     'r': [0.93, 0.94, 0.80, 0.87],
        ...     'CI_lower': [0.92, 0.93, 0.76, 0.85],
        ...     'CI_upper': [0.95, 0.95, 0.83, 0.89]
        ... })
        >>> fig, ax = plot_comparison_bars(
        ...     df, x_col='version', y_col='r', group_col='location',
        ...     ci_lower_col='CI_lower', ci_upper_col='CI_upper'
        ... )

        >>> # Faceted plot
        >>> fig, axes = plot_comparison_bars(
        ...     df, x_col='model', y_col='AIC', facet_col='location'
        ... )
    """
    # Input validation
    if df.empty:
        raise ValueError("DataFrame is empty")

    if x_col not in df.columns:
        raise KeyError(f"Column '{x_col}' not found in DataFrame")

    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not found in DataFrame")

    # Set default labels
    if xlabel is None:
        xlabel = x_col.replace('_', ' ').title()
    if ylabel is None:
        ylabel = y_col.replace('_', ' ').title()

    # Get default colors if not provided
    if colors is None:
        try:
            config = load_config_from_file('plotting')
            colors = config.get('colors', {})
        except:
            colors = {}

    # Determine if we need faceting
    if facet_col is not None:
        facet_values = sorted(df[facet_col].unique())
        n_facets = len(facet_values)

        # Create subplots
        fig, axes = plt.subplots(1, n_facets, figsize=(figsize[0] * n_facets / 2, figsize[1]),
                                 squeeze=False)
        axes = axes.flatten()

        # Plot each facet
        for i, facet_value in enumerate(facet_values):
            df_facet = df[df[facet_col] == facet_value]
            _plot_single_comparison_bars(
                ax=axes[i],
                df=df_facet,
                x_col=x_col,
                y_col=y_col,
                group_col=group_col,
                ci_lower_col=ci_lower_col,
                ci_upper_col=ci_upper_col,
                annotation_col=annotation_col,
                colors=colors,
                xlabel=xlabel,
                ylabel=ylabel if i == 0 else None,  # Only first subplot gets ylabel
                title=f"{facet_col.replace('_', ' ').title()}: {facet_value}",
                orientation=orientation,
                bar_width=bar_width,
                show_legend=show_legend
            )

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, axes

    else:
        # Single plot (no faceting)
        fig, ax = plt.subplots(figsize=figsize)

        _plot_single_comparison_bars(
            ax=ax,
            df=df,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            ci_lower_col=ci_lower_col,
            ci_upper_col=ci_upper_col,
            annotation_col=annotation_col,
            colors=colors,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            orientation=orientation,
            bar_width=bar_width,
            show_legend=show_legend
        )

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax


def _plot_single_comparison_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    ci_lower_col: Optional[str] = None,
    ci_upper_col: Optional[str] = None,
    annotation_col: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    orientation: str = 'vertical',
    bar_width: float = 0.35,
    show_legend: bool = True
) -> None:
    """
    Internal helper function to plot a single comparison bar chart.

    This function handles the actual plotting logic for plot_comparison_bars().
    Not intended to be called directly by users.
    """
    # Get unique x categories
    x_categories = df[x_col].unique()
    x_positions = np.arange(len(x_categories))

    # Determine if we have grouping
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].unique()
        n_groups = len(groups)

        # Calculate bar positions for grouped bars
        offset = bar_width * (n_groups - 1) / 2
        group_positions = {
            group: x_positions + i * bar_width - offset
            for i, group in enumerate(groups)
        }

        # Plot each group
        for group in groups:
            df_group = df[df[group_col] == group]

            # Get y-values in order of x_categories
            y_values = []
            ci_lower_values = []
            ci_upper_values = []

            for x_cat in x_categories:
                df_cat = df_group[df_group[x_col] == x_cat]
                if len(df_cat) > 0:
                    y_values.append(df_cat[y_col].values[0])

                    if ci_lower_col and ci_upper_col:
                        ci_lower_values.append(df_cat[ci_lower_col].values[0])
                        ci_upper_values.append(df_cat[ci_upper_col].values[0])
                else:
                    y_values.append(0)
                    if ci_lower_col and ci_upper_col:
                        ci_lower_values.append(0)
                        ci_upper_values.append(0)

            # Get color
            color = colors.get(group, None) if colors else None

            # Plot bars
            positions = group_positions[group]
            if orientation == 'vertical':
                bars = ax.bar(positions, y_values, bar_width, label=group, color=color, alpha=0.8)

                # Add error bars if provided
                if ci_lower_col and ci_upper_col and ci_lower_values and ci_upper_values:
                    yerr_lower = [y_values[i] - ci_lower_values[i] for i in range(len(y_values))]
                    yerr_upper = [ci_upper_values[i] - y_values[i] for i in range(len(y_values))]
                    ax.errorbar(positions, y_values,
                               yerr=[yerr_lower, yerr_upper],
                               fmt='none', color='black', capsize=4, alpha=0.6)
            else:  # horizontal
                bars = ax.barh(positions, y_values, bar_width, label=group, color=color, alpha=0.8)

                if ci_lower_col and ci_upper_col and ci_lower_values and ci_upper_values:
                    xerr_lower = [y_values[i] - ci_lower_values[i] for i in range(len(y_values))]
                    xerr_upper = [ci_upper_values[i] - y_values[i] for i in range(len(y_values))]
                    ax.errorbar(y_values, positions,
                               xerr=[xerr_lower, xerr_upper],
                               fmt='none', color='black', capsize=4, alpha=0.6)

            # Add annotations if provided
            if annotation_col and annotation_col in df_group.columns:
                for i, x_cat in enumerate(x_categories):
                    df_cat = df_group[df_group[x_col] == x_cat]
                    if len(df_cat) > 0 and pd.notna(df_cat[annotation_col].values[0]):
                        annotation_text = str(df_cat[annotation_col].values[0])
                        if orientation == 'vertical':
                            ax.text(positions[i], y_values[i], annotation_text,
                                   ha='center', va='bottom', fontsize=9)
                        else:
                            ax.text(y_values[i], positions[i], annotation_text,
                                   ha='left', va='center', fontsize=9)

    else:
        # No grouping - simple bar plot
        y_values = [df[df[x_col] == x_cat][y_col].values[0] for x_cat in x_categories]

        if orientation == 'vertical':
            bars = ax.bar(x_positions, y_values, bar_width * 2, alpha=0.8)

            # Add error bars if provided
            if ci_lower_col and ci_upper_col:
                ci_lower_values = [df[df[x_col] == x_cat][ci_lower_col].values[0] for x_cat in x_categories]
                ci_upper_values = [df[df[x_col] == x_cat][ci_upper_col].values[0] for x_cat in x_categories]
                yerr_lower = [y_values[i] - ci_lower_values[i] for i in range(len(y_values))]
                yerr_upper = [ci_upper_values[i] - y_values[i] for i in range(len(y_values))]
                ax.errorbar(x_positions, y_values,
                           yerr=[yerr_lower, yerr_upper],
                           fmt='none', color='black', capsize=4, alpha=0.6)
        else:  # horizontal
            bars = ax.barh(x_positions, y_values, bar_width * 2, alpha=0.8)

            if ci_lower_col and ci_upper_col:
                ci_lower_values = [df[df[x_col] == x_cat][ci_lower_col].values[0] for x_cat in x_categories]
                ci_upper_values = [df[df[x_col] == x_cat][ci_upper_col].values[0] for x_cat in x_categories]
                xerr_lower = [y_values[i] - ci_lower_values[i] for i in range(len(y_values))]
                xerr_upper = [ci_upper_values[i] - y_values[i] for i in range(len(y_values))]
                ax.errorbar(y_values, x_positions,
                           xerr=[xerr_lower, xerr_upper],
                           fmt='none', color='black', capsize=4, alpha=0.6)

        # Add annotations if provided
        if annotation_col and annotation_col in df.columns:
            for i, x_cat in enumerate(x_categories):
                df_cat = df[df[x_col] == x_cat]
                if len(df_cat) > 0 and pd.notna(df_cat[annotation_col].values[0]):
                    annotation_text = str(df_cat[annotation_col].values[0])
                    if orientation == 'vertical':
                        ax.text(x_positions[i], y_values[i], annotation_text,
                               ha='center', va='bottom', fontsize=9)
                    else:
                        ax.text(y_values[i], x_positions[i], annotation_text,
                               ha='left', va='center', fontsize=9)

    # Set labels and formatting
    if orientation == 'vertical':
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_categories, rotation=0 if len(x_categories) <= 4 else 45, ha='right')
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontweight='bold')
    else:  # horizontal
        ax.set_yticks(x_positions)
        ax.set_yticklabels(x_categories)
        if ylabel:
            ax.set_xlabel(ylabel, fontweight='bold')
        if xlabel:
            ax.set_ylabel(xlabel, fontweight='bold')

    if title:
        ax.set_title(title, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, axis='y' if orientation == 'vertical' else 'x')

    if show_legend and group_col is not None:
        ax.legend(loc='best', framealpha=0.95)


def plot_piecewise_trajectory(
    theta_data: pd.DataFrame,
    prob_data: Optional[pd.DataFrame] = None,
    segment_col: str = 'Segment',
    paradigm_col: str = 'paradigm',
    time_col: str = 'Days_within',
    theta_obs_col: str = 'theta_observed',
    theta_pred_col: str = 'theta_predicted',
    prob_obs_col: str = 'prob_observed',
    prob_pred_col: str = 'prob_predicted',
    ci_lower_col: str = 'CI_lower',
    ci_upper_col: str = 'CI_upper',
    slope_col: str = 'slope',
    data_type_col: str = 'data_type',
    segment_order: List[str] = None,
    paradigm_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    output_path: Optional[Path] = None,
    title: str = 'Piecewise Trajectory by Segment',
    suptitle: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create two-panel piecewise trajectory plot (Early vs Late segments).

    Designed for piecewise LMM visualizations where forgetting trajectory
    is divided into temporal segments (e.g., Early consolidation window
    vs Late decay period). Each panel shows trajectories for multiple
    paradigms/factors with observed data points and model predictions.

    Args:
        theta_data: DataFrame with theta-scale plot data
            Required columns: segment_col, paradigm_col, time_col,
            theta_obs_col, theta_pred_col, ci_lower_col, ci_upper_col,
            slope_col, data_type_col
        prob_data: Optional DataFrame with probability-scale data
            If provided, creates a 2x2 grid (theta + probability scales)
            Required columns: same as theta_data but with prob_obs_col, prob_pred_col
        segment_col: Column name for segment variable (default 'Segment')
        paradigm_col: Column name for paradigm/factor grouping (default 'paradigm')
        time_col: Column name for time within segment (default 'Days_within')
        theta_obs_col: Column name for observed theta values (default 'theta_observed')
        theta_pred_col: Column name for predicted theta values (default 'theta_predicted')
        prob_obs_col: Column name for observed probability values (default 'prob_observed')
        prob_pred_col: Column name for predicted probability values (default 'prob_predicted')
        ci_lower_col: Column name for CI lower bound (default 'CI_lower')
        ci_upper_col: Column name for CI upper bound (default 'CI_upper')
        slope_col: Column name for segment-paradigm slopes (default 'slope')
        data_type_col: Column name for data type marker (default 'data_type')
        segment_order: List of segment names in display order (default ['Early', 'Late'])
        paradigm_colors: Dict mapping paradigm names to colors
            Default: {'IFR': '#E74C3C', 'ICR': '#3498DB', 'IRE': '#2ECC71'}
        figsize: Figure size (width, height) in inches
        output_path: Optional path to save plot
        title: Plot title (per panel)
        suptitle: Optional super title for entire figure

    Returns:
        fig: Matplotlib figure
        axes: Array of matplotlib axes (1x2 for theta only, 2x2 with prob_data)

    Example:
        >>> theta_df = pd.read_csv('step06_piecewise_theta_data.csv')
        >>> prob_df = pd.read_csv('step06_piecewise_probability_data.csv')
        >>> fig, axes = plot_piecewise_trajectory(theta_df, prob_df)
        >>> fig.savefig('piecewise_plot.png', dpi=300)
    """
    # Set defaults
    if segment_order is None:
        segment_order = ['Early', 'Late']

    if paradigm_colors is None:
        paradigm_colors = {
            'IFR': '#E74C3C',  # Red - Free Recall
            'ICR': '#3498DB',  # Blue - Cued Recall
            'IRE': '#2ECC71'   # Green - Recognition
        }

    # Determine layout
    if prob_data is not None:
        nrows, ncols = 2, 2
        figsize = (figsize[0], figsize[1] * 1.6)  # Taller for 2 rows
    else:
        nrows, ncols = 1, 2

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Get paradigms from data
    paradigms = theta_data[paradigm_col].dropna().unique()

    def plot_panel(ax, data, segment, obs_col, pred_col, ylabel, show_legend=True):
        """Plot single panel for one segment."""
        seg_data = data[data[segment_col] == segment]

        for paradigm in paradigms:
            color = paradigm_colors.get(paradigm, None)
            para_data = seg_data[seg_data[paradigm_col] == paradigm]

            # Plot predictions (smooth line)
            pred_data = para_data[para_data[data_type_col] == 'predicted']  # Note: 'predicted' not 'prediction'
            pred_sorted = None
            if len(pred_data) > 0:
                pred_sorted = pred_data.sort_values(time_col)
                ax.plot(pred_sorted[time_col], pred_sorted[pred_col],
                       color=color, linewidth=2, label=paradigm, alpha=0.9)

            # Plot observed data points with error bars
            obs_data = para_data[para_data[data_type_col] == 'observed']
            if len(obs_data) > 0:
                ax.errorbar(obs_data[time_col], obs_data[obs_col],
                           yerr=[obs_data[obs_col] - obs_data[ci_lower_col],
                                 obs_data[ci_upper_col] - obs_data[obs_col]],
                           fmt='o', color=color, markersize=8,
                           capsize=4, capthick=1.5, alpha=0.7)

            # Add slope annotation
            slope_val = para_data[slope_col].dropna().iloc[0] if len(para_data) > 0 and slope_col in para_data.columns else None
            if slope_val is not None and pred_sorted is not None:
                # Position annotation at end of line
                if len(pred_sorted) > 0:
                    x_pos = pred_sorted[time_col].iloc[-1]
                    y_pos = pred_sorted[pred_col].iloc[-1]
                    ax.annotate(f'{slope_val:.3f}/day',
                               xy=(x_pos, y_pos), xytext=(5, 0),
                               textcoords='offset points', fontsize=8,
                               color=color, alpha=0.8)

        ax.set_xlabel('Days within Segment', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{segment} Segment', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)

        if show_legend:
            ax.legend(loc='upper right', framealpha=0.95, fontsize=9)

    # Plot theta-scale panels (row 0)
    for col, segment in enumerate(segment_order):
        plot_panel(axes[0, col], theta_data, segment,
                  theta_obs_col, theta_pred_col, 'Theta (IRT Ability)',
                  show_legend=(col == 1))

    # Plot probability-scale panels (row 1) if provided
    if prob_data is not None:
        for col, segment in enumerate(segment_order):
            plot_panel(axes[1, col], prob_data, segment,
                      prob_obs_col, prob_pred_col, 'Probability',
                      show_legend=(col == 1))
            axes[1, col].set_ylim(0, 1)  # Probability scale 0-1

    # Add suptitle
    if suptitle:
        fig.suptitle(suptitle, fontweight='bold', fontsize=14, y=1.02)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

    return fig, axes
