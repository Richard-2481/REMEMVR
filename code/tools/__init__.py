"""
REMEMVR - Statistical Analysis Tools Package

A modular toolkit for analyzing episodic memory data from the REMEMVR study.

Modules:
- analysis_irt: Item Response Theory analysis (GRM models) - requires torch
- analysis_lmm: Linear Mixed Models for trajectory analysis
- config: Configuration management (paths, plotting, models)
- plotting: Generic plotting functions (trajectories, diagnostics, histograms)
- validation: Statistical validation and lineage tracking

Usage:
    # Import specific modules as needed
    from tools.analysis_irt import calibrate_irt  # Requires torch
    from tools.analysis_lmm import fit_lmm_trajectory
    from tools.validation import validate_lmm_residuals
    from tools.plotting import convert_theta_to_probability

Development:
    pytest tests/ -v

Version: 2.0.0 (Refactored 2025-11-22 - v4.X naming conventions)
"""

__version__ = "2.0.0"
__author__ = "REMEMVR Project"

# Lazy imports - only import when accessed to avoid torch dependency at package load
# Use direct module imports: from tools.analysis_irt import function_name

__all__ = [
    # IRT Analysis (requires torch)
    'analysis_irt',
    # LMM Analysis
    'analysis_lmm',
    # Config Management
    'config',
    # Plotting
    'plotting',
    # Validation
    'validation',
]
