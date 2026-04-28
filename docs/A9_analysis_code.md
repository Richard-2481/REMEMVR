---
layout: default
title: "A9. Analysis Code"
parent: Home
nav_order: 10
---

# A9. Analysis Code (Python)

All statistical analyses were conducted in Python using statsmodels (linear mixed models), custom IRT calibration, and standard scientific computing libraries (NumPy, SciPy, pandas). Each research question folder contains both the analysis scripts and the resulting figures.

## Code Organisation

Analysis code is organised by thesis chapter, with each research question (RQ) following a standardised 8-step pipeline. Each RQ folder contains a `code/` set of Python scripts and a `plots/` folder with generated figures.

### Data Processing

| File | Purpose |
|------|---------|
| [`data_processing/data.py`](https://github.com/Richard-2481/REMEMVR/tree/main/code/data_processing/data.py) | Master data loading and variable extraction |
| [`data_processing/column_mapping.py`](https://github.com/Richard-2481/REMEMVR/tree/main/code/data_processing/column_mapping.py) | Column standardisation and categorical encoding |

### Analysis Toolkit

Shared modules imported by the per-RQ analysis scripts.

[Browse tools/](https://github.com/Richard-2481/REMEMVR/tree/main/code/tools){: .btn }

| Module | Lines | Purpose |
|--------|-------|---------|
| `analysis_irt.py` | 732 | IRT calibration (2PL, GRM) and item purification |
| `analysis_lmm.py` | 2,211 | Linear mixed model fitting, diagnostics, and reporting |
| `analysis_ctt.py` | 751 | Classical test theory (reliability, item analysis) |
| `model_averaging.py` | 815 | Multi-model averaging with Akaike weights |
| `model_selection.py` | 847 | AIC/BIC model comparison and ranking |
| `plotting.py` | 1,347 | Trajectory, diagnostic, and comparison plots |
| `analysis_regression.py` | 582 | OLS/Ridge regression for individual differences |
| `analysis_lpa.py` | 473 | Latent profile analysis |
| `analysis_stats.py` | 588 | Statistical utilities (effect sizes, corrections) |
| `analysis_extensions.py` | 499 | Extended model families (piecewise, AR1) |
| `bootstrap.py` | 305 | Bootstrap confidence intervals and permutation tests |
| `variance_decomposition.py` | 1,022 | ICC, variance partitioning, random effects extraction |
| `validation.py` | 3,027 | Cross-validation, assumption checking, sensitivity |
| `clinical.py` | 363 | Clinical norms and percentile conversion |
| `sem_calibration.py` | 836 | Structural equation model calibration |
| `data.py` | 410 | Data loading and transformation utilities |
| `config.py` | 402 | Analysis configuration and path management |

### Support Scripts

| File | Purpose |
|------|---------|
| [`scripts/compute_norms.py`](https://github.com/Richard-2481/REMEMVR/tree/main/code/scripts/compute_norms.py) | Demographically-adjusted T-scores for RAVLT and RPM |
| [`scripts/run_loocv_all_models.py`](https://github.com/Richard-2481/REMEMVR/tree/main/code/scripts/run_loocv_all_models.py) | Leave-one-out cross-validation with Ridge regularisation |

### Chapter 4: Forgetting Trajectories (35 RQs, 133 figures)

[Browse Chapter 4 code and figures](https://github.com/Richard-2481/REMEMVR/tree/main/code/ch4_forgetting){: .btn }

| RQ Series | Topic | Key Analyses |
|-----------|-------|-------------|
| 4.1 | General trajectories | Power-law vs logarithmic model selection, ICC, age effects |
| 4.2 | Domain effects | What/Where/When trajectory comparisons |
| 4.3 | Paradigm effects | Free recall/Cued recall/Recognition trajectories |
| 4.4 | Schema effects | Congruent/Common/Incongruent analysis |
| 4.5 | Spatial effects | Source/Destination location trajectories |

### Chapter 5: Metacognition (39 RQs, 85 figures)

[Browse Chapter 5 code and figures](https://github.com/Richard-2481/REMEMVR/tree/main/code/ch5_metacognition){: .btn }

| RQ Series | Topic | Key Analyses |
|-----------|-------|-------------|
| 5.1 | General confidence | Confidence trajectories, ICC ratio |
| 5.2 | Calibration | Resolution, overconfidence, Brier decomposition |
| 5.3--5.5 | Domain/Paradigm/Schema confidence | Modality-specific metacognition |
| 5.6--5.7 | Hard-easy effect | Stability over time, Dunning-Kruger |
| 5.8 | Spatial confidence | Source-destination dissociation |
| 5.9 | Cross-trajectory | Confidence-accuracy comparisons |

### Chapter 6: Individual Differences (28 RQs, 102 figures)

[Browse Chapter 6 code and figures](https://github.com/Richard-2481/REMEMVR/tree/main/code/ch6_individual_differences){: .btn }

| RQ Series | Topic | Key Analyses |
|-----------|-------|-------------|
| 6.1 | Cognitive battery | RAVLT, BVMT, RPM, NART prediction |
| 6.2 | Age moderation | VR scaffolding hypothesis |
| 6.3 | Capacity dissociation | Accuracy vs confidence differential prediction |
| 6.4 | Process-specificity | Transfer appropriate processing |
| 6.5 | Self-report | Sleep, VR experience, education, DASS |
| 6.6 | Latent profiles | Visualisers vs verbalisers |
| 6.7 | Reverse inference | REMEMVR to traditional test prediction |

## Standard Analysis Pipeline

Each RQ follows an 8-step pipeline:

1. **step00:** Data extraction and stratification
2. **step01:** IRT calibration (Pass 1)
3. **step02:** Item purification
4. **step03:** IRT calibration (Pass 2, refined)
5. **step04:** Data reshaping for modelling
6. **step05:** Candidate model fitting (LMM with multiple functional forms)
7. **step06:** Model selection and post-hoc tests
8. **step07--08:** Visualisation, sensitivity analyses, cross-validation

## Statistical Framework

- **Estimation:** Maximum likelihood (ML, not REML)
- **Inference:** Wald z-tests (asymptotic normal, statsmodels MixedLM)
- **Reporting:** B, SE, z, p, 95% CI for all fixed effects
- **Model selection:** AIC-based with Akaike weights and model averaging
