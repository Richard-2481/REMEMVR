"""
IRT Analysis Tool for REMEMVR

Provides modular functions for Item Response Theory (IRT) calibration using deepirtools.

Core Functions:
- prepare_irt_data(): Convert long format to response matrix + Q-matrix
- configure_irt_model(): Build IWAVE GRM model
- fit_irt_model(): Fit the model to data
- extract_theta_scores(): Extract ability estimates
- extract_item_parameters(): Extract item discrimination and difficulty
- calibrate_irt(): Main pipeline function

Dependencies:
- deepirtools (IRT modeling)
- torch (neural network backend)
- pandas, numpy (data manipulation)

TDD Status: GREEN phase (making tests pass)
"""

import pandas as pd
import numpy as np
import torch
import deepirtools
from typing import Dict, List, Tuple, Optional, Union


def prepare_irt_input_from_long(
    df_long: pd.DataFrame,
    groups: Dict[str, List[str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Prepare IRT data from long format.

    Converts long-format dataframe to:
    1. Response matrix (wide format, stacked by Composite_ID)
    2. Q-matrix (item-to-factor assignments)
    3. Missing data mask
    4. Item list (ordered)
    5. Composite IDs (UID_T# format)

    Args:
        df_long: Long format dataframe with columns [UID, test, item_name, score]
        groups: Dictionary mapping factor names to domain code patterns
                Example: {'What': ['-N-'], 'Where': ['-U-', '-D-'], 'When': ['-O-']}

    Returns:
        Tuple of:
        - response_matrix: torch.Tensor of shape [n_observations, n_items]
        - Q_matrix: torch.Tensor of shape [n_items, n_factors]
        - missing_mask: torch.Tensor of shape [n_observations, n_items] (1=observed, 0=missing)
        - item_list: List of item names (ordered)
        - composite_ids: List of Composite_IDs (UID_T# format)

    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """

    # ─── Input Validation ────────────────────────────────────────────────────
    if df_long.empty:
        raise ValueError("Empty DataFrame provided to prepare_irt_data()")

    required_cols = ['UID', 'test', 'item_name', 'score']
    missing_cols = [col for col in required_cols if col not in df_long.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ─── Create Composite_ID ─────────────────────────────────────────────────
    df = df_long.copy()
    df['Composite_ID'] = df['UID'].astype(str) + '_T' + df['test'].astype(str)

    # ─── Pivot to Wide Format ────────────────────────────────────────────────
    response_df = df.pivot(
        index='Composite_ID',
        columns='item_name',
        values='score'
    )

    composite_ids = response_df.index.tolist()
    item_list = response_df.columns.tolist()
    n_items = len(item_list)
    n_factors = len(groups)

    # ─── Build Response Matrix and Missing Mask ──────────────────────────────
    # Convert to numpy, then to torch tensors
    response_np = response_df.values
    response_matrix = torch.from_numpy(np.nan_to_num(response_np, nan=0.0)).float()
    missing_mask = torch.from_numpy(~np.isnan(response_np)).float()

    # ─── Build Q-Matrix ──────────────────────────────────────────────────────
    # Q-matrix: [n_items, n_factors], binary (0 or 1)
    # Each item loads on exactly one factor (confirmatory model)

    Q_matrix = torch.zeros((n_items, n_factors), dtype=torch.float32)

    for item_idx, item_name in enumerate(item_list):
        for factor_idx, (factor_name, patterns) in enumerate(groups.items()):
            # Check if any pattern matches this item
            if any(pattern in item_name for pattern in patterns):
                Q_matrix[item_idx, factor_idx] = 1
                break  # Item can only load on one factor (confirmatory)

    # Validate that each item loads on exactly one factor
    items_per_factor = Q_matrix.sum(dim=1)
    if not torch.all(items_per_factor == 1):
        unassigned_items = [item_list[i] for i in range(n_items) if items_per_factor[i] != 1]
        raise ValueError(
            f"Some items don't match any factor pattern: {unassigned_items}\n"
            f"Groups provided: {groups}"
        )

    return response_matrix, Q_matrix, missing_mask, item_list, composite_ids


def configure_irt_model(
    n_items: int,
    n_factors: int,
    n_cats: List[int],
    Q_matrix: torch.Tensor,
    correlated_factors: Union[bool, List[int]],
    device: str = 'cpu',
    seed: int = 123
) -> deepirtools.IWAVE:
    """
    Configure deepirtools IWAVE model for Graded Response Model (GRM).

    Args:
        n_items: Number of items
        n_factors: Number of latent factors
        n_cats: List of category counts per item (e.g., [2, 2, 3, 4] for mixed)
        Q_matrix: Item-to-factor loading matrix [n_items, n_factors]
        correlated_factors: If True, correlate all factors. If list, specify which factors to correlate.
        device: 'cpu' or 'cuda' (GPU)
        seed: Random seed for reproducibility

    Returns:
        Configured (unfitted) IWAVE model

    Raises:
        ValueError: If device is invalid or Q_matrix shape doesn't match
    """

    # ─── Input Validation ────────────────────────────────────────────────────
    valid_devices = ['cpu', 'cuda']
    if device not in valid_devices and not device.startswith('cuda:'):
        raise ValueError(f"Invalid device '{device}'. Must be 'cpu' or 'cuda'.")

    if Q_matrix.shape != (n_items, n_factors):
        raise ValueError(
            f"Q_matrix shape {Q_matrix.shape} doesn't match expected ({n_items}, {n_factors})"
        )

    if len(n_cats) != n_items:
        raise ValueError(f"n_cats length ({len(n_cats)}) must equal n_items ({n_items})")

    # ─── Set Device ──────────────────────────────────────────────────────────
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU.")
        device = 'cpu'

    torch_device = torch.device(device)

    # ─── Determine Correlated Factors ────────────────────────────────────────
    if correlated_factors is True:
        correlated_factors_list = list(range(n_factors))
    elif correlated_factors is False:
        correlated_factors_list = []
    else:
        correlated_factors_list = correlated_factors

    # ─── Set Random Seed ─────────────────────────────────────────────────────
    deepirtools.manual_seed(seed)

    # ─── Instantiate Model ───────────────────────────────────────────────────
    model = deepirtools.IWAVE(
        model_type="grm",
        latent_size=n_factors,
        n_cats=n_cats,
        Q=Q_matrix,
        device=torch_device,
        correlated_factors=correlated_factors_list
    )

    return model


def fit_irt_grm(
    model: deepirtools.IWAVE,
    response_matrix: torch.Tensor,
    missing_mask: torch.Tensor,
    batch_size: int = 2048,
    iw_samples: int = 100,
    mc_samples: int = 1
) -> deepirtools.IWAVE:
    """
    Fit IWAVE model to response data.

    Args:
        model: Configured IWAVE model (from configure_irt_model)
        response_matrix: Response data [n_observations, n_items]
        missing_mask: Missing data mask [n_observations, n_items] (1=observed, 0=missing)
        batch_size: Batch size for gradient descent (default: 2048, validated "Med" level)
        iw_samples: Importance-weighted samples for ELBO (default: 100, validated "Med" level)
        mc_samples: Monte Carlo samples during fitting (default: 1, per thesis validation)

    Returns:
        Fitted IWAVE model

    Note:
        Model is fitted in-place. The function returns the model for convenience.

    Validated Settings (from thesis/analyses/ANALYSES_DEFINITIVE.md):
        - batch_size=2048, iw_samples=100, mc_samples=1 ("Med" iteration level)
        - These settings balance precision with reasonable runtime (~60 min for 100 items)
    """

    # ─── Move Data to Device ─────────────────────────────────────────────────
    device = model.device
    data = response_matrix.to(device, non_blocking=True)
    mask = missing_mask.to(device, non_blocking=True)

    # ─── Fit Model ───────────────────────────────────────────────────────────
    print(f"Fitting IRT model on {device}...")
    model.fit(
        data=data,
        missing_mask=mask,
        batch_size=batch_size,
        iw_samples=iw_samples,
        mc_samples=mc_samples
    )
    print("IRT model fitting complete.")

    # ─── Clear GPU Cache ─────────────────────────────────────────────────────
    if str(device) != 'cpu':
        torch.cuda.empty_cache()

    return model


def extract_theta_from_irt(
    model: deepirtools.IWAVE,
    response_matrix: torch.Tensor,
    missing_mask: torch.Tensor,
    composite_ids: List[str],
    factor_names: List[str],
    scoring_batch_size: int = 2048,
    mc_samples: int = 100,
    iw_samples: int = 100,
    invert_scale: bool = False
) -> pd.DataFrame:
    """
    Extract theta scores (ability estimates) from fitted IRT model.

    Args:
        model: Fitted IWAVE model
        response_matrix: Response data [n_observations, n_items]
        missing_mask: Missing data mask [n_observations, n_items]
        composite_ids: List of Composite_IDs (UID_T# format)
        factor_names: List of factor names (e.g., ['What', 'Where', 'When'])
        scoring_batch_size: Batch size for scoring (default: 2048, validated "Med" level)
        mc_samples: Monte Carlo samples for scoring (default: 100, validated "Med" level)
        iw_samples: Importance-weighted samples for scoring (default: 100, validated "Med" level)
        invert_scale: If True, multiply theta scores by -1 for interpretability

    Returns:
        DataFrame with columns: [UID, test, Theta_Factor1, Theta_Factor2, ...]

    Validated Settings (from thesis/analyses/ANALYSES_DEFINITIVE.md):
        - mc_samples=100, iw_samples=100 for scoring ("Med" iteration level)
        - Higher samples during scoring ensures accurate ability estimates
    """

    # ─── Move Data to Device ─────────────────────────────────────────────────
    device = model.device
    data = response_matrix.to(device, non_blocking=True)
    mask = missing_mask.to(device, non_blocking=True)

    # ─── Score All Observations in Batches ───────────────────────────────────
    N = data.size(0)
    all_thetas = []

    with torch.no_grad():
        for i in range(0, N, scoring_batch_size):
            data_batch = data[i:i + scoring_batch_size]
            mask_batch = mask[i:i + scoring_batch_size]

            theta_batch = model.scores(
                data=data_batch,
                missing_mask=mask_batch,
                mc_samples=mc_samples,
                iw_samples=iw_samples
            )

            all_thetas.append(theta_batch.cpu())

    thetas = torch.cat(all_thetas, dim=0).numpy()

    # ─── Clear GPU Cache ─────────────────────────────────────────────────────
    if str(device) != 'cpu':
        torch.cuda.empty_cache()

    # ─── Build DataFrame ─────────────────────────────────────────────────────
    theta_cols = [f"Theta_{name}" for name in factor_names]
    df_thetas = pd.DataFrame(thetas, columns=theta_cols, index=composite_ids)
    df_thetas.index.name = 'Composite_ID'
    df_thetas.reset_index(inplace=True)

    # ─── Split Composite_ID into UID and test ────────────────────────────────
    df_thetas[['UID', 'test']] = df_thetas['Composite_ID'].str.split(
        '_T', n=1, expand=True
    )
    df_thetas['test'] = pd.to_numeric(df_thetas['test'])

    # ─── Invert Scale if Requested ───────────────────────────────────────────
    if invert_scale:
        df_thetas[theta_cols] = df_thetas[theta_cols] * -1

    # ─── Reorder Columns ─────────────────────────────────────────────────────
    df_thetas = df_thetas[['UID', 'test'] + theta_cols]

    return df_thetas


def extract_parameters_from_irt(
    model: deepirtools.IWAVE,
    item_list: List[str],
    factor_names: List[str],
    n_cats: List[int]
) -> pd.DataFrame:
    """
    Extract item parameters (discrimination, difficulty) from fitted IRT model.

    Discrimination = L2 norm of factor loadings
    Difficulty = mean(-intercepts / discrimination)

    Args:
        model: Fitted IWAVE model
        item_list: List of item names
        factor_names: List of factor names
        n_cats: List of category counts per item

    Returns:
        DataFrame with columns: [item_name, Difficulty, Overall_Discrimination,
                                 Discrim_Factor1, Discrim_Factor2, ...]
        Note: item_name is returned as a regular column (not index)
    """

    with torch.no_grad():
        # ─── Extract Model Parameters ────────────────────────────────────────
        loadings = model.loadings.detach().cpu()  # [n_items, n_factors]
        intercepts = model.intercepts.detach().cpu()  # [n_items] or [n_items, max_thresholds]

        # ─── Calculate Discrimination ────────────────────────────────────────
        # Overall discrimination = L2 norm of loadings across factors
        overall_discrimination = torch.linalg.norm(loadings, dim=1).numpy()

        # ─── Calculate Difficulty ────────────────────────────────────────────
        # Difficulty = mean threshold difficulty = mean(-intercepts / discrimination)

        difficulties = []
        is_dichotomous_case = intercepts.dim() == 1

        for j, k in enumerate(n_cats):
            if k <= 1:
                # No meaningful difficulty for single-category items
                difficulties.append(float('nan'))
            else:
                if is_dichotomous_case:
                    # All dichotomous: intercepts is 1D tensor
                    alpha_j = intercepts[j]
                else:
                    # Polytomous: intercepts is 2D tensor [n_items, max_thresholds]
                    # Take first (k-1) intercepts for item j
                    alpha_j = intercepts[j, :(k - 1)]

                # Threshold difficulties = -alpha / a
                b_j = -alpha_j / overall_discrimination[j]
                difficulties.append(b_j.mean().item())

        difficulties = np.array(difficulties, dtype=float)

        # ─── Build DataFrame ─────────────────────────────────────────────────
        params_dict = {
            'Difficulty': difficulties,
            'Overall_Discrimination': overall_discrimination
        }

        # Add factor-specific discrimination
        for factor_idx, factor_name in enumerate(factor_names):
            params_dict[f'Discrim_{factor_name}'] = loadings[:, factor_idx].numpy()

        df_items = pd.DataFrame(params_dict, index=item_list)
        df_items.index.name = 'item_name'

        # Convert index to column for automation robustness
        # (prevents loss when saving with index=False)
        df_items.reset_index(inplace=True)

        return df_items


def calibrate_irt(
    df_long: pd.DataFrame,
    groups: Dict[str, List[str]],
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main IRT calibration pipeline.

    Performs full workflow:
    1. Prepare data (response matrix, Q-matrix)
    2. Determine category counts per item
    3. Configure IRT model
    4. Fit model
    5. Extract theta scores
    6. Extract item parameters

    Args:
        df_long: Long format dataframe with [UID, test, item_name, score]
        groups: Factor definitions (e.g., {'What': ['-N-'], 'Where': ['-U-']})
        config: Configuration dictionary with keys:
            - factors: List of factor names (must match groups keys)
            - correlated_factors: bool or list of indices
            - device: 'cpu' or 'cuda'
            - model_fit: dict with {batch_size, iw_samples, mc_samples}
            - model_scores: dict with {scoring_batch_size, mc_samples, iw_samples}
            - invert_scale: (optional) bool, default False
            - seed: (optional) int, default 123

    Returns:
        Tuple of (df_thetas, df_items):
        - df_thetas: DataFrame with [UID, test, Theta_Factor1, ...]
        - df_items: DataFrame with [item_name, Difficulty, Overall_Discrimination, ...]
        Note: Both DataFrames have standard column format (no meaningful indices)

    Example:
        ```python
        # Validated "Med" settings from thesis/analyses/ANALYSES_DEFINITIVE.md
        config = {
            'factors': ['What', 'Where', 'When'],
            'correlated_factors': True,
            'device': 'cpu',
            'seed': 42,
            'model_fit': {
                'batch_size': 2048,
                'iw_samples': 100,
                'mc_samples': 1
            },
            'model_scores': {
                'scoring_batch_size': 2048,
                'mc_samples': 100,
                'iw_samples': 100
            }
        }

        df_thetas, df_items = calibrate_irt(df_long, groups, config)
        ```

    Note:
        The validated "Med" settings (~60 min runtime) should be used for publication.
        Lower settings (iw_samples=5, mc_samples=1) run faster but produce less
        precise estimates.
    """

    print("\n" + "=" * 60)
    print("IRT CALIBRATION PIPELINE")
    print("=" * 60)

    # ─── Step 1: Prepare Data ────────────────────────────────────────────────
    print("\nStep 1: Preparing IRT data...")
    response_matrix, Q_matrix, missing_mask, item_list, composite_ids = prepare_irt_input_from_long(
        df_long, groups
    )

    n_items = len(item_list)
    n_factors = len(groups)
    n_observations = len(composite_ids)

    print(f"  Observations: {n_observations}")
    print(f"  Items: {n_items}")
    print(f"  Factors: {n_factors}")
    print(f"  Missing data: {(~missing_mask.bool()).sum().item()} cells ({(~missing_mask.bool()).sum().item() / (n_observations * n_items) * 100:.2f}%)")

    # ─── Step 2: Determine Category Counts ───────────────────────────────────
    print("\nStep 2: Determining category counts...")

    # Detect unique values per item (excluding NaN)
    n_cats = []
    for item_idx, item_name in enumerate(item_list):
        # Get all scores for this item (excluding missing)
        item_scores = df_long[df_long['item_name'] == item_name]['score']
        unique_scores = item_scores.dropna().unique()
        n_categories = len(unique_scores)

        # Need at least 2 categories for IRT
        if n_categories < 2:
            print(f"  Warning: Item '{item_name}' has only {n_categories} unique values")
            n_categories = 2  # Minimum for IRT

        n_cats.append(n_categories)

    print(f"  Category counts: {set(n_cats)}")

    # ─── Step 3: Configure Model ─────────────────────────────────────────────
    print("\nStep 3: Configuring IRT model...")

    model = configure_irt_model(
        n_items=n_items,
        n_factors=n_factors,
        n_cats=n_cats,
        Q_matrix=Q_matrix,
        correlated_factors=config['correlated_factors'],
        device=config.get('device', 'cpu'),
        seed=config.get('seed', 123)
    )

    print(f"  Model type: GRM (Graded Response Model)")
    print(f"  Device: {model.device}")
    print(f"  Correlated factors: {config['correlated_factors']}")

    # ─── Step 4: Fit Model ───────────────────────────────────────────────────
    print("\nStep 4: Fitting IRT model...")

    model = fit_irt_grm(
        model=model,
        response_matrix=response_matrix,
        missing_mask=missing_mask,
        **config['model_fit']
    )

    # ─── Step 5: Extract Theta Scores ────────────────────────────────────────
    print("\nStep 5: Extracting theta scores...")

    df_thetas = extract_theta_from_irt(
        model=model,
        response_matrix=response_matrix,
        missing_mask=missing_mask,
        composite_ids=composite_ids,
        factor_names=config['factors'],
        invert_scale=config.get('invert_scale', False),
        **config['model_scores']
    )

    print(f"  Theta scores extracted: {len(df_thetas)} observations")

    # ─── Step 6: Extract Item Parameters ─────────────────────────────────────
    print("\nStep 6: Extracting item parameters...")

    df_items = extract_parameters_from_irt(
        model=model,
        item_list=item_list,
        factor_names=config['factors'],
        n_cats=n_cats
    )

    print(f"  Item parameters extracted: {len(df_items)} items")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IRT CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Theta scores shape: {df_thetas.shape}")
    print(f"Item parameters shape: {df_items.shape}")
    print()

    return df_thetas, df_items


def filter_items_by_quality(
    df_items: pd.DataFrame,
    a_threshold: float = 0.4,
    b_threshold: float = 3.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purify IRT items using Decision D039 thresholds.

    Implements 2-pass IRT methodology: Exclude items where:
    - Discrimination (a) < 0.4 on ANY factor
    - |Difficulty (b)| > 3.0 on ANY factor

    Args:
        df_items: Item parameters from IRT calibration (Pass 1)
                  Must contain columns: item_name, factor, a, b
        a_threshold: Minimum discrimination threshold (default: 0.4)
        b_threshold: Maximum absolute difficulty threshold (default: 3.0)

    Returns:
        Tuple of:
        - df_purified: Items that passed purification criteria
        - df_excluded: Items that failed, with exclusion_reason column

    Example:
        ```python
        df_thetas_pass1, df_items_pass1 = calibrate_irt(df_long, groups, config)
        df_items_purified, df_items_excluded = purify_items(df_items_pass1)

        # Re-calibrate with purified items
        df_thetas_pass2, df_items_pass2 = calibrate_irt(df_long_purified, groups, config)
        ```

    Decision D039 Context:
        Temporal items in REMEMVR are inherently difficult (expected 40-50% retention).
        Without purification, difficult items (|b|>3.0) and low-discrimination items (a<0.4)
        introduce measurement error. 2-pass purification ensures only psychometrically
        sound items contribute to ability estimates.
    """

    print("\n" + "=" * 60)
    print("ITEM PURIFICATION (Decision D039)")
    print("=" * 60)
    print(f"Thresholds: a >= {a_threshold}, |b| <= {b_threshold}")

    # Detect format: univariate (simple) or multivariate (from calibrate_irt)
    is_multivariate = 'Difficulty' in df_items.columns

    if is_multivariate:
        print("Format: Multivariate IRT (from calibrate_irt)")
        # Map column names: Difficulty → b
        # For discrimination, find which Discrim_* column is non-zero for each item
        df_normalized = df_items.copy()
        df_normalized['b'] = df_normalized['Difficulty']

        # Find primary dimension and discrimination for each item
        discrim_cols = [col for col in df_normalized.columns if col.startswith('Discrim_')]

        factors = []
        discriminations = []

        for _, row in df_normalized.iterrows():
            # Find which dimension has non-zero discrimination
            discrim_values = {col.replace('Discrim_', ''): row[col] for col in discrim_cols}

            # Get the dimension with max discrimination (handles floating point near-zero)
            primary_dim = max(discrim_values.items(), key=lambda x: x[1])

            factors.append(primary_dim[0])
            discriminations.append(primary_dim[1])

        df_normalized['factor'] = factors
        df_normalized['a'] = discriminations
    else:
        print("Format: Univariate IRT (simple format)")
        # Already has 'factor', 'a', 'b' columns
        df_normalized = df_items.copy()

    # Identify items to exclude
    excluded_items = []

    for _, row in df_normalized.iterrows():
        item_name = row['item_name']
        factor = row['factor']
        a = row['a']
        b = row['b']

        # Check discrimination threshold
        if a < a_threshold:
            excluded_items.append({
                'item_name': item_name,
                'factor': factor,
                'a': a,
                'b': b,
                'exclusion_reason': f'a < {a_threshold} on {factor}'
            })
            continue

        # Check difficulty threshold (absolute value)
        if abs(b) > b_threshold:
            excluded_items.append({
                'item_name': item_name,
                'factor': factor,
                'a': a,
                'b': b,
                'exclusion_reason': f'|b| > {b_threshold} on {factor} (b={b:.2f})'
            })
            continue

    # Create excluded DataFrame
    if excluded_items:
        df_excluded = pd.DataFrame(excluded_items)
    else:
        df_excluded = pd.DataFrame(columns=['item_name', 'factor', 'a', 'b', 'exclusion_reason'])

    # Get purified items (those not in excluded list)
    excluded_names = df_excluded['item_name'].unique()
    df_purified = df_items[~df_items['item_name'].isin(excluded_names)].copy()

    # Summary
    n_total = len(df_items['item_name'].unique())
    n_excluded = len(excluded_names)
    n_purified = n_total - n_excluded

    print(f"\nTotal items: {n_total}")
    print(f"Excluded: {n_excluded} ({n_excluded/n_total*100:.1f}%)")
    print(f"Purified: {n_purified} ({n_purified/n_total*100:.1f}%)")

    if n_excluded > 0:
        print("\nExcluded items:")
        for _, row in df_excluded.iterrows():
            print(f"  - {row['item_name']}: {row['exclusion_reason']}")

    print("=" * 60 + "\n")

    return df_purified, df_excluded


def calibrate_grm(df_long: pd.DataFrame, groups: Dict[str, List[str]], config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper for calibrate_irt() for config.yaml compatibility.

    This function provides an alias to calibrate_irt() with the same signature.
    Exists for backwards compatibility with config.yaml files that specify
    'calibrate_grm' as the IRT calibration function.

    Args:
        df_long: Long format dataframe with columns [UID, test, item_name, score]
        groups: Dictionary mapping factor names to domain code patterns
        config: Configuration dictionary with IRT parameters

    Returns:
        Same as calibrate_irt():
        - df_thetas: Theta scores (ability estimates)
        - df_items: Item parameters (discrimination and difficulty)

    Example:
        ```python
        # Both calls are equivalent:
        df_thetas, df_items = calibrate_grm(df_long, groups, config)
        df_thetas, df_items = calibrate_irt(df_long, groups, config)
        ```
    """
    return calibrate_irt(df_long, groups, config)
