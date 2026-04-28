#!/usr/bin/env python3
"""Define Nested Models: Specify 4 nested regression models with increasing complexity. Models range from"""

import sys
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_define_nested_models.log"
OUTPUT_DIR = RQ_DIR / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define 4 nested models
NESTED_MODELS = [
    {
        "model_name": "Minimal",
        "num_predictors": 2,
        "predictor_list": ["Age", "RAVLT_T"],
        "rationale": "Core age and verbal memory model"
    },
    {
        "model_name": "Core",
        "num_predictors": 3,
        "predictor_list": ["Age", "RAVLT_T", "BVMT_T"],
        "rationale": "Add visuospatial memory to core model"
    },
    {
        "model_name": "Extended",
        "num_predictors": 5,
        "predictor_list": ["Age", "RAVLT_T", "BVMT_T", "RPM_T", "Education"],
        "rationale": "Add fluid intelligence and education"
    },
    {
        "model_name": "Full",
        "num_predictors": 8,
        "predictor_list": ["Age", "RAVLT_T", "BVMT_T", "RPM_T", "Education", "DASS_Total", "Sleep", "Sex"],
        "rationale": "Complete model with mood, sleep, and demographic factors (NART not available)"
    },
    {
        "model_name": "Full+Retention",
        "num_predictors": 10,
        "predictor_list": ["Age", "RAVLT_T", "BVMT_T", "RPM_T", "Education", "DASS_Total", "Sleep", "Sex", "RAVLT_Pct_Ret_T", "BVMT_Pct_Ret_T"],
        "rationale": "Add percent retention predictors to test incremental validity of forgetting rate measures"
    }
]

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Define Nested Models")
        # Load Analysis Dataset
        log("Loading analysis dataset...")

        analysis_file = RQ_DIR / 'data' / 'step02_analysis_input.csv'
        df_analysis = pd.read_csv(analysis_file)

        log(f"Analysis dataset: {len(df_analysis)} rows, {len(df_analysis.columns)} columns")
        log(f"Available columns: {df_analysis.columns.tolist()}")
        # Validate Predictors Exist
        log("Checking all predictors exist in dataset...")

        available_predictors = df_analysis.columns.tolist()
        all_valid = True

        for model in NESTED_MODELS:
            model_name = model['model_name']
            predictor_list = model['predictor_list']

            log(f"Model '{model_name}': {len(predictor_list)} predictors")

            # Check each predictor exists
            missing_predictors = [p for p in predictor_list if p not in available_predictors]

            if missing_predictors:
                log(f"Model '{model_name}' references missing predictors: {missing_predictors}")
                all_valid = False
            else:
                log(f"All predictors available for '{model_name}'")

        if not all_valid:
            raise ValueError("One or more models reference missing predictors")
        # Validate Nesting Structure
        log("Checking models are properly nested...")

        # Check that each model is subset of next model
        for i in range(len(NESTED_MODELS) - 1):
            current_model = NESTED_MODELS[i]
            next_model = NESTED_MODELS[i + 1]

            current_predictors = set(current_model['predictor_list'])
            next_predictors = set(next_model['predictor_list'])

            if not current_predictors.issubset(next_predictors):
                extra_predictors = current_predictors - next_predictors
                log(f"Nesting violation: '{current_model['model_name']}' has predictors not in '{next_model['model_name']}': {extra_predictors}")
                all_valid = False
            else:
                log(f"'{current_model['model_name']}' properly nested in '{next_model['model_name']}'")

        if not all_valid:
            raise ValueError("Models are not properly nested")
        # Check for Duplicate Predictors
        log("Checking for duplicate predictors within models...")

        for model in NESTED_MODELS:
            model_name = model['model_name']
            predictor_list = model['predictor_list']

            if len(predictor_list) != len(set(predictor_list)):
                duplicates = [p for p in predictor_list if predictor_list.count(p) > 1]
                log(f"Model '{model_name}' has duplicate predictors: {duplicates}")
                all_valid = False
            else:
                log(f"No duplicate predictors in '{model_name}'")

        if not all_valid:
            raise ValueError("One or more models have duplicate predictors")
        # Save Model Specifications (CSV)
        log("Saving model specifications...")

        # Convert to DataFrame (predictor_list as string for CSV)
        models_data = []
        for model in NESTED_MODELS:
            models_data.append({
                'model_name': model['model_name'],
                'num_predictors': model['num_predictors'],
                'predictor_list': ', '.join(model['predictor_list']),
                'rationale': model['rationale']
            })

        df_models = pd.DataFrame(models_data)

        models_file = OUTPUT_DIR / "step03_nested_models.csv"
        df_models.to_csv(models_file, index=False, encoding='utf-8')
        log(f"Model specifications: {models_file} ({len(df_models)} models)")
        # Save Detailed Specifications (TXT)
        specs_file = OUTPUT_DIR / "step03_model_specs.txt"
        log(f"Saving detailed specifications to {specs_file}")

        with open(specs_file, 'w', encoding='utf-8') as f:
            f.write("Nested Regression Model Specifications\n")
            f.write("=" * 70 + "\n\n")
            f.write("RQ: 7.8.3 - Parsimonious Predictive Model with Cross-Validation\n")

            for i, model in enumerate(NESTED_MODELS, 1):
                f.write(f"Model {i}: {model['model_name']}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of Predictors: {model['num_predictors']}\n")
                f.write(f"Predictors: {', '.join(model['predictor_list'])}\n")
                f.write(f"Rationale: {model['rationale']}\n")
                f.write("\n")

            f.write("Nesting Structure:\n")
            f.write("-" * 70 + "\n")
            f.write("Minimal ⊂ Core ⊂ Extended ⊂ Full ⊂ Full+Retention\n")
            f.write("\n")
            f.write("Note: Full model reduced from 9 to 8 predictors (NART not available)\n")
            f.write("Note: Full+Retention adds percent retention T-scores for RAVLT and BVMT\n")

        log(f"Detailed specifications: {specs_file}")
        # Validation Check
        log("Validating output structure...")

        # Validate CSV structure
        expected_columns = ['model_name', 'num_predictors', 'predictor_list', 'rationale']
        validation_result = validate_dataframe_structure(
            df=df_models,
            expected_columns=expected_columns,
            expected_rows=5
        )

        if validation_result.get('valid', False):
            log(f"Output structure validated")
        else:
            log(f"Output structure validation failed: {validation_result.get('message', '')}")
            raise ValueError("Output validation failed")

        # Print summary
        log("Model specifications:")
        for model in NESTED_MODELS:
            log(f"  {model['model_name']:10s}: {model['num_predictors']} predictors")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
