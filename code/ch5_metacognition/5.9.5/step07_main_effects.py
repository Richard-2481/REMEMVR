#!/usr/bin/env python3
"""Step 7: Extract main effects - separate models by measure"""
import sys
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_main_effects.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

try:
    log("Step 7: Main Effects")

    # Load data
    df = pd.read_csv(RQ_DIR / "data" / "step03_merged_data_long.csv")

    # Extract main effects from Step 3 mixed model
    df_step3 = pd.read_csv(RQ_DIR / "data" / "step03_mixed_model_results.csv")

    # Get main effects (not interactions)
    main_effects = df_step3[~df_step3['term'].str.contains(':')].copy()
    main_effects['effect'] = main_effects['term']
    main_effects['interpretation'] = 'Main effect'

    # Fit separate models by measure using direct statsmodels
    df_acc = df[df['measure'] == 'accuracy']
    df_conf = df[df['measure'] == 'confidence']

    log("Accuracy-only model")
    acc_model = smf.mixedlm(
        formula='theta ~ location + TSVR_hours',
        data=df_acc,
        groups=df_acc['UID'],
        re_formula='~1'
    ).fit(reml=False)

    log("Confidence-only model")
    conf_model = smf.mixedlm(
        formula='theta ~ location + TSVR_hours',
        data=df_conf,
        groups=df_conf['UID'],
        re_formula='~1'
    ).fit(reml=False)
    
    # Extract location effects (handle both DataFrame and SimpleTable formats)
    acc_table = acc_model.summary().tables[1]
    if isinstance(acc_table, pd.DataFrame):
        acc_fe = acc_table.reset_index()
    else:
        acc_fe = pd.DataFrame(acc_table.data[1:], columns=acc_table.data[0])

    conf_table = conf_model.summary().tables[1]
    if isinstance(conf_table, pd.DataFrame):
        conf_fe = conf_table.reset_index()
    else:
        conf_fe = pd.DataFrame(conf_table.data[1:], columns=conf_table.data[0])
    
    # Save
    output_path = RQ_DIR / "data" / "step07_main_effects_results.csv"
    main_effects[['effect', 'beta', 'se', 't_stat', 'p_uncorrected', 'p_bonferroni', 'interpretation']].to_csv(
        output_path, index=False, encoding='utf-8'
    )
    log(f"{output_path.name}")
    
    log("Step 7 complete")
    sys.exit(0)
except Exception as e:
    log(f"{str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
