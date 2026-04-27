#!/usr/bin/env python3
"""
Quick script to update RQ 5.4.3 from Log to Recip+Log model
Updates step02_fit_lmm.py to use recip_TSVR instead of TSVR_hours
"""

import re
from pathlib import Path

RQ_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = RQ_DIR / "code"

# Update step02_fit_lmm.py
step02_file = CODE_DIR / "step02_fit_lmm.py"
content = step02_file.read_text()

# Replace docstring
content = re.sub(
    r'PURPOSE:\s+Fit Linear Mixed Model.*?Random: ~log_TSVR.*?\n\n',
    '''PURPOSE:
    Fit Linear Mixed Model with 3-way Age x Congruence x Time interaction.
    Random intercepts and slopes for recip_TSVR by participant.
    UPDATED: RQ 5.4.1 ROOT established Recip+Log as two-process forgetting model
    (rapid 1/(t+1) + slow log(t+1)). Random slope on recip_TSVR per ROOT spec.

INPUTS:
    - data/step01_lmm_input.csv (1200 rows, long format)

OUTPUTS:
    - data/step02_lmm_model.pkl (pickled model object)
    - data/step02_lmm_model_summary.txt (model summary text)
    - data/step02_fixed_effects.csv (24 fixed effects terms, including recip_TSVR)

MODEL FORMULA:
    theta ~ recip_TSVR + log_TSVR + Age_c + C(congruence, Treatment('Common')) +
            Age_c:recip_TSVR + Age_c:log_TSVR +
            C(congruence):recip_TSVR + C(congruence):log_TSVR +
            Age_c:C(congruence) +
            Age_c:C(congruence):recip_TSVR + Age_c:C(congruence):log_TSVR

    Random: ~recip_TSVR | UID (random intercepts and slopes per RQ 5.4.1 ROOT)

''',
    content,
    flags=re.DOTALL
)

# Replace formula definition
content = re.sub(
    r'theta ~ 1 \+ TSVR_hours \+ log_TSVR',
    'theta ~ 1 + recip_TSVR + log_TSVR',
    content
)

# Replace all TSVR_hours interactions
content = content.replace('Age_c:TSVR_hours', 'Age_c:recip_TSVR')
content = content.replace('Congruent:TSVR_hours', 'Congruent:recip_TSVR')
content = content.replace('Incongruent:TSVR_hours', 'Incongruent:recip_TSVR')
content = content.replace('Age_c:Congruent:TSVR_hours', 'Age_c:Congruent:recip_TSVR')
content = content.replace('Age_c:Incongruent:TSVR_hours', 'Age_c:Incongruent:recip_TSVR')

# Replace re_formula
content = content.replace("re_formula='~log_TSVR'  # CORRECTED: log_TSVR not TSVR_hours (per 5.4.1)",
                         "re_formula='~recip_TSVR'  # UPDATED: recip_TSVR for two-process forgetting (per 5.4.1 ROOT)")
content = content.replace('random_structure = "random intercepts + slopes for log_TSVR (per 5.4.1)"',
                         'random_structure = "random intercepts + slopes for recip_TSVR (per 5.4.1 ROOT)"')
content = content.replace('[SUCCESS] Model fitted with random slopes on log_TSVR',
                         '[SUCCESS] Model fitted with random slopes on recip_TSVR (two-process forgetting)')

# Replace log comments
content = re.sub(
    r'\[INFO\] Random effects:.*?\n',
    '[INFO] Random effects: ~recip_TSVR | UID (per RQ 5.4.1 ROOT Recip+Log model)\\n[INFO] Two-process forgetting: rapid 1/(t+1) + slow log(t+1)\\n',
    content
)

# Replace the fixed effects logging section
content = re.sub(
    r'\[INFO\] Model formula \(fixed effects\):.*?\+\s+Age_c:Incongruent:log_TSVR',
    '[INFO] Model formula (fixed effects - Recip+Log two-process):\n  theta ~ 1 + recip_TSVR + log_TSVR + Age_c + Congruent + Incongruent\n        + Age_c:recip_TSVR + Age_c:log_TSVR\n        + Congruent:recip_TSVR + Congruent:log_TSVR\n        + Incongruent:recip_TSVR + Incongruent:log_TSVR\n        + Age_c:Congruent + Age_c:Incongruent\n        + Age_c:Congruent:recip_TSVR + Age_c:Congruent:log_TSVR\n        + Age_c:Incongruent:recip_TSVR + Age_c:Incongruent:log_TSVR',
    content,
    flags=re.DOTALL
)

# Replace summary section
content = re.sub(
    r'Fixed Effects: \d+ terms\n.*?- Main effects: Intercept, TSVR_hours',
    'Fixed Effects: {len(fixed_effects)} terms\n  - Main effects: Intercept, recip_TSVR',
    content
)

step02_file.write_text(content)
print(f"✓ Updated {step02_file}")

print("\nAll files updated successfully!")
print("\nNext steps:")
print("1. Run: poetry run python results/ch5/5.4.3/code/step01_prepare_lmm_input.py")
print("2. Run: poetry run python results/ch5/5.4.3/code/step02_fit_lmm.py")
print("3. Run remaining steps 03-05")
