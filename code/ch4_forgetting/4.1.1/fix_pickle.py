#!/usr/bin/env python3
"""
Fix pickle issue by combining individual model pickle files into expected format.
"""
import pickle
from pathlib import Path

# RQ root
RQ_ROOT = Path(__file__).parent.parent

# Load individual model pickles
models = {}
model_names = ['Linear', 'Quadratic', 'Logarithmic', 'LinLog', 'QuadLog']
model_files = {
    'Linear': 'lmm_Linear.pkl',
    'Quadratic': 'lmm_Quadratic.pkl',
    'Logarithmic': 'lmm_Log.pkl',
    'LinLog': 'lmm_Lin+Log.pkl',
    'QuadLog': 'lmm_Quad+Log.pkl'
}

print("Loading individual model pickle files...")
for name, filename in model_files.items():
    filepath = RQ_ROOT / 'data' / filename
    if filepath.exists():
        with open(filepath, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"  ✓ Loaded {name} from {filename}")
    else:
        print(f"  ✗ Missing {filename}")

# Save combined pickle
output_path = RQ_ROOT / 'data' / 'step05_model_fits.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(models, f)

print(f"\n✓ Created combined pickle: {output_path}")
print(f"  Keys: {list(models.keys())}")
