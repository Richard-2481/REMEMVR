#!/usr/bin/env python3
"""
Verification utilities for rq_analysis agent v5.0.0
Ensures 4_analysis.yaml contains only verified, correct information
"""

import ast
import inspect
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ToolVerifier:
    """Verifies that functions specified in 3_tools.yaml actually exist"""
    
    def __init__(self):
        self.tool_map = {}
        self.validator_map = {}
        self.build_verification_maps()
    
    def build_verification_maps(self):
        """Scan tools directory to build function inventory"""
        tools_dir = project_root / "tools"
        
        for py_file in tools_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            module_name = f"tools.{py_file.stem}"
            
            # Parse Python file to extract function definitions
            with open(py_file) as f:
                try:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_key = f"{module_name}.{node.name}"
                            self.tool_map[func_key] = {
                                'module': module_name,
                                'function': node.name,
                                'file': str(py_file),
                                'signature': self._extract_signature(node)
                            }
                            
                            # Track validators separately
                            if node.name.startswith("validate"):
                                self.validator_map[node.name] = func_key
                                
                except SyntaxError as e:
                    print(f"Warning: Could not parse {py_file}: {e}")
    
    def _extract_signature(self, func_node: ast.FunctionDef) -> str:
        """Extract function signature from AST node"""
        args = []
        for arg in func_node.args.args:
            arg_str = arg.arg
            # Try to get type annotation
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Add defaults (simplified)
        defaults = func_node.args.defaults
        if defaults:
            for i, default in enumerate(defaults, start=len(args) - len(defaults)):
                if i < len(args):
                    args[i] += f" = {ast.unparse(default)}"
        
        # Get return type
        return_str = ""
        if func_node.returns:
            return_str = f" -> {ast.unparse(func_node.returns)}"
        
        return f"{func_node.name}({', '.join(args)}){return_str}"
    
    def verify_function(self, module: str, function: str) -> Dict[str, Any]:
        """Verify if a function exists and get its details"""
        func_key = f"{module}.{function}"
        
        if func_key in self.tool_map:
            return {
                'exists': True,
                'details': self.tool_map[func_key]
            }
        
        # Try to find similar functions
        similar = self.find_similar_functions(module, function)
        return {
            'exists': False,
            'similar': similar
        }
    
    def find_similar_functions(self, module: str, function: str) -> List[str]:
        """Find functions with similar names"""
        similar = []
        
        # Check for functions in same module
        for key in self.tool_map:
            if key.startswith(module):
                func_name = key.split('.')[-1]
                # Simple similarity: shared prefix or contains substring
                if (func_name.startswith(function[:4]) or 
                    function[:4] in func_name or
                    func_name in function):
                    similar.append(key)
        
        return similar[:5]  # Return top 5 matches
    
    def find_best_validator(self, step_type: str, module: str = None) -> str:
        """Find most specific validator for a step"""
        # Priority order
        candidates = [
            f"validate_{step_type}",
            f"validate_{step_type.replace('_', '')}",
            f"check_{step_type}",
        ]
        
        if module:
            module_type = module.split('.')[-1]
            candidates.extend([
                f"validate_{module_type}",
                f"validate_{module_type}_output"
            ])
        
        # Add generic fallbacks
        candidates.extend([
            "validate_data_columns",
            "validate_data",
            "check_data"
        ])
        
        for candidate in candidates:
            if candidate in self.validator_map:
                return self.validator_map[candidate]
        
        # Return most generic validator
        return "tools.validation.validate_data_columns"


class PathVerifier:
    """Verifies and corrects file paths"""
    
    def __init__(self, rq_id: str):
        self.rq_id = rq_id
        self.project_root = project_root
        self.path_corrections = {
            "data/cache/master.xlsx": "data/dfnonvr.csv",
            "master.xlsx": "data/dfnonvr.csv",
            "data/cache/dfData.csv": "data/dfdata.csv",
            "data/master.xlsx": "data/dfnonvr.csv",
        }
    
    def verify_input_path(self, path: str, step_number: int) -> Tuple[str, bool, str]:
        """
        Verify input path exists or correct it
        Returns: (corrected_path, exists, note)
        """
        # Try direct path
        full_path = self.project_root / path
        if full_path.exists():
            return path, True, "verified"
        
        # Try corrections
        if path in self.path_corrections:
            corrected = self.path_corrections[path]
            full_corrected = self.project_root / corrected
            if full_corrected.exists():
                return corrected, True, f"corrected from {path}"
        
        # Check if it's a derived file from previous step
        if "step" in path and step_number > 0:
            # Extract step number from path
            match = re.search(r'step(\d+)', path)
            if match:
                ref_step = int(match.group(1))
                if ref_step < step_number:
                    # Valid reference to previous step output
                    return path, False, "derived from previous step"
        
        # Check for chapter cross-references
        if "results/ch" in path:
            # This is output from another RQ
            return path, False, "cross-RQ dependency"
        
        return path, False, "not found"
    
    def fix_output_path(self, path: str) -> str:
        """Convert flat paths to hierarchical structure"""
        # Already hierarchical?
        if f"results/{self.rq_id}" in path:
            return path
        
        # Fix flat paths
        if path.startswith("data/"):
            filename = path[5:]  # Remove "data/"
            return f"results/{self.rq_id}/data/{filename}"
        elif path.startswith("plots/"):
            filename = path[6:]  # Remove "plots/"
            return f"results/{self.rq_id}/plots/{filename}"
        elif path.startswith("logs/"):
            filename = path[5:]  # Remove "logs/"
            return f"results/{self.rq_id}/logs/{filename}"
        elif path.startswith("results/") and self.rq_id not in path:
            # Wrong results folder
            filename = path.split('/')[-1]
            return f"results/{self.rq_id}/results/{filename}"
        
        return path


class OperationSpecifier:
    """Makes vague operations specific"""
    
    def __init__(self):
        self.operation_map = {
            # Data loading
            "Load CSV": "pd.read_csv('{path}')",
            "Load Excel": "pd.read_excel('{path}')",
            "Save CSV": "df.to_csv('{path}', index=False)",
            
            # Data manipulation
            "Create composite_ID": "df['composite_ID'] = df['UID'] + '_' + df['test'].astype(str)",
            "Create composite ID": "df['composite_ID'] = df['UID'] + '_' + df['test'].astype(str)",
            "Dichotomize": "df[cols] = (df[cols] >= {threshold}).astype(int)",
            "Dichotomize values": "df[cols] = (df[cols] >= {threshold}).astype(int)",
            "Filter columns": "df = df[{columns}]",
            "Merge datasets": "df = pd.merge(df1, df2, on='{key}', how='{how}')",
            "Reshape to wide": "df_wide = df.pivot(index='{index}', columns='{columns}', values='{values}')",
            "Reshape to long": "df_long = df.melt(id_vars={id_vars}, value_vars={value_vars})",
            
            # Statistical operations  
            "Apply Bonferroni": "from statsmodels.stats.multitest import multipletests; reject, p_corrected, _, _ = multipletests(p_values, method='bonferroni', alpha={alpha})",
            "Apply FDR": "from statsmodels.stats.multitest import multipletests; reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha={alpha})",
            "Compute correlation": "corr = df[cols].corr(method='{method}')",
            "Fit regression": "from statsmodels.api import OLS, add_constant; X_const = add_constant(X); model = OLS(y, X_const).fit()",
            "Bootstrap": "from sklearn.utils import resample; X_boot, y_boot = resample(X, y, n_samples=len(X), random_state={seed})",
            
            # Validation operations
            "Check missing": "assert df.isnull().sum().sum() == 0, 'Missing values found'",
            "Check range": "assert df['{col}'].between({min}, {max}).all(), 'Values out of range'",
            "Check shape": "assert df.shape == {expected_shape}, f'Shape mismatch: {{df.shape}}'",
        }
    
    def make_specific(self, operation: str, context: Dict[str, Any] = None) -> List[str]:
        """Convert vague operation to specific code"""
        if context is None:
            context = {}
        
        # Check if already specific (contains function call)
        if '(' in operation and ')' in operation:
            return [operation]
        
        # Look for mapping
        for vague, specific in self.operation_map.items():
            if vague.lower() in operation.lower():
                # Format with context if available
                try:
                    formatted = specific.format(**context)
                except KeyError:
                    formatted = specific
                return [formatted]
        
        # If no mapping found, return as-is but flag for review
        return [f"# TODO: Specify - {operation}"]


class CapabilityVerifier:
    """Verifies that requested capabilities exist in tools"""
    
    def __init__(self):
        self.capability_map = {
            'bootstrap': ['bootstrap_regression', 'resample', 'bootstrap_ci'],
            'cross_validation': ['cross_validate', 'KFold', 'cross_val_score'],
            'multiple_comparisons': ['multipletests', 'bonferroni', 'fdr_correction'],
            'effect_size': ['cohens_d', 'compute_effect_size', 'f2_score'],
            'power_analysis': ['power_analysis', 'tt_ind_solve_power'],
        }
    
    def verify_capability(self, capability: str, available_tools: Dict) -> Tuple[bool, List[str]]:
        """
        Check if capability exists in available tools
        Returns: (exists, alternative_functions)
        """
        if capability in self.capability_map:
            alternatives = self.capability_map[capability]
            
            found = []
            for alt in alternatives:
                for tool_key in available_tools:
                    if alt in tool_key.lower():
                        found.append(tool_key)
            
            return len(found) > 0, found
        
        return False, []


def create_verification_report(rq_id: str, plan_file: str, tools_file: str) -> Dict[str, Any]:
    """
    Main verification function that checks everything
    Returns comprehensive verification report
    """
    report = {
        'rq_id': rq_id,
        'timestamp': pd.Timestamp.now().isoformat(),
        'checks': {},
        'corrections': [],
        'warnings': [],
        'errors': []
    }
    
    # Initialize verifiers
    tool_verifier = ToolVerifier()
    path_verifier = PathVerifier(rq_id)
    op_specifier = OperationSpecifier()
    cap_verifier = CapabilityVerifier()
    
    # Load files
    with open(tools_file) as f:
        tools_yaml = yaml.safe_load(f)
    
    # Verify each tool
    tools_verified = 0
    tools_total = len(tools_yaml.get('analysis_tools', {}))
    
    for tool_name, tool_spec in tools_yaml.get('analysis_tools', {}).items():
        result = tool_verifier.verify_function(tool_spec['module'], tool_spec['function'])
        
        if result['exists']:
            tools_verified += 1
        else:
            report['errors'].append(f"Tool not found: {tool_spec['module']}.{tool_spec['function']}")
            if result['similar']:
                report['warnings'].append(f"Similar functions: {result['similar']}")
    
    report['checks']['tools_verified'] = f"{tools_verified}/{tools_total}"
    
    # Additional checks would go here...
    
    return report


if __name__ == "__main__":
    # Test verification
    import sys
    if len(sys.argv) > 1:
        rq_id = sys.argv[1]
        plan_file = f"results/{rq_id}/docs/2_plan.md"
        tools_file = f"results/{rq_id}/docs/3_tools.yaml"
        
        report = create_verification_report(rq_id, plan_file, tools_file)
        print(yaml.dump(report, default_flow_style=False))
    else:
        # Run basic tests
        verifier = ToolVerifier()
        print(f"Found {len(verifier.tool_map)} functions")
        print(f"Found {len(verifier.validator_map)} validators")
        
        # Test path corrections
        path_ver = PathVerifier("ch7/7.1.1")
        test_paths = [
            "data/cache/master.xlsx",
            "data/dfnonvr.csv",
            "data/step01_output.csv"
        ]
        for path in test_paths:
            corrected, exists, note = path_ver.verify_input_path(path, 1)
            print(f"{path} -> {corrected} ({note})")