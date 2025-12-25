#!/usr/bin/env python3
"""
Test Format Converter
Converts bench-format tests to FlagGems pytest format.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class TestConverter:
    """Converts bench-format tests to FlagGems pytest format."""

    def __init__(self, input_dir: str, output_dir: str, operator_name: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.operator_name = operator_name

        # File contents
        self.triton_code = None
        self.baseline_code = None
        self.accuracy_test = None
        self.performance_test = None

    def identify_files(self) -> Dict[str, Path]:
        """Identify the 4 input files."""
        files = list(self.input_dir.glob("*.py"))

        if len(files) != 4:
            raise ValueError(f"Expected 4 Python files, found {len(files)}")

        identified = {}

        for file in files:
            content = file.read_text()

            # Identify file type by content patterns
            if "triton.jit" in content or "@triton.jit" in content:
                identified["triton"] = file
            elif "def " in content and "torch.Tensor" in content and "@" not in content:
                # Baseline implementation (no decorators)
                identified["baseline"] = file
            elif "@label" in content or "@parametrize" in content:
                # Test file - distinguish by filename or function name
                filename_lower = file.name.lower()
                if "accuracy" in filename_lower:
                    identified["accuracy_test"] = file
                elif "performance" in filename_lower or "benchmark" in filename_lower:
                    identified["performance_test"] = file
                else:
                    # Check function name pattern
                    if "def test_" in content and "benchmark" not in content:
                        identified["accuracy_test"] = file
                    else:
                        identified["performance_test"] = file

        # If we couldn't identify by name, use heuristics
        if "accuracy_test" not in identified or "performance_test" not in identified:
            test_files = [f for f in files if f not in identified.values()]
            if len(test_files) == 2:
                # Assume first is accuracy, second is performance
                identified["accuracy_test"] = test_files[0]
                identified["performance_test"] = test_files[1]

        return identified

    def read_files(self, files: Dict[str, Path]):
        """Read all input files."""
        self.triton_code = files["triton"].read_text()
        self.baseline_code = files["baseline"].read_text()
        self.accuracy_test = files["accuracy_test"].read_text()
        self.performance_test = files["performance_test"].read_text()

    def convert_decorators(self, code: str) -> str:
        """Convert bench decorators to pytest decorators."""
        # Convert @label("xxx") to @pytest.mark.xxx
        code = re.sub(r'@label\(["\'](\w+)["\']\)', r'@pytest.mark.\1', code)

        # Convert @parametrize to @pytest.mark.parametrize
        code = re.sub(r'@parametrize\(', r'@pytest.mark.parametrize(', code)

        return code

    def convert_imports(self, code: str) -> str:
        """Convert bench imports to pytest/flaggems imports."""
        # Remove bench-related imports
        lines = code.split('\n')
        new_lines = []

        for line in lines:
            # Skip bench imports
            if 'import bench' in line or 'from bench' in line:
                continue
            new_lines.append(line)

        # Add pytest and flaggems imports at the beginning
        imports = [
            "import pytest",
            "import triton",
            "",
            "import flag_gems",
            f"from flag_gems.experimental_ops.{self.operator_name} import {self.operator_name} as gems_{self.operator_name}"
        ]

        # Find where to insert imports (after initial comments)
        insert_idx = 0
        for i, line in enumerate(new_lines):
            if line.strip() and not line.strip().startswith('#'):
                insert_idx = i
                break

        # Insert imports
        new_lines = new_lines[:insert_idx] + imports + [''] + new_lines[insert_idx:]

        return '\n'.join(new_lines)

    def convert_function_calls(self, code: str) -> str:
        """Convert bench function calls to appropriate implementations."""
        # Replace bench.relu with baseline function name
        code = re.sub(r'bench\.\w+\(', f'{self.operator_name}_baseline(', code)

        # Replace bench.triton.relu with gems_relu
        code = re.sub(r'bench\.triton\.\w+\(', f'gems_{self.operator_name}(', code)

        # Replace device variable with flag_gems.device (but not parameter name)
        code = re.sub(r'=\s*device\b', '=flag_gems.device', code)
        code = re.sub(r'\(device\)', '(flag_gems.device)', code)

        # Replace to_reference with input.cpu()
        code = re.sub(r'to_reference\([^,]+,\s*True\)', 'input.cpu()', code)

        # Replace assert_close with torch.testing.assert_close
        code = re.sub(
            r'assert_close\(([^,]+),\s*([^,]+),\s*dtype[^)]*\)',
            r'torch.testing.assert_close(\1.cpu(), \2, rtol=1e-3, atol=1e-3)',
            code
        )

        return code

    def convert_test_function_names(self, code: str) -> str:
        """Convert test function names to pytest format."""
        # Ensure test functions start with test_
        code = re.sub(r'def (\w+)\(', lambda m: f'def test_{m.group(1)}(' if not m.group(1).startswith('test_') else m.group(0), code)

        # Convert benchmark functions to test_performance
        code = re.sub(r'def test_(\w+)_benchmark\(', r'def test_\1_performance(', code)

        return code

    def extract_baseline_function(self) -> str:
        """Extract baseline function and rename it."""
        # Find the function definition
        match = re.search(r'def (\w+)\(.*?\).*?(?=\ndef |\Z)', self.baseline_code, re.DOTALL)
        if match:
            func_code = match.group(0)
            # Rename function to <operator>_baseline
            func_code = re.sub(r'def \w+\(', f'def {self.operator_name}_baseline(', func_code)
            return func_code
        return ""

    def merge_test_files(self) -> str:
        """Merge baseline and test files into one."""
        # Start with header
        header = f"# {self.operator_name.upper()} operator test\n\n# PyTorch baseline\nimport torch\n\n"

        # Add baseline function
        baseline_func = self.extract_baseline_function()

        # Convert accuracy test
        accuracy_converted = self.convert_decorators(self.accuracy_test)
        accuracy_converted = self.convert_imports(accuracy_converted)
        accuracy_converted = self.convert_function_calls(accuracy_converted)
        accuracy_converted = self.convert_test_function_names(accuracy_converted)

        # Convert performance test
        perf_converted = self.convert_decorators(self.performance_test)
        perf_converted = self.convert_function_calls(perf_converted)
        perf_converted = self.convert_test_function_names(perf_converted)

        # Extract only test functions from performance test (skip imports)
        perf_lines = perf_converted.split('\n')
        perf_test_start = 0
        for i, line in enumerate(perf_lines):
            if line.strip().startswith('@pytest.mark'):
                perf_test_start = i
                break
        perf_test_only = '\n'.join(perf_lines[perf_test_start:])

        # Combine all parts
        final_code = header + baseline_func + '\n\n' + accuracy_converted + '\n\n' + perf_test_only

        return final_code

    def generate_output_files(self):
        """Generate output files."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy triton operator
        triton_output = self.output_dir / f"{self.operator_name}.py"
        triton_output.write_text(self.triton_code)
        print(f"✓ Generated: {triton_output}")

        # Generate merged test file
        test_code = self.merge_test_files()
        test_output = self.output_dir / f"{self.operator_name}_test.py"
        test_output.write_text(test_code)
        print(f"✓ Generated: {test_output}")

    def convert(self):
        """Main conversion process."""
        print(f"Converting {self.operator_name} operator...")

        # Step 1: Identify files
        print("Step 1: Identifying input files...")
        files = self.identify_files()
        print(f"  - Triton: {files['triton'].name}")
        print(f"  - Baseline: {files['baseline'].name}")
        print(f"  - Accuracy test: {files['accuracy_test'].name}")
        print(f"  - Performance test: {files['performance_test'].name}")

        # Step 2: Read files
        print("Step 2: Reading files...")
        self.read_files(files)

        # Step 3: Generate output
        print("Step 3: Generating output files...")
        self.generate_output_files()

        print(f"\n✓ Conversion complete!")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert bench-format tests to FlagGems pytest format"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the 4 input files"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write output files"
    )
    parser.add_argument(
        "operator_name",
        help="Name of the operator (e.g., relu)"
    )

    args = parser.parse_args()

    try:
        converter = TestConverter(args.input_dir, args.output_dir, args.operator_name)
        converter.convert()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
