# Test Format Converter

This tool converts bench-format tests to FlagGems pytest format.

## Usage

```bash
python src/flag_gems/experimental_ops/tools/convert.py <input_dir> <output_dir> <operator_name>
```

## Input Files

The tool expects 4 files in the input directory:
1. Triton operator implementation (e.g., `relu_triton.py`)
2. PyTorch baseline implementation (e.g., `relu_baseline.py`)
3. Accuracy test file (e.g., `test_relu_accuracy.py`)
4. Performance test file (e.g., `test_relu_performance.py`)

## Output Files

The tool generates 2 files:
1. `<operator_name>.py` - Triton operator implementation (copied)
2. `<operator_name>_test.py` - Converted test file with baseline

## Example

```bash
python src/flag_gems/experimental_ops/tools/convert.py ./input_files ./output_files relu
```
