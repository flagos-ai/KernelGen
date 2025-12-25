# Test Examples for Converter Tool

This directory contains example input files for testing the converter tool.

## Files

1. **relu_triton.py** - Triton operator implementation
2. **relu_baseline.py** - PyTorch baseline implementation
3. **test_relu_accuracy.py** - Accuracy test (bench format)
4. **test_relu_performance.py** - Performance test (bench format)

## Usage

Test the converter tool with these files:

```bash
cd /share/project/tj/fork/FlagGems
python src/flag_gems/experimental_ops/tools/kernelgen_to_flaggems.py \
    src/flag_gems/experimental_ops/tools/tests \
    /tmp/output \
    relu
```

## Expected Output

The tool should generate:
- `/tmp/output/relu.py` - Triton operator (copied)
- `/tmp/output/relu_test.py` - Converted test file with baseline
