---
name: kernelgen-specialize-for-flaggems
description: Automatically migrate GPU Triton operators to Ascend NPU using the dedicated MCP specialize_kernel tool and integrate into the FlagGems project. Complete workflow: MCP auto-specialization -> FlagGems adaptation -> file placement -> operator registration -> auto testing -> performance verification. Supports four integration modes: vendor-ops (default), vendor-fused, override-builtin, experimental. Trigger scenarios: migrating CUDA Triton operators to FlagGems, adding Ascend backend operators for FlagGems, optimizing NPU implementation of existing FlagGems operators.
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# GPU Triton Operator MCP Auto-Migration + FlagGems Integration

This skill combines the MCP auto-specialization tool (`specialize_kernel`) with FlagGems framework integration, enabling a fully automated workflow from GPU Triton operators to Ascend NPU FlagGems operators.

---

## Usage

```bash
/kernelgen-specialize-for-flaggems <source_file_path_or_op_name> [integration_mode] [output_directory]
```

**Parameters**:
- `source_file_path_or_op_name`: Required, source file path or operator name of the GPU Triton operator
- `integration_mode`: Optional, default `vendor-ops`
  - `vendor-ops`: Ascend-specific operator (default), placed in `runtime/backend/_ascend/ops/`
  - `vendor-fused`: Ascend fused operator, placed in `runtime/backend/_ascend/ops/`
  - `override-builtin`: Override built-in operator, placed in `src/flag_gems/ops/`
  - `experimental`: Experimental operator, placed in `experimental_ops/`
- `output_directory`: Optional, default `/home/FlagGems`

**Examples**:
```bash
# Default mode (vendor-ops)
/kernelgen-specialize-for-flaggems /home/test/gelu_kernel.py

# Specify integration mode
/kernelgen-specialize-for-flaggems /home/test/softmax.py vendor-fused

# Specify output directory
/kernelgen-specialize-for-flaggems /home/test/layernorm.py experimental /home/MyFlagGems
```

---

## Automated Workflow (8 Steps)

### 1. Read Source Code
Read the GPU Triton kernel code from the file path provided by the user.

### 2. Extract Kernel Information
Automatically extract:
- Kernel name (e.g., `gelu_kernel`)
- Function signature and parameters
- Operator function description

### 3. Call MCP Tool for Auto-Specialization
Use the `mcp__kernelgen-mcp__specialize_kernel` tool:
```python
{
    "kernel_name": "gelu",
    "triton_code": "<complete source code>",
    "target_platform": "huawei",
    "func_desc": "GELU activation function",
    "arg_names": "x",
    "arg_type": "torch.Tensor",
    "arg_descs": "input tensor",
    "output_arg_desc": "output tensor"
}
```

### 4. FlagGems Framework Adaptation
Adapt the specialized code returned by MCP to FlagGems format based on the integration mode:

#### vendor-ops / vendor-fused Mode
```python
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import volume

@libentry()
@triton.jit
def gelu_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE

    for sub in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        idx = offset + sub + tl.arange(0, BLOCK_SIZE_SUB)
        mask = idx < N
        x = tl.load(in_ptr + idx, mask=mask, care_padding=False)
        out = x * 0.5 * (1.0 + tl.erf(x / 1.4142135623730951))
        tl.store(out_ptr + idx, out, mask=mask)

def gelu(x: torch.Tensor) -> torch.Tensor:
    N = volume(x.shape)
    out = torch.empty_like(x)

    # Dynamically compute BLOCK_SIZE
    min_block = triton.next_power_of_2(triton.cdiv(N, 65535))
    BLOCK_SIZE = max(32768, min_block)
    BLOCK_SIZE_SUB = 1024

    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        gelu_kernel[grid](x, out, N, BLOCK_SIZE, BLOCK_SIZE_SUB)

    return out
```

#### override-builtin Mode
```python
import torch
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic

@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_kernel(x):
    return x * 0.5 * (1.0 + tl.erf(x / 1.4142135623730951))

def gelu(A):
    return gelu_kernel(A)
```

#### experimental Mode
```python
import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel_experimental(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr
):
    # MCP specialized code
    pid = tl.program_id(0)
    # ... full implementation
    pass

def gelu_experimental(x: torch.Tensor) -> torch.Tensor:
    # Experimental wrapper function
    pass
```

### 5. File Placement
Save files to the correct location based on integration mode:

| Mode | Target Directory | Filename Format |
|------|-----------------|-----------------|
| vendor-ops | `runtime/backend/_ascend/ops/` | `<op_name>.py` |
| vendor-fused | `runtime/backend/_ascend/ops/` | `<op_name>_fused.py` |
| override-builtin | `src/flag_gems/ops/` | `<op_name>.py` |
| experimental | `experimental_ops/` | `<op_name>_exp.py` |

### 6. Operator Registration
Automatically register operators based on mode:

#### vendor-ops Registration
Update `runtime/backend/_ascend/__init__.py`:
```python
from .ops.gelu import gelu

__all__ = ["gelu", ...]
```

#### override-builtin Registration
Update `src/flag_gems/ops/__init__.py`:
```python
from .gelu import gelu

__all__ = ["gelu", ...]
```

#### experimental Registration
Update `experimental_ops/__init__.py`:
```python
from .gelu_exp import gelu_experimental

__all__ = ["gelu_experimental", ...]
```

### 7. Auto Test Execution

#### Unit Tests
```bash
cd /home/FlagGems

# vendor-ops / override-builtin
pytest tests/test_unary_pointwise_ops.py::test_gelu -v --tb=short

# experimental
pytest experimental_tests/test_<op_name>_exp.py -v --tb=short
```

Record test results:
- Pass/fail status
- Execution time
- Error messages (if any)

#### Performance Tests
```bash
cd /home/FlagGems

# vendor-ops / override-builtin
pytest benchmark/test_unary_pointwise_perf.py::test_perf_gelu -v -s --tb=short

# experimental
pytest experimental_tests/benchmark_<op_name>_exp.py -v -s --tb=short
```

Record performance data:
- NPU execution time
- CPU execution time
- Speedup ratio

### 8. Generate Integration Report
Automatically generate `FLAGGEMS_INTEGRATION_REPORT.md` containing:

```markdown
# <op_name> FlagGems Integration Report

**Migration date**: 2026-03-23
**Source platform**: CUDA GPU
**Target platform**: Ascend NPU
**Migration method**: MCP auto-specialization + FlagGems integration
**Integration mode**: vendor-ops

---

## Integration Results Overview

### MCP Auto-Specialization
- **Status**: Success
- **Specialized code**: Generated
- **Key optimizations**:
  - Dynamic BLOCK_SIZE computation (ensuring coreDim <= 65535)
  - Intra-core tiling (BLOCK_SIZE_SUB=1024)
  - care_padding=False optimization

### FlagGems Adaptation
- **Target file**: runtime/backend/_ascend/ops/gelu.py
- **Operator registration**: __init__.py updated
- **Framework compatibility**: Using @libentry() decorator

### Unit Tests
- **Status**: All passed
- **Test cases**: 12/12
- **Execution time**: 8.32s

### Performance Tests
- **Status**: All passed
- **Test cases**: 8/8
- **Execution time**: 5.67s

---

## Performance Data Analysis

| Input Shape | Data Type | NPU (ms) | CPU (ms) | Speedup |
|-------------|-----------|----------|----------|---------|
| (1024,) | float32 | 0.12 | 0.45 | 3.75x |
| (1024, 1024) | float32 | 1.23 | 8.91 | 7.24x |
| (512, 512, 4) | float16 | 0.89 | 6.34 | 7.12x |

**Average speedup**: 6.04x

---

## Code Comparison

### Before MCP Specialization (GPU)
```python
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(x_ptr + offset, mask=mask)
    out = x * 0.5 * (1.0 + tl.erf(x / 1.4142135623730951))
    tl.store(out_ptr + offset, out, mask=mask)
```

### After MCP Specialization (NPU + FlagGems)
```python
@libentry()
@triton.jit
def gelu_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE

    for sub in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        idx = offset + sub + tl.arange(0, BLOCK_SIZE_SUB)
        mask = idx < N
        x = tl.load(in_ptr + idx, mask=mask, care_padding=False)
        out = x * 0.5 * (1.0 + tl.erf(x / 1.4142135623730951))
        tl.store(out_ptr + idx, out, mask=mask)
```

**Key improvements**:
- Added BLOCK_SIZE_SUB intra-core tiling
- care_padding=False to reduce dependencies
- @libentry() FlagGems decorator
- Dynamic BLOCK_SIZE computation

---

## Verification Checklist

- [x] MCP auto-specialization successful
- [x] FlagGems framework adaptation complete
- [x] Files placed in correct directory
- [x] Operator registered in __init__.py
- [x] All unit tests passed
- [x] All performance tests passed
- [x] Speedup meets expectations (> 3x)

---

## Usage Example

```python
import torch
import flag_gems

# Set to use Ascend backend
flag_gems.use_gems()

x = torch.randn(1024, 1024, device='npu')
out = torch.nn.functional.gelu(x)  # Automatically uses Ascend-optimized version
```

---

## Next Steps

1. Validate performance on more input shapes
2. Fuse with other operators to reduce kernel launch overhead
3. Adjust BLOCK_SIZE_SUB parameters for further optimization
4. Consider contributing to the official FlagGems repository
```

---

## MCP Tool Call Details

### Tool Name
`mcp__kernelgen-mcp__specialize_kernel`

### Required Parameters
- `kernel_name` (str): Operator name
- `triton_code` (str): Complete GPU Triton source code
- `target_platform` (str): Must be set to `"huawei"`

### Optional Parameters
- `func_desc` (str): Operator function description
- `arg_names` (str): Input parameter names (comma-separated)
- `arg_type` (str): Input parameter types
- `arg_descs` (str): Input parameter descriptions (comma-separated)
- `output_arg_desc` (str): Output parameter description
- `check_result` (dict): Previous verification results (if code has errors that need fixing)

### Return Value
```python
{
    "triton_code": "<Ascend-specialized code>",
    "mode": "specialize",
    "target_platform": "huawei"
}
```

---

## FlagGems Integration Mode Details

### vendor-ops (Default, Recommended)
**Applicable scenarios**:
- Ascend-specific operator implementation
- Does not affect other backends
- Requires Ascend-specific optimizations

**Directory structure**:
```
runtime/backend/_ascend/ops/
├── __init__.py
├── gelu.py
├── softmax.py
└── layernorm.py
```

**Code template**:
```python
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import volume

@libentry()
@triton.jit
def kernel(...):
    # MCP specialized kernel
    pass

def op_name(x: torch.Tensor) -> torch.Tensor:
    # Wrapper function
    with torch_device_fn.device(x.device):
        kernel[grid](...)
    return out
```

### vendor-fused
**Applicable scenarios**:
- Multi-operator fusion optimization
- Ascend-specific fusion patterns
- Reducing kernel launch overhead

**Naming convention**:
- Filename: `<op1>_<op2>_fused.py`
- Function name: `<op1>_<op2>_fused`

**Example**:
```python
# gelu_add_fused.py
def gelu_add_fused(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Fused GELU + Add
    pass
```

### override-builtin
**Applicable scenarios**:
- Replacing FlagGems built-in operators
- Takes effect globally (all backends)
- Should be used with caution

**Notes**:
- Will overwrite the original implementation
- Must ensure compatibility
- Recommended to validate in experimental mode first

### experimental
**Applicable scenarios**:
- Experimental operators
- Not yet fully validated
- Rapid prototyping

**Directory structure**:
```
experimental_ops/
├── __init__.py
├── gelu_exp.py
└── experimental_tests/
    ├── test_gelu_exp.py
    └── benchmark_gelu_exp.py
```

---

## Common Issue Handling

### Issue 1: MCP Tool Call Failure
**Error message**: `MCP server not available`

**Solution**:
1. Check if MCP service is running
2. Verify MCP configuration in settings.json
3. Confirm permission settings include `mcp__kernelgen-mcp__specialize_kernel`

### Issue 2: FlagGems Test Failure
**Error message**: `ModuleNotFoundError: No module named 'flag_gems'`

**Solution**:
```bash
cd /home/FlagGems
pip install -e .
```

### Issue 3: Operator Registration Not Taking Effect
**Symptom**: Original implementation is still used when calling the operator

**Solution**:
1. Check if `__init__.py` has correct imports
2. Confirm `flag_gems.use_gems()` has been called
3. Restart the Python environment

### Issue 4: Performance Below Expectations
**Symptom**: Speedup < 2x

**Solution**:
1. Set environment variable: `export TRITON_ALL_BLOCKS_PARALLEL=1`
2. Adjust BLOCK_SIZE_SUB parameter (try 2048, 4096)
3. Check input data scale (small data may not be suitable for NPU)

---

## Performance Optimization Recommendations

### 1. Environment Variable Configuration
```bash
export TRITON_ALL_BLOCKS_PARALLEL=1  # Reduce scheduling overhead
export TRITON_DEBUG=1                # Debug mode (optional)
```

### 2. Tiling Parameter Tuning
```python
# Default configuration
BLOCK_SIZE = 32768
BLOCK_SIZE_SUB = 1024

# Large data optimization
BLOCK_SIZE = 65536
BLOCK_SIZE_SUB = 2048

# Small data optimization
BLOCK_SIZE = 16384
BLOCK_SIZE_SUB = 512
```

### 3. Operator Fusion
Fuse multiple small operators into one large operator to reduce kernel launch overhead:
```python
# Before fusion: 3 kernel launches
x = gelu(x)
x = add(x, y)
x = relu(x)

# After fusion: 1 kernel launch
x = gelu_add_relu_fused(x, y)
```

---

## Complete Examples

### Example 1: Migrate GELU Operator (vendor-ops Mode)

**Input**:
```bash
/kernelgen-specialize-for-flaggems /home/test/gelu_cuda.py vendor-ops
```

**Automated workflow**:
1. Read `/home/test/gelu_cuda.py`
2. Call MCP tool to specialize for Ascend
3. Adapt to FlagGems vendor-ops format
4. Save to `runtime/backend/_ascend/ops/gelu.py`
5. Update `runtime/backend/_ascend/__init__.py`
6. Run unit tests: `pytest tests/test_unary_pointwise_ops.py::test_gelu -v`
7. Run performance tests: `pytest benchmark/test_unary_pointwise_perf.py::test_perf_gelu -v -s`
8. Generate report: `FLAGGEMS_INTEGRATION_REPORT.md`

**Output**:
```
MCP auto-specialization successful
FlagGems adaptation complete
File saved: runtime/backend/_ascend/ops/gelu.py
Operator registration complete
Unit tests: 12/12 passed (8.32s)
Performance tests: 8/8 passed (5.67s)
Average speedup: 6.04x

Detailed report: /home/FlagGems/FLAGGEMS_INTEGRATION_REPORT.md
```

### Example 2: Migrate Softmax Operator (override-builtin Mode)

**Input**:
```bash
/kernelgen-specialize-for-flaggems /home/test/softmax_cuda.py override-builtin
```

**Notes**:
- Will overwrite the FlagGems built-in softmax implementation
- Must ensure the new implementation is compatible with all test cases
- Recommended to validate in experimental mode first

### Example 3: Migrate Fused Operator (vendor-fused Mode)

**Input**:
```bash
/kernelgen-specialize-for-flaggems /home/test/gelu_add_cuda.py vendor-fused
```

**Output files**:
- `runtime/backend/_ascend/ops/gelu_add_fused.py`
- Function name: `gelu_add_fused(x, y)`

---

## Relationship with Other Skills

### vs. kernelgen-specialize
- **Similarities**: Both use MCP tool for auto-specialization
- **Differences**: This skill additionally integrates into the FlagGems framework

### vs. kernelgen-specialize
- **Similarities**: Both integrate into the FlagGems framework
- **Differences**: This skill uses MCP auto-specialization, no manual code modification needed

### Recommended Use Cases
- **This skill**: Rapid migration + auto-integration, suitable for most scenarios
- **kernelgen-specialize**: Only need specialized code, no FlagGems integration needed
- **kernelgen-specialize**: Need manual control over code details

---

## Technical Details

### MCP Specialization Key Optimizations
1. **Dynamic BLOCK_SIZE computation**
   ```python
   min_block = triton.next_power_of_2(triton.cdiv(N, 65535))
   BLOCK_SIZE = max(32768, min_block)
   ```

2. **Intra-core tiling (Sub-tiling)**
   ```python
   for sub in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
       idx = offset + sub + tl.arange(0, BLOCK_SIZE_SUB)
       # Process BLOCK_SIZE_SUB-sized data blocks
   ```

3. **care_padding=False optimization**
   ```python
   x = tl.load(ptr + idx, mask=mask, care_padding=False)
   ```

### FlagGems Framework Requirements
1. **Use @libentry() decorator**
   ```python
   from flag_gems.utils import libentry

   @libentry()
   @triton.jit
   def kernel(...):
       pass
   ```

2. **Device context management**
   ```python
   from flag_gems.runtime import torch_device_fn

   with torch_device_fn.device(x.device):
       kernel[grid](...)
   ```

3. **Shape utility functions**
   ```python
   from flag_gems.utils.shape_utils import volume

   N = volume(x.shape)  # Compute total number of elements
   ```

---

## Limitations and Notes

1. **MCP service dependency**
   - Requires `kernelgen-mcp` MCP server to be running
   - Requires network connectivity (if MCP service is remote)

2. **FlagGems environment requirements**
   - Requires FlagGems installation: `pip install -e /home/FlagGems`
   - Requires Ascend NPU device and drivers

3. **Code review recommendations**
   - MCP auto-generated code may require manual review
   - Recommended to validate in experimental mode first, then switch to vendor-ops

4. **Performance expectations**
   - Small data (< 1000 elements) may not be suitable for NPU
   - Recommended to test on actual business data scales

---

## Troubleshooting

### Debug Mode
```bash
export TRITON_DEBUG=1
export TRITON_PRINT_AUTOTUNING=1
```

### View Generated IR
```bash
export TRITON_CACHE_DIR=/tmp/triton_cache
ls /tmp/triton_cache  # View compilation cache
```

### Performance Profiling
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.NPU],
    record_shapes=True
) as prof:
    out = gelu(x)

print(prof.key_averages().table(sort_by="npu_time_total"))
```

---

## References

- [FlagGems Official Documentation](https://github.com/FlagOpen/FlagGems)
- [Ascend Triton Development Guide](https://www.hiascend.com/document)
- [MCP Tool Usage Instructions](internal documentation)
- [kernelgen-specialize Skill](kernelgen-specialize.md)
- [kernelgen-specialize-for-flaggems Skill](kernelgen-specialize-for-flaggems.md)
