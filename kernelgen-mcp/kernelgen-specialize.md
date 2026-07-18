---
name: kernelgen-specialize
description: Migrate GPU Triton operators to Huawei Ascend NPU. Supports two modes: (1) Automatic migration - automatically specialize operators to the Ascend platform via the dedicated MCP specialize_kernel tool; (2) Manual migration guide - provides complete architecture difference analysis, code modification steps, common issue resolution (coreDim exceeding limits, UB overflow, memory access alignment, etc.) and performance optimization tips. Trigger scenarios: GPU to NPU migration, device='cuda' changed to 'npu', Ascend adaptation, "Triton migration", "NPU porting", "Ascend migration", etc.
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


# Complete Guide: Migrating GPU Triton Operators to Ascend NPU

This guide covers the complete process of migrating Triton operators from GPU to Ascend NPU, including architecture difference analysis, code migration steps, common issue resolution, and performance optimization tips.

---

## Quick Migration Checklist

Basic steps for migrating GPU Triton operators to NPU:

- [ ] Add `import torch_npu` import statement
- [ ] Change `device='cuda'` to `device='npu'`
- [ ] Remove GPU-specific APIs (e.g., `triton.runtime.driver.active.get_active_torch_device()`)
- [ ] Remove GPU device consistency check assertions
- [ ] Check Grid configuration, ensure coreDim <= 65535
- [ ] Verify BLOCK_SIZE does not cause UB overflow
- [ ] Ensure memory access alignment: 32 bytes for VV scenarios, 512 bytes for CV scenarios

---

## Automated Migration Tool (MCP Integration)

### Feature Description

By using the dedicated `specialize_kernel` tool from `kernelgen-mcp`, GPU Triton operators can be automatically specialized to the Ascend platform.

### Usage

**Automatic trigger**: When the user mentions requests like "migrate this operator to Ascend", "help me specialize this kernel", etc., the MCP tool is automatically invoked.

**Tool call parameters**:
- `kernel_name`: Operator name (e.g., "gelu", "softmax")
- `triton_code`: Complete Triton kernel code string
- `target_platform`: Set to `"huawei"` (Ascend platform)
- Other optional parameters:
  - `func_desc`: Operator function description
  - `arg_names`: Input parameter names (comma-separated)
  - `arg_type`: Input parameter types
  - `arg_descs`: Input parameter descriptions (comma-separated)
  - `output_arg_desc`: Output parameter description
  - `check_result`: Previous verification results (if code has errors that need fixing)

### Tool Call Example

```
Using MCP tool for automatic migration:

mcp__kernelgen-mcp__specialize_kernel:
  kernel_name: "gelu"
  triton_code: |
    @triton.jit
    def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        output = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(output_ptr + offsets, output, mask=mask)
  target_platform: "huawei"
  func_desc: "GELU activation function: output = x * 0.5 * (1 + erf(x / sqrt(2)))"
  arg_names: "x"
  arg_type: "torch.Tensor"
  arg_descs: "input tensor"
  output_arg_desc: "output tensor, same shape as input"
```

### Automated Migration Workflow

When the user requests migrating a Triton operator to Ascend, Claude should execute the following steps:

1. **Read source code**: Read the Triton kernel code from the file path provided by the user
2. **Extract information**: Extract kernel name and parameter information from the code
3. **Call MCP tool**: Use `mcp__kernelgen-mcp__specialize_kernel` with `target_platform="huawei"`
4. **Save specialized code**: Save the specialized code returned by MCP to the specified directory
5. **Copy related files**: Copy test files, benchmark files, and reference implementations to the output directory
6. **Auto-execute tests**: Run unit tests to verify correctness
7. **Auto-execute performance tests**: Run benchmarks to obtain speedup data
8. **Generate test report**: Summarize test results and performance data

### MCP Tool Return Content

The tool returns a dictionary containing the following fields:
- `triton_code`: Triton kernel code specialized for the Ascend platform
- `mode`: The operation mode used ("specialize")
- `target_platform`: The target platform name ("huawei")

### Post-Migration Verification Checklist

After generating code with the MCP tool, the following checks are still required:
- [ ] Has `import torch_npu` been added
- [ ] Has `device='cuda'` been changed to `device='npu'`
- [ ] Have GPU-specific APIs been removed
- [ ] Does the Grid configuration satisfy coreDim <= 65535
- [ ] Will BLOCK_SIZE cause UB overflow (192KB limit)
- [ ] Is memory access alignment correct (32 bytes for VV scenarios, 512 bytes for CV scenarios)

### Auto Test Execution

After migration is complete, the following tests **must** be automatically executed:

#### 1. Unit Tests
```bash
cd <output_dir>
pytest <kernel_name>_test.py -v --tb=short 2>&1 | head -100
```
- Verify operator correctness
- Test multiple input shapes and data types
- Compare NPU implementation with CPU reference implementation
- Record pass/fail status and execution time

#### 2. Performance Tests
```bash
cd <output_dir>
pytest <kernel_name>_benchmark.py -v -s --tb=short 2>&1 | head -120
```
- Measure Triton(NPU) vs PyTorch(CPU) execution time
- Calculate speedup ratio
- Output detailed performance data
- Timeout setting: 300 seconds

#### 3. Test Report Generation

Automatically generate `TEST_REPORT.md` containing:
- Pass/fail unit test status (X/Y passed)
- Performance comparison table (PyTorch vs Triton execution time)
- Speedup statistics (Speedup: Xx)
- Performance analysis (data scale, memory access patterns, applicable scenarios)
- Optimization suggestions (environment variables, Tiling parameters, use cases)

#### 4. Update README

Add test result summary to README.md:
```markdown
## Test Results

### Unit Tests: X/Y Passed
### Performance Tests: X/Y Passed

See [TEST_REPORT.md](TEST_REPORT.md) for detailed test report
```

### Notes

1. **MCP service dependency**: Requires `kernelgen-mcp` MCP server to be configured
2. **NPU device requirement**: Tests need to be executed in an environment with Ascend NPU
3. **Test failure handling**: If tests fail, check Grid configuration, UB usage, memory access alignment; can pass previous error as `check_result` parameter to get auto-fixed code
4. **Performance optimization**: If speedup is not satisfactory, try setting `export TRITON_ALL_BLOCKS_PARALLEL=1`

---

## Architecture Difference Comparison

### Core Differences Overview

| Dimension | GPU (NVIDIA) | Ascend NPU |
|-----------|-------------|------------------|
| Grid nature | Logical task dimensions (decoupled from physical cores) | Physical core group mapping (bound to AI Core topology) |
| Core count/dimension limits | No hard limits on grid dimensions/size | Grid size <= total AI Cores, and <= 65535 |
| Parallelism model | Can bind multi-dimensional axes (3D grid), each thread executes once | Physical cores (Vector/Cube), each core executes one Block, supports repeated scheduling |
| Memory access alignment | No mandatory requirements | VV scenarios: 32-byte alignment, CV scenarios: 512-byte alignment |
| On-chip memory | Shared Memory | UB (Unified Buffer), 192KB for Atlas 800T/I A2 |

### Core Migration Principles

1. **Abandon GPU "free logical grid definition"**, switch to Ascend "physical core group binding"
2. **Memory access alignment requirements**: 32-byte alignment for VV scenarios, 512-byte alignment for CV scenarios
3. **Remove GPU-specific synchronization APIs**
4. **Prefer 1D Grid**: 2D NPU adaptation will also be merged into 1D, actual grid values should align with chip physical core count
   - For example: `(20,)` and `(4, 5)` have the same effect

---

## Complete Migration Example (Vector Addition)

```diff
+ import torch_npu  # [NEW] Import Ascend NPU PyTorch adaptation library

import triton
import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()  # [REMOVE] GPU device auto-detection

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
-   assert x.device == DEVICE and y.device == DEVICE  # [REMOVE] GPU device check
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
- x = torch.rand(size, device='cuda')  # [BEFORE]
+ x = torch.rand(size, device='npu')   # [AFTER]
- y = torch.rand(size, device='cuda')  # [BEFORE]
+ y = torch.rand(size, device='npu')   # [AFTER]

output_torch = x + y
output_triton = add(x, y)
print(f'Max difference: {torch.max(torch.abs(output_torch - output_triton))}')
```

---

## Data Tiling Strategy

A reasonable data tiling strategy is critical for performance optimization. Common tiling parameters:

| Parameter | Description |
|-----------|-------------|
| `ncore` | Number of cores used (cross-core tiling) |
| `xblock` | Inter-core data block size (inter-core tiling) |
| `xblock_sub` | Intra-core tiling granularity (intra-core fine-grained partitioning) |

### Efficient Triton Implementation Example (GELU Operator)

```python
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel,
                         XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    # Inter-core tiling: each core processes XBLOCK-sized data
    xoffset = tl.program_id(0) * XBLOCK

    # Intra-core tiling: further divided into XBLOCK_SUB-sized chunks
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# Call example
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```

**Note**: The on-chip memory (UB) capacity of Atlas 800T/I A2 is **192KB**. When designing tiling strategies, ensure that the data volume per computation round does not exceed this limit.

---

## Common Issue Diagnosis and Resolution

### Issue 1: coreDim Exceeding Limit

**Error message**: `coreDim=xxxx can't be greater than UINT16_MAX`

**Cause**: NPU's coreDim parameter cannot exceed 65535. When processing large-scale data, simple grid partitioning may cause this limit to be exceeded.

**Solution 1**: Set environment variable (recommended)
```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```
When enabled, the compiler automatically adjusts the logical core count to match the physical core count, reducing scheduling overhead.

**Solution 2**: Dynamically compute BLOCK_SIZE
```python
N = inp.numel()
# Formula: coreDim = ceil(N / BLOCK_SIZE) <= 65535
min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block_size)  # At least 32768

grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
kernel[grid_fn](out, N, BLOCK_SIZE=BLOCK_SIZE)
```

### Issue 2: UB Space Overflow

**Error message**: `ub overflow, requires xxxx bits while 1572684 bits available!`

**Cause**: Memory usage exceeds the NPU's UB cache capacity (192KB = 1572864 bits).

**Solution**: Introduce BLOCK_SIZE_SUB parameter for block processing

```python
@triton.jit
def kernel(inp, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # Block processing to avoid UB overflow
    for sub_idx in range(tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)):
        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        # Load and process data in batches
        data = tl.load(inp + offsets, mask=mask, other=0)
        result = process(data)
        tl.store(out + offsets, result, mask=mask)

# Call example
MAIN_BLOCK_SIZE = 32768   # Ensure coreDim compliance
SUB_BLOCK_SIZE = 1024     # Control UB usage
grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
kernel[grid](inp, out, N, MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
```

### Issue 3: Discrete Memory Access and Scalar Inefficient Mapping

**Diagnostic method**:
```bash
export TRITON_DEBUG=1
# View generated IR
bishengir-compile xxx.ttadapter --target=Ascend910B3 \
    --enable-auto-multi-buffer=True \
    --enable-hfusion-compile=true \
    --enable-hivm-compile=true \
    --enable-triton-kernel-compile=true \
    --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```

**Root cause**: For `[1024, 32]` 2D data, if stride is set to `(32,)` instead of `(32, 1)`, it causes non-contiguous memory access, degrading performance.

**Optimization solution**: Adjust `block_ptr` shape/stride
```python
# Before optimization (inefficient)
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024,),
    strides=(32,),      # Non-contiguous memory access
    offsets=(i_t * 16,),
    block_shape=(BT,),
    order=(0,)
)

# After optimization (efficient)
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),
    strides=(32, 1),    # Contiguous memory access
    offsets=(i_t * BT, 0),
    block_shape=(BT, 32),
    order=(1, 0)        # Row-major order
)
```

---

## Performance Optimization Guide

### 1. Grid Core Allocation Optimization

When the Grid core count is large, use environment variables to improve performance:
```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

### 2. Instruction-Level Parallelism Optimization

#### Using `care_padding=False` to Reduce Synchronization

When `tl.load` uses a mask, NPU by default fills default values with the Vector core first, then transfers data with MTE2, creating a dependency. Using `care_padding=False` removes the default value filling, improving parallelism:

```python
# Before optimization (with dependency)
data = tl.load(input + idx, mask=mask)

# After optimization (reduced dependency)
data = tl.load(input + idx, mask=mask, care_padding=False)
```

**Applicable condition**: The unfilled portion does not affect subsequent computation results.

#### Using For Loops to Increase Tiling

Change single sequential execution to multiple block processing, enabling "data-in/compute/data-out" pipelining:

```python
# Before optimization: load all data at once
offset = tl.arange(0, max_num_tokens)
data = tl.load(ptr + offset, mask=offset < num)
tl.store(out + offset, process(data), mask=offset < num)

# After optimization: block processing
BLOCK_SIZE = 1024
num_loop = tl.cdiv(max_num_tokens, BLOCK_SIZE)
blk_offset = tl.arange(0, BLOCK_SIZE)
for i in range(num_loop):
    offset = blk_offset + i * BLOCK_SIZE
    data = tl.load(ptr + offset, mask=offset < num)
    tl.store(out + offset, process(data), mask=offset < num)
```

### 3. Data Type Optimization

The A2/A3 vector computation units do not support certain data types for some operations, causing fallback to scalar computation:

| OP Name | Unsupported Data Types | Recommended Alternative |
|---------|----------------------|------------------------|
| Vector ADD | int64 | int32 |
| Vector CMP | int64, int32 | float32 |

**Example: Avoid int64/int32 for CMP operations**
```python
# Before optimization (inefficient)
cols = tl.arange(0, BLOCK_N)  # cols is int64
xbar = tl.where(cols < N, x - mean, 0.0)

# After optimization (efficient)
cols = tl.arange(0, BLOCK_N)
cols_cmp = cols.to(tl.float32)  # Convert to float32
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

---

## Compilation Optimization Options (Autotune Configuration)

Pass compilation options in `triton.Config`:

```python
def get_autotune_config():
    return [
        triton.Config({'XS': 1 * 128, 'multibuffer': True}),
    ]
```

### Available Options

| Option | Capability | Default |
|--------|-----------|---------|
| `multibuffer` | Enable pipelined parallel data transfer | True |
| `unit_flag` | Cube output optimization | None |
| `limit_auto_multi_buffer_only_for_local_buffer` | CV operator optimization | None |
| `limit_auto_multi_buffer_of_local_buffer` | Cube operator double buffer scope | None, options: `["no-limit", "no-l0c"]` |
| `set_workspace_multibuffer` | Used with the above options | None, e.g., `[2,4]` |
| `enable_hivm_auto_cv_balance` | CV balance optimization | None |
| `tile_mix_vector_loop` | CV operator vector tiling count | None, e.g., `[2,4,8]` |
| `tile_mix_cube_loop` | CV operator cube tiling count | None, e.g., `[2,4,8]` |

**Note**: CV operators refer to operators that use both Cube Core and Vector Core during computation.

---

## Concurrent Task Count Configuration Recommendations

Ascend NPU has multiple compute cores (AI Cores), including Cube Cores (matrix multiplication) and Vector Cores (vector computation).

| Operator Type | Concurrent Task Count Configuration |
|--------------|-------------------------------------|
| Vector-only operators (no `tl.dot`) | Concurrent tasks = Vector Core count |
| Operators with `tl.dot` | Concurrent tasks = AI Core count |

Get physical core count:
```python
from triton.runtime import driver
properties = driver.active.utils.get_device_properties()
```

**Notes**:
- When concurrent task count > physical core count, batch scheduling occurs, incurring additional overhead
- Maximum concurrent task count cannot exceed 65535
