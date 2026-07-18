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


# vLLM Operator Optimization & Integration Pipeline

## Automatic Execution Authorization

**Important**: When the user invokes `/kernel-vllm-optimize`, all of the following operations are considered authorized — **no step-by-step user confirmation needed**:
- Calling MCP tools to optimize operator code
- Reading and editing vLLM project files (activation.py, test_activation.py, benchmark_activation.py)
- Running pytest accuracy tests and benchmark performance tests in the conda environment
- Automatically fixing code and re-running tests if tests fail

The entire pipeline should execute to completion without pausing to ask the user, unless a severe error occurs that cannot be resolved automatically.

---

# vLLM Operator Optimization & Integration Pipeline

Perform MCP iterative optimization on existing Triton operator code, then automatically integrate into the vLLM project (register CustomOp, add accuracy tests, add performance benchmarks).

**Core Principle**: No new operator generation — only optimize existing operators, then integrate into vLLM. **Must record the initial speedup baseline before optimization** to quantify the optimization effectiveness.

## Prerequisites

- `kernelgen-mcp` MCP server must be configured and running
- Python environment with `torch` and `triton` installed (environment and paths are dynamically detected in PHASE 0)
- vLLM project must be installed in editable mode (`pip install -e .`) so that modified operator code takes effect immediately
- vLLM project root directory (dynamically detected in PHASE 0)

## Arguments

- `$ARGUMENTS` - Space-separated parameters:
  1. Operator source directory path (required) — contains existing `*_triton.py`, `*_torch.py`, `test_*.py`, `benchmark_*.py`
  2. Target speedup ratio (optional, default 1.2)
  3. Maximum optimization iterations (optional, default 10)

Automatically extracted from directory/file names:
- **Operator name**: Extracted from `*_triton.py` filename (e.g., `index_put_triton.py` → `index_put`)
- **GPU name**: Extracted from directory name (e.g., `index_put_cc_nvidia` → `nvidia`), defaults to `nvidia` if not found
- **Operator type**: Auto-inferred from name (contains `_and_mul` → `act_and_mul`, otherwise → `act`), user can specify explicitly

Examples:
- `/kernel-vllm-optimize /path/to/relu_cc_nvidia` — Default target 1.2x, max 10 iterations
- `/kernel-vllm-optimize /path/to/softmax_cc_nvidia 1.5` — Target 1.5x, max 10 iterations
- `/kernel-vllm-optimize /path/to/silu_and_mul_cc_huawei 1.3 20` — Target 1.3x, max 20 iterations

---

## vLLM Key Paths

All paths relative to `${VLLM_ROOT}` (dynamically detected in PHASE 0):

| Type | Relative Path |
|------|--------------|
| Operator definition (kernel + wrapper + CustomOp + Registry) | `vllm/model_executor/layers/activation.py` |
| CustomOp base class | `vllm/model_executor/custom_op.py` |
| Accuracy tests | `tests/kernels/core/test_activation.py` |
| Performance benchmark | `benchmarks/kernels/benchmark_activation.py` |

## Naming Conventions

Using `relu` + `huawei` as an example:

| Entity | Naming Format | Example |
|--------|--------------|---------|
| Full operator name | `{base_op_name}_{gpu}` | `relu_huawei` |
| Triton kernel | `_{full_op_name}_kernel` | `_relu_huawei_kernel` |
| Triton wrapper | `{full_op_name}_triton` | `relu_huawei_triton` |
| CustomOp registration name | `{full_op_name}` | `relu_huawei` |
| CustomOp class name | `{OpCamelCase}{GpuCamelCase}` | `ReluHuawei` |
| Benchmark func-name | `{full_op_name}` | `relu_huawei` |

**Class name conversion rules**: Base operator name and GPU identifier each capitalized and concatenated.
- `relu` + `nvidia` → `ReluNvidia`
- `gelu_new` + `huawei` → `GeluNewHuawei`

## Operator Type Determination Rules

- Names containing `_and_mul` suffix → `act_and_mul` type
- Other standalone activation functions → `act` type
- User can specify explicitly; explicit specification takes priority

---

## Workflow

### PHASE 0: Environment Detection & Initialization

**Step 0.0: Dynamic Environment Detection** ⭐

Two things need to be detected: **which Python environment has torch + triton installed**, and **the vLLM project root directory**.

**Step 0.0a: Try Current Python**

```bash
python -c "
import torch, triton, os
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('cuda=' + str(torch.version.cuda))
print('cuda_available=' + str(torch.cuda.is_available()))
"
```

- If successful → `PYTHON_CMD=python`, continue
- If failed (import error) → proceed to Step 0.0b

**Step 0.0b: Scan Conda Environments**

When current Python lacks torch/triton, automatically scan conda environments:

```bash
for env in $(conda env list --json | python -c "import sys,json; print('\n'.join(json.load(sys.stdin)['envs']))"); do
  name=$(basename "$env")
  if conda run -n "$name" python -c "import torch, triton" 2>/dev/null; then
    echo "FOUND_ENV=$name"
    break
  fi
done
```

- Found → `PYTHON_CMD=conda run -n {found_env} python`
- Not found → proceed to Step 0.0c

Re-execute the diagnostic script from 0.0a with the found `${PYTHON_CMD}` to verify the environment.

**Step 0.0c: Auto-create Environment (only if both 0.0a and 0.0b fail)**

No existing environment has torch + triton installed; automatically create a new environment.

**1. Detect CUDA Version**

```bash
nvidia-smi | grep -oP "CUDA Version: \K[\d.]+"
```

If `nvidia-smi` is unavailable, try fallback:
```bash
nvcc --version 2>/dev/null | grep -oP "release \K[\d.]+"
```

**2. Create Conda Environment**

```bash
ENV_NAME="vllm_opt_env"
conda create -n ${ENV_NAME} python=3.10 -y
```

**3. Install PyTorch (select wheel based on CUDA version)**

| CUDA Version | Install Command |
|-------------|----------------|
| 11.8 | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 12.4+ | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Uncertain | Ask the user to provide the correct torch install command |

**4. Install Remaining Dependencies**

```bash
conda run -n ${ENV_NAME} pip install triton pytest numpy
```

**Note**: vLLM editable installation is handled in Step 0.0e (requires locating VLLM_ROOT first).

**5. Verify Installation**

```bash
conda run -n ${ENV_NAME} python -c "
import torch, triton
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('cuda=' + str(torch.version.cuda))
print('cuda_available=' + str(torch.cuda.is_available()))
"
```

- Success → `PYTHON_CMD=conda run -n ${ENV_NAME} python`
- Failure → error and abort: `ERROR: Auto-environment creation failed. Please check the error message and configure manually.`

**Step 0.0d: Locate vLLM Project Root Directory**

Search for directories containing the `vllm` project identifier:

```bash
for candidate in "." "../vllm" "../../vllm" "/share/project/cws/vllm"; do
  if [ -f "$candidate/pyproject.toml" ] && grep -q 'name = "vllm"' "$candidate/pyproject.toml" 2>/dev/null; then
    echo "VLLM_ROOT=$(cd "$candidate" && pwd)"
    break
  fi
done
```

- Found → record `VLLM_ROOT`
- Not found → ask the user: `vLLM project directory not found. Please provide the absolute path to the vLLM repository root.`

Verify key files exist:
```bash
ls ${VLLM_ROOT}/vllm/model_executor/layers/activation.py
ls ${VLLM_ROOT}/vllm/model_executor/custom_op.py
```

**Step 0.0e: Ensure vLLM is Installed in Editable Mode** ⭐

vLLM tests use `from vllm.model_executor.layers.activation import ...`, so the vllm package must be importable and modified code must take effect immediately.

**1. Check if vllm is Already Installed**

```bash
${PYTHON_CMD} -c "
import vllm, os
vllm_init = os.path.abspath(vllm.__file__)
print('VLLM_INSTALLED=' + vllm_init)
"
```

- Success and `vllm.__file__` points to `${VLLM_ROOT}/vllm/__init__.py` → correctly installed (editable mode), continue
- Success but points to another path → warning: `WARNING: vllm install path ({path}) does not match project directory (${VLLM_ROOT}). Modifications may not take effect. Consider reinstalling.`
- Failure (import error) → proceed to step 2 for installation

**2. Editable Install vLLM**

vLLM has C++ extensions; full compilation is slow. Try pure Python installation first (skip C extension compilation):

```bash
cd ${VLLM_ROOT} && VLLM_USE_PRECOMPILED=1 ${PYTHON_CMD} -m pip install -e . 2>&1
```

If `VLLM_USE_PRECOMPILED=1` fails (no precompiled wheel), try without C extension compilation:

```bash
cd ${VLLM_ROOT} && MAX_JOBS=1 ${PYTHON_CMD} -m pip install -e . --no-build-isolation 2>&1
```

If still fails (missing build tools), **fallback to PYTHONPATH approach**:

```bash
export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH}"
```

And verify import works:
```bash
${PYTHON_CMD} -c "import vllm; print('vllm imported from:', vllm.__file__)"
```

If the PYTHONPATH approach also fails → error: `ERROR: Cannot install or import vllm. Please manually run 'cd ${VLLM_ROOT} && pip install -e .'`

**Step 0.0f: Record Environment Variables**

Finalize two variables for all subsequent steps:
- `PYTHON_CMD`: Full command to execute Python (e.g., `python` or `conda run -n xxx python`)
- `VLLM_ROOT`: vLLM project root directory

All subsequent commands follow this pattern:
- `cd ${VLLM_ROOT} && ${PYTHON_CMD} -m pytest ...`
- **Do not** hardcode conda environment names or absolute paths

**Step 0.1: Parse Arguments**

Parse from `$ARGUMENTS`:
- `operator_path`: Operator source directory path
- `target_speedup`: Target speedup ratio (default 1.2)
- `max_iterations`: Maximum optimization iterations (default 10)

Extract from directory/file names:
- `op_name`: Operator name
- `gpu_name`: GPU name (default `nvidia`)
- `full_op_name`: `{op_name}_{gpu_name}`
- `class_name`: CamelCase version of `{full_op_name}`
- `op_type`: `act` or `act_and_mul`
- `func_name`: Exported Python wrapper function name from the Triton file

**Step 0.2: Create Working Copy**

Copy source directory files to a working directory for optimization, avoiding pollution of original files:
```bash
WORK_DIR="<operator_path>_vllm_optimize_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR/iterations"
cp <operator_path>/*_triton.py <operator_path>/*_torch.py <operator_path>/test_*.py <operator_path>/benchmark_*.py "$WORK_DIR/"
# Also try copying *_test.py and *_benchmark.py formats
cp <operator_path>/*_test.py <operator_path>/*_benchmark.py "$WORK_DIR/" 2>/dev/null || true
```

**Step 0.3: Initialize Version Management & Optimization Log**

Maintain the following runtime state (tracked in memory, with file backups):

```
safe_version_code = None       # Latest Triton code that passed correctness tests
safe_version_speedup = None    # Speedup of the safe version
best_version_code = None       # Code with the best historical speedup
best_version_speedup = 0       # Best historical speedup
initial_speedup = None         # ⭐ Pre-optimization baseline speedup

optimization_log = []
# Each record: { "iteration", "action", "correctness", "speedup", "status" }
```

**Step 0.4: Version File Backup Rules**

Whenever safe_version is updated, also back up the code to `$WORK_DIR/iterations/`:
```bash
cp *_triton.py "$WORK_DIR/iterations/triton_iter_{N}_safe.py"
```

---

### PHASE 1: Pre-checks

**Step 1.1: Verify Working Directory Structure**

Check required files in `WORK_DIR`:
- `*_triton.py` — Triton implementation (required)
- `*_torch.py` — PyTorch baseline (required)
- `test_*.py` or `*_test.py` — Test file (required, abort if missing)
- `benchmark_*.py` or `*_benchmark.py` — Performance test (required)

**Step 1.2: Read and Analyze Source Files**

Read all source files, focusing on:

**1.2a: Triton Code Analysis**
- Identify kernel functions and wrapper functions
- Check for known code defects (e.g., function name reference errors, undefined variables)
- Record current BLOCK_SIZE, num_warps, and other configurations
- Determine kernel pattern (Pattern A: Pointwise 1D grid / Pattern B: Row-based 2D grid)

**1.2b: Triton Compatibility Pre-check** ⭐

Before starting optimization, identify the following Triton limitations and record as `dtype_constraints`:

| Feature | Supported dtypes | Unsupported dtypes |
|---------|-----------------|-------------------|
| `tl.atomic_add` | float32, float16, int32 | **bfloat16**, int8, int16, int64, float64 |
| `tl.atomic_max/min` | int32, int64 | All floating-point types |
| `tl.atomic_cas` | int32, int64 | Floating-point types |

**1.2c: Benchmark Review** ⭐

**Must review benchmark code before running benchmarks**, checking for these issues:

1. **Fairness check**: Are PyTorch baseline and Triton implementation measured on the same device (GPU)?
   - Check for `.cpu()` calls causing PyTorch to be measured on CPU while Triton runs on GPU
   - If found, **fix the benchmark immediately** so both use `triton.testing.do_bench` on GPU
2. **Data scale check**: Are the tensor sizes large enough?
   - Too small sizes are dominated by kernel launch overhead and can't show Triton's advantage
   - Recommend at least 1K~16K level update counts
3. **Measurement method check**: Is `triton.testing.do_bench` or `torch.cuda.Event` properly used for GPU timing?
   - `time.perf_counter()` is only suitable for CPU timing, not GPU kernel timing
4. **Warmup check**: Are there enough warmup iterations (recommend ≥10)?

If issues are found, **fix them before PHASE 2** and record the fixes.

---

### PHASE 2: Correctness Verification + Initial Benchmark Baseline (max 3 fix iterations)

**Step 2.1: Run Tests**
```bash
cd $WORK_DIR && ${PYTHON_CMD} -m pytest test_*.py *_test.py -v 2>&1
```

**Step 2.2: Tests Pass → Save as safe_version**

Back up file to `$WORK_DIR/iterations/triton_iter_0_safe.py`.

**Step 2.3: Record Initial Benchmark Baseline** ⭐

**Important**: After correctness passes and before optimization, must run the benchmark first to record the initial speedup.

```bash
cd $WORK_DIR && ${PYTHON_CMD} -m pytest benchmark_*.py *_benchmark.py -v -s 2>&1
```

Parse output, record:
- `initial_speedup` = average speedup across all cases
- `safe_version_speedup` = `initial_speedup`
- `best_version_speedup` = `initial_speedup`

Output initial baseline summary:
```
┌─────────────────────────────────────────┐
│ Initial Benchmark Baseline (Pre-opt)     │
│ Operator: {op_name}                      │
│ Average Speedup: {initial_speedup}x      │
│ Target Speedup: {target_speedup}x        │
└─────────────────────────────────────────┘
```

**Step 2.4: Tests Fail → Call MCP to Fix**

Call `mcp__kernelgen-mcp__optimize_kernel` with `check_result` to fix:
```
Parameters:
- kernel_name: Operator name
- triton_code: Complete code of the current Triton implementation
- check_result: { "success": false, "error": "Error stack trace", "test_case": "Failed test case" }
```

Max 3 fix attempts; if still failing, abort with error `FAILED_CORRECTNESS`.

---

### PHASE 3: Performance Optimization Loop (max {max_iterations} iterations)

**Step 3.1: Run Benchmark**
```bash
cd $WORK_DIR && ${PYTHON_CMD} -m pytest benchmark_*.py *_benchmark.py -v -s 2>&1
```

Parse output, extract PyTorch time, Triton time, and speedup for each input size.
Record current overall speedup, update `safe_version_speedup`.

**Step 3.2: Check Exit Conditions**

- Average speedup ≥ target → update `best_version`, proceed to **PHASE 4**
- Reached `max_iterations` → restore `best_version_code`, proceed to **PHASE 4**
- Current speedup > `best_version_speedup` → update best_version

**Step 3.3: Profiling Analysis (execute on first iteration and key iterations)** ⭐

Before the first optimization, and when speedup shows no improvement for 2 consecutive iterations, perform profiling analysis.

Use `torch.cuda.Event` timing to measure each phase of the wrapper:
1. **Index preprocessing time** (broadcast, flat_idx calculation, etc.)
2. **Data preparation time** (clone, contiguous, dtype conversions, etc.)
3. **Kernel execution time** (pure Triton kernel portion)
4. **Post-processing time** (dtype back-conversion, etc.)

Record as `profiling_breakdown`, passed to MCP in Step 3.4.

**Step 3.4: Build Optimization Context Summary** ⭐

Before calling MCP, **must** build an optimization context summary and pass it to MCP via the `func_desc` parameter:

```
func_desc template:

"Operator: {op_name}
Function: {operator function description}

== Current Optimization State ==
Iteration: #{N} (max {max_iterations})
Current speedup: {current_speedup}x
Target speedup: {target_speedup}x
Best historical speedup: {best_version_speedup}x

== Triton Compatibility Constraints ==
{dtype_constraints content, e.g.: bf16 doesn't support atomic_add, needs upcast to float32}

== Profiling Analysis (if available) ==
| Phase | Time (ms) | Proportion |
|-------|-----------|------------|
| Index Prep | {x}ms | {y}% |
| Data Prep | {x}ms | {y}% |
| Kernel Exec | {x}ms | {y}% |
| Post-process | {x}ms | {y}% |
Bottleneck: {bottleneck}

== Benchmark Details ==
| Input Size | PyTorch (ms) | Triton (ms) | Speedup |
|------------|-------------|-------------|---------|
| {size_1}   | {torch_ms}  | {triton_ms} | {spd}x  |

== Optimization History ==
- Iter {i}: {action}, Result: {status}, Speedup: {speedup}x

== Bottleneck Analysis ==
{Analyze performance bottlenecks based on benchmark and profiling data}

== Optimization Constraints (Important!) ==
- Do not significantly increase kernel complexity or parameter count
- If bottleneck is in wrapper rather than kernel, focus on optimizing wrapper logic
- Must follow the Triton compatibility constraints above
- Optimized code must maintain the same wrapper interface (input/output signatures unchanged)"
```

**Step 3.5: Call MCP to Get Optimized Code**

```
Use MCP tool: mcp__kernelgen-mcp__optimize_kernel

Parameters:
- kernel_name: Operator name
- triton_code: Complete code of current safe_version
- func_desc: Optimization context summary built in Step 3.4
- device: GPU type (nvidia / haiguang / moore / tianshu / huawei)
```

**Step 3.6: MCP Result Usability Check** ⭐

Before writing to file, **must first check if the MCP-returned code is usable**:

1. **Basic syntax check**: Is the code valid Python? Does it contain complete kernel and wrapper functions?
2. **Complexity check**: If the new code's kernel parameter count increases more than 3x, or line count increases more than 5x, consider it over-complex and skip this iteration
3. **Interface compatibility check**: Is the wrapper function's input/output signature consistent with the original version?
4. **Known defect check**: Does it reference undefined variables? Does it use operations marked as unsupported in `dtype_constraints`?

If the check fails:
- Log entry: `{"status": "skipped", "action": "MCP returned unusable code: {reason}"}`
- **Do not write to file**, proceed directly to next iteration
- In the next iteration, explain in func_desc why the previous iteration was skipped

**Step 3.7: Apply Optimized Code**

Write the MCP-returned `triton_code` to the `*_triton.py` file.
Back up the pre-optimization `safe_version_code` (memory + file).

**Important**: If the MCP-returned code is obviously low quality (invalid syntax, unreasonable approach), CC should make independent judgments and manual optimizations rather than blindly adopting the MCP result.

**Step 3.8: Verify Correctness**
```bash
cd $WORK_DIR && ${PYTHON_CMD} -m pytest test_*.py *_test.py -v 2>&1
```

**Case A: Tests Pass**
- Update `safe_version_code`, back up to iterations/, return to Step 3.1

**Case B: Tests Fail → Call MCP to Fix (max 2 attempts)**
```
Parameters:
- kernel_name: Operator name
- triton_code: Current failing code
- func_desc: Optimization context + "Note: The last optimization caused correctness failure. Please fix correctness while maintaining performance optimizations. Triton compatibility constraints: {dtype_constraints}"
- check_result: { "success": false, "error": "Error stack trace", "test_case": "Failed test case" }
```

- Fix successful → status="fixed_then_applied", update safe_version, return to Step 3.1
- Fix failed (both attempts failed) → **Roll back** to safe_version (restore from backup file), status="rolled_back", continue to next iteration

---

### PHASE 4: vLLM Integration

After optimization is complete, integrate the best version of the code into the vLLM project.

#### Step 4.1: Extract and Adapt Triton Kernel + Wrapper

Extract kernel and wrapper from `best_version_code`, rename according to vLLM naming conventions:

- kernel: `_{full_op_name}_kernel` (e.g., `_relu_nvidia_kernel`)
- wrapper: `{full_op_name}_triton` (e.g., `relu_nvidia_triton`)

**Wrapper signature adaptation**:
- In vLLM, wrapper signatures are unified as `(output: torch.Tensor, input: torch.Tensor)` (may have extra parameters like `limit`)
- If the original wrapper signature differs (e.g., return-value style `def xxx(input) -> output`), transform to inplace style

**Must add multi-GPU safety wrapper**:
```python
with torch.cuda.device(input.device):
    # kernel launch
```

#### Step 4.2: Add Code to activation.py

**File**: `${VLLM_ROOT}/vllm/model_executor/layers/activation.py`

First read the current file content, then add at appropriate positions:

**4.2a: Add Triton Kernel + Wrapper**

Add near existing Triton kernel functions in the file. Choose wrapper implementation based on kernel pattern:

**Pattern A — Pointwise (`act` type common)**:
```python
@triton.jit
def _{full_op_name}_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = ...  # Extract kernel logic from best_version_code
    tl.store(output_ptr + offsets, y, mask=mask)

def {full_op_name}_triton(output: torch.Tensor, input: torch.Tensor):
    input_contig = input.contiguous()
    n_elements = input_contig.numel()
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(input.device):
        _{full_op_name}_kernel[grid](input_contig, output, n_elements, BLOCK_SIZE=1024)
```

**Pattern B — Row-based (`act_and_mul` type common)**:
```python
@triton.jit
def _{full_op_name}_kernel(
    o_ptr, o_stride, x_ptr, x_stride,
    d: tl.constexpr, BLOCK_SIZE: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i
    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d
    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)
    result = ...  # Extract kernel logic from best_version_code
    result = result.to(x_ptr.dtype.element_ty)
    tl.store(o_row_ptr + offsets, result, mask=mask)

def {full_op_name}_triton(output: torch.Tensor, input: torch.Tensor):
    b, n = input.shape
    assert input.ndim == 2
    assert n % 2 == 0
    d = n // 2
    def grid(meta):
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))
    with torch.cuda.device(input.device):
        _{full_op_name}_kernel[grid](
            output, output.stride(0),
            input, input.stride(0),
            d=d, BLOCK_SIZE=1024,
        )
```

**Notes**:
- Use the actual kernel logic from best_version_code — the above are structural templates only
- MCP-optimized kernels may have different BLOCK_SIZE or additional parameters; preserve them
- If the kernel has extra constexpr parameters (e.g., `limit`), the wrapper must also receive and pass them

**4.2b: Add CustomOp Class**

Add at the end of existing CustomOp classes (before the Registry dictionary):

**act type (reference ReluMCP)**:
```python
@CustomOp.register("{full_op_name}")
class {class_name}(CustomOp):
    """Generated by kernelgen-mcp for {gpu_name} GPU. Optimized via MCP."""

    def __init__(self):
        super().__init__()

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        # Implement based on PyTorch logic from *_torch.py
        return ...

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        {full_op_name}_triton(out, x)
        return out
```

**act_and_mul type (reference SwigluStepAndMul)**:
```python
@CustomOp.register("{full_op_name}")
class {class_name}(CustomOp):
    """Generated by kernelgen-mcp for {gpu_name} GPU. Optimized via MCP."""

    def __init__(self):
        super().__init__()

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        # Implement based on PyTorch logic from *_torch.py
        gate, up = x[..., :d], x[..., d:]
        return ...

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        {full_op_name}_triton(out, x)
        return out
```

**With constructor parameters**: Reference `SwigluStepAndMul`'s `limit` parameter pattern, adding `__init__` parameters and `extra_repr`.

**4.2c: Register in Registry**

- **act type**: Add to `_ACTIVATION_REGISTRY` LazyDict:
  ```python
  "{full_op_name}": lambda: {class_name}(),
  ```
- **act_and_mul type**: Add to `_ACTIVATION_AND_MUL_REGISTRY` LazyDict:
  ```python
  "{full_op_name}": lambda: {class_name}(),
  ```

**Note**: Read the current file content first, check for existing registrations with the same name (avoid duplicates), and append to existing entries.

#### Step 4.3: Add Accuracy Tests

**File**: `${VLLM_ROOT}/tests/kernels/core/test_activation.py`

First read the current file content, then:

**4.3a: Add Import**
```python
from vllm.model_executor.layers.activation import (
    ...,
    {class_name},
    {full_op_name}_triton,
)
```

**4.3b: Add Tests Based on Operator Type**

**`act` type — 1 modification**:
Append to the `@pytest.mark.parametrize("activation", [...])` list in `test_activation`:
```python
pytest.param(({class_name}, {full_op_name}_triton), id="{class_name}"),
```
No other modifications needed; `test_activation` automatically determines whether to skip opcheck via `hasattr(fn, '_schema')`.

**`act_and_mul` type — 4 modifications**:

1. **parametrize list**: Append `"{full_op_name}"`
2. **elif branch**: Add
   ```python
   elif activation == "{full_op_name}":
       layer = {class_name}()
       fn = {full_op_name}_triton
   ```
3. **Precision tolerance list**: Add `"{full_op_name}"` to the relaxed tolerance `if activation in [...]`
4. **opcheck exclusion**: Add `and activation != "{full_op_name}"` to the exclusion condition

#### Step 4.4: Add Performance Benchmark

**File**: `${VLLM_ROOT}/benchmarks/kernels/benchmark_activation.py`

First read the current file content, then:

**4.4a**: Add `"{full_op_name}"` to the `--func-name` `choices` list

**4.4b**: In most cases, no function body modification needed (the existing `else: layer = op_registry[func_name]()` default branch handles it automatically). Add elif branch when constructor parameters are needed.

---

### PHASE 5: Verification Under vLLM Framework

**Step 5.1: Run Accuracy Tests** (only run current operator's cases)
```bash
cd ${VLLM_ROOT} && ${PYTHON_CMD} -m pytest tests/kernels/core/test_activation.py -v -k "{class_name}"
```

**Step 5.2: Run Performance Benchmark**
```bash
cd ${VLLM_ROOT} && CUDA_VISIBLE_DEVICES=0 ${PYTHON_CMD} benchmarks/kernels/benchmark_activation.py --func-name {full_op_name} --dtype bfloat16
```

**Step 5.3: Handle Failure**

If tests fail under the vLLM framework:
- Check if import paths are correct
- Check if CustomOp registration name matches the Registry
- Check if wrapper signature is in `(output, input)` format
- Check if `cuda:1` tests fail due to missing `torch.cuda.device(input.device)`
- Check if opcheck exclusion is correct (Triton ops don't support opcheck)
- Fix and re-run tests, max 3 fix attempts
- If still failing, report `FAILED_INTEGRATION`, preserve files for manual debugging

---

### PHASE 6: Generate Report

**Step 6.1: Generate `final_report.md`**

Save to working directory `$WORK_DIR/final_report.md`:

```markdown
# vLLM Kernel Optimization & Integration Report

## Summary
| Metric | Value |
|--------|-------|
| Operator | {op_name} |
| GPU | {gpu_name} |
| Op Type | {op_type} |
| Initial Speedup (Baseline) | {initial_speedup}x |
| Final Speedup | {final}x |
| Best Speedup | {best_version_speedup}x |
| Improvement | +{improvement}% |
| Total Iterations | {N} |
| Target Speedup | {target}x |
| Status | {SUCCESS/PARTIAL/FAILED} |

## Optimization History
| Iter | Action | Correctness | Speedup | Status |
|------|--------|-------------|---------|--------|
| ...  | ...    | ...         | ...     | ...    |

## vLLM Integration
| Component | Location |
|-----------|----------|
| Kernel + Wrapper + CustomOp + Registry | vllm/model_executor/layers/activation.py |
| Accuracy Test | tests/kernels/core/test_activation.py |
| Benchmark | benchmarks/kernels/benchmark_activation.py |

## Performance (Best Version)
| Input Size | PyTorch (ms) | Triton (ms) | Speedup |
|------------|--------------|-------------|---------|
| ...        | ...          | ...         | ...     |

## Profiling Breakdown (if available)
| Phase | Time (ms) | Percentage |
|-------|-----------|------------|
| ...   | ...       | ...        |
```

**Step 6.2: Output Summary**

```
============================================================
VLLM OPTIMIZATION & INTEGRATION COMPLETE
============================================================
Operator: {op_name}
GPU: {gpu_name}
Type: {op_type}
Status: {SUCCESS | PARTIAL | FAILED}
Initial Speedup (Baseline): {initial_speedup}x  ← pre-optimization
Final Speedup:              {final_speedup}x     ← post-optimization
Improvement:                +{improvement}%

vLLM Files (under ${VLLM_ROOT}/):
  Activation: vllm/model_executor/layers/activation.py (kernel + wrapper + CustomOp + Registry)
  Test:       tests/kernels/core/test_activation.py
  Benchmark:  benchmarks/kernels/benchmark_activation.py

Verify Commands:
  cd ${VLLM_ROOT}
  ${PYTHON_CMD} -m pytest tests/kernels/core/test_activation.py -v -k "{class_name}"
  CUDA_VISIBLE_DEVICES=0 ${PYTHON_CMD} benchmarks/kernels/benchmark_activation.py --func-name {full_op_name} --dtype bfloat16
============================================================
```

---

## MCP Tool Reference

### mcp__kernelgen-mcp__optimize_kernel

Optimize Triton kernel code. Can be used for performance optimization or error fixing.

**Input Parameters:**
```json
{
  "kernel_name": "Operator name",
  "triton_code": "Current complete Triton code",
  "func_desc": "Operator function description + optimization context summary",
  "check_result": { "success": false, "error": "Error message" },
  "device": "Target chip type"
}
```

**Returns:**
```json
{
  "triton_code": "Optimized/fixed complete code"
}
```

**Usage Scenarios:**
1. **Performance optimization**: Provide `kernel_name`, `triton_code`, `func_desc`
2. **Error fixing**: Additionally provide `check_result`
3. **Fix after optimization**: Provide both `check_result` + `func_desc`

**Important**: MCP-returned code should be reviewed by CC for quality. If the code has obvious issues, CC should optimize it independently or make corrections based on the MCP result.

---

## vLLM Core Mechanism Reference

### CustomOp Registration & Dispatch
1. `@CustomOp.register("name")` decorator registers the class to `op_registry`
2. `CustomOp.__init__` calls `dispatch_forward()` to select the forward method based on platform
3. Benchmark instantiates operators via `op_registry[func_name]()`

### Triton Operator Multi-GPU Safety
All Triton wrappers must wrap kernel launches with `with torch.cuda.device(input.device):` to prevent launching kernels on the wrong GPU in multi-GPU scenarios.

### Difference Between Triton Ops and C Ops in Tests
- C ops: Verified with `opcheck(fn, (out, x))`
- Triton ops: **Cannot use opcheck** (no `_schema` attribute), automatically determined via `hasattr(fn, '_schema')`

### Kernel Pattern Identification
- **Pattern A (Pointwise)**: 1D grid, `n_elements`, `program_id(0)` → common for `act` type
- **Pattern B (Row-based)**: 2D grid, `o_stride/x_stride`, `program_id(0)+program_id(1)` → common for `act_and_mul` type

---

## Quick Reference: Naming Lookup Table

Using `relu` + `huawei` as an example:

| Location | Content |
|----------|---------|
| activation.py — kernel | `def _relu_huawei_kernel(...)` |
| activation.py — wrapper | `def relu_huawei_triton(output, input)` |
| activation.py — class | `@CustomOp.register("relu_huawei")` / `class ReluHuawei(CustomOp)` |
| activation.py — registry | `"relu_huawei": lambda: ReluHuawei()` (in `_ACTIVATION_REGISTRY`) |
| test_activation.py — import | `from ... import ReluHuawei, relu_huawei_triton` |
| test_activation.py — param | `pytest.param((ReluHuawei, relu_huawei_triton), id="ReluHuawei")` |
| benchmark — choices | `"relu_huawei"` |
| benchmark — run command | `--func-name relu_huawei` |

---

## Error Handling

| Scenario | Action |
|----------|--------|
| Python environment missing torch/triton | Automatically scan conda environments or create new environment |
| vLLM project directory not found | Ask user to provide vLLM repository root path |
| CUDA not available | Warn user that tests will fail |
| MCP service unavailable | Error: `ERROR: kernelgen-mcp MCP server not available.` |
| Source directory missing required files | Error with list of missing files, abort |
| PHASE 2 correctness fails after 3 fix attempts | Error: `FAILED_CORRECTNESS`, abort |
| PHASE 5 vLLM verification fails after 3 attempts | Error: `FAILED_INTEGRATION`, preserve files for manual debugging |
| MCP returns unusable code | Skip that iteration, log the reason, continue to next iteration |

## Exit Codes

| Code | Meaning |
|------|---------|
| SUCCESS | Target speedup achieved and vLLM integration verification passed |
| PARTIAL | Speedup improved but did not reach target, vLLM integration completed |
| FAILED_CORRECTNESS | Correctness cannot pass during optimization phase |
| FAILED_INTEGRATION | Optimization succeeded but vLLM integration verification failed |
| FAILED_MCP | MCP service error |
| NO_TEST | Source directory missing test files |
