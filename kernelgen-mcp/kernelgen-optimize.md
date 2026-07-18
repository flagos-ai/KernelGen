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


# Triton Kernel Optimization Skill (MCP Version)

Optimize Triton kernels using expert knowledge from MCP service. Uses the `optimize_kernel` tool from the `kernelgen-mcp` MCP server.

## Prerequisites

- `kernelgen-mcp` MCP server must be configured and running
- Python environment with `torch` and `triton` installed (dynamically detected in PHASE 0)
- If the operator code imports flag_gems, `flag_gems` must also be installed (detected in PHASE 0)

## Arguments

- `$ARGUMENTS` - Space-separated parameters:
  1. Operator directory path (required)
  2. Target speedup ratio (optional, default 1.2)
  3. Maximum optimization iterations (optional, default 10)

Examples:
- `/optimize-triton-mcp /path/to/rot90` — Default target 1.2x, max 10 iterations
- `/optimize-triton-mcp /path/to/rot90 1.5` — Target 1.5x, max 10 iterations
- `/optimize-triton-mcp /path/to/rot90 1.5 20` — Target 1.5x, max 20 iterations

---

## Optimization Workflow

### PHASE 0: Environment Detection & Initialization

**Step 0.0: Dynamically Detect Python Environment** ⭐

**Step 0.0a: Try Current Python**

```bash
python -c "
import torch, triton
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('cuda=' + str(torch.version.cuda))
print('cuda_available=' + str(torch.cuda.is_available()))
"
```

- If successful → `PYTHON_CMD=python`, continue
- If failed (import error) → proceed to Step 0.0b

**Step 0.0b: Scan Conda Environments**

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

**Step 0.0c: Auto-create Environment (only if both 0.0a and 0.0b fail)**

**1. Detect CUDA Version**

```bash
nvidia-smi | grep -oP "CUDA Version: \K[\d.]+"
```

Fallback:
```bash
nvcc --version 2>/dev/null | grep -oP "release \K[\d.]+"
```

**2. Create Conda Environment and Install Base Dependencies**

```bash
ENV_NAME="triton_opt_env"
conda create -n ${ENV_NAME} python=3.10 -y
```

**3. Install PyTorch (select wheel based on CUDA version)**

| CUDA Version | Install Command |
|-------------|----------------|
| 11.8 | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 12.4+ | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Uncertain | Ask the user |

**4. Install Triton and Test Dependencies**

```bash
conda run -n ${ENV_NAME} pip install triton pytest numpy
```

**5. Verify Installation**

```bash
conda run -n ${ENV_NAME} python -c "
import torch, triton
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('cuda_available=' + str(torch.cuda.is_available()))
"
```

- Success → `PYTHON_CMD=conda run -n ${ENV_NAME} python`
- Failure → report error and abort

**Step 0.0d: Check FlagGems Dependency (on demand)** ⭐

Check if operator source code imports flag_gems:

```bash
grep -l "import flag_gems\|from flag_gems" <operator_path>/*.py 2>/dev/null
```

If the operator code depends on flag_gems, ensure it is installed:

```bash
${PYTHON_CMD} -c "import flag_gems; print('flag_gems=' + flag_gems.__version__)" 2>&1
```

- Already installed → continue
- Not installed → install FlagGems following these steps:

**FlagGems Installation Process** (ref: https://github.com/FlagOpen/FlagGems):

1. Locate FlagGems project directory:
```bash
for candidate in "../FlagGems" "../../FlagGems" "../../flagems/FlagGems"; do
  if [ -f "$candidate/pyproject.toml" ] && grep -q "flag.gems" "$candidate/pyproject.toml" 2>/dev/null; then
    echo "FLAGGEMS_ROOT=$(cd "$candidate" && pwd)"
    break
  fi
done
```

2. Install build dependencies:
```bash
${PYTHON_CMD} -m pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

3. Install FlagGems in editable mode (pure Python, no C extension compilation):
```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pip install --no-build-isolation -e .
```

4. Install test dependencies:
```bash
${PYTHON_CMD} -m pip install "pytest>=7.1.0" "numpy>=1.26" "scipy>=1.14" PyYAML sqlalchemy packaging
```

5. Verify installation:
```bash
${PYTHON_CMD} -c "
import flag_gems
print('flag_gems=' + flag_gems.__version__)
print('device=' + str(flag_gems.device))
"
```

If the above steps fail (cannot locate project directory or installation error), ask the user: `FlagGems is not installed. Please provide the FlagGems repository root path, or install it manually and retry.`

**Step 0.0e: Record Environment Variables**

All subsequent commands use `${PYTHON_CMD}` instead of hardcoded `python`.

**Step 0.1: Create Output Directory**
```bash
OUTPUT_DIR="<operator_path>_mcp_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/iterations"
```

**Step 0.2: Initialize Version Management & Optimization Log**

Maintain the following runtime state (tracked in memory, with file backups):

```
# Version management
safe_version_code = None       # Latest Triton code that passed correctness tests
safe_version_speedup = None    # Speedup of the safe version
best_version_code = None       # Code with the best historical speedup
best_version_speedup = 0       # Best historical speedup

# Optimization log (list, appended each iteration)
optimization_log = []
# Each record format:
# {
#   "iteration": 1,
#   "action": "Optimization description, e.g.: Increased BLOCK_SIZE from 128 to 256",
#   "correctness": true/false,
#   "speedup": 1.05 or null (when correctness fails),
#   "status": "applied" / "rolled_back" / "fixed_then_applied"
# }
```

**Step 0.3: Version File Backup Rules**

Whenever safe_version is updated, also back up the code to `$OUTPUT_DIR/iterations/`:
```bash
cp *_triton.py "$OUTPUT_DIR/iterations/triton_iter_{N}_safe.py"
```
This ensures rollback doesn't depend on memory and has reliable file backups.

---

### PHASE 1: Pre-checks

**Step 1.1: Verify Directory Structure**

Check required files:
- `*_triton.py` - Triton implementation (required)
- `*_torch.py` - PyTorch baseline (required)
- `test_*.py` or `*_test.py` - Test file (required, abort if missing)
- `benchmark_*.py` or `*_benchmark.py` - Performance test (required)

**Step 1.2: Read and Analyze Source Files**

Read all files, focusing on the following:

**1.2a: Triton Code Analysis**
- Identify kernel functions and wrapper functions
- Check for known code defects (e.g., function name reference errors, undefined variables)
- Record current BLOCK_SIZE, num_warps, and other configurations

**1.2b: Triton Compatibility Pre-check (Important!)**

Before starting optimization, identify the following Triton limitations and record them to avoid repeated pitfalls:

| Feature | Supported dtypes | Unsupported dtypes |
|---------|-----------------|-------------------|
| `tl.atomic_add` | float32, float16, int32 | **bfloat16**, int8, int16, int64, float64 |
| `tl.atomic_max/min` | int32, int64 | All floating-point types |
| `tl.atomic_cas` | int32, int64 | Floating-point types |

Record these limitations as `dtype_constraints`, passed as constraints in subsequent optimization and MCP calls.

**1.2c: Benchmark Review (Important!)** ⭐

**Must review benchmark code before running benchmarks**, checking for these issues:

1. **Fairness check**: Are PyTorch baseline and Triton implementation measured on the same device?
   - Check for `.cpu()` calls causing PyTorch to be measured on CPU while Triton runs on GPU
   - If found, **fix the benchmark immediately** so both use `triton.testing.do_bench` on GPU
2. **Data scale check**: Are the tensor sizes large enough?
   - Too small sizes (e.g., 64 elements) are dominated by kernel launch overhead and can't show Triton's advantage
   - Recommend at least 1K~16K level update counts
3. **Measurement method check**: Is `triton.testing.do_bench` or `torch.cuda.Event` properly used for GPU timing?
   - `time.perf_counter()` is only suitable for CPU timing, not GPU kernel timing
4. **Warmup check**: Are there enough warmup iterations (recommend ≥10)?

If benchmark issues are found, **fix them before PHASE 2** and record the fixes in the optimization log.

---

### PHASE 2: Correctness Loop (max 3 iterations)

**Step 2.1: Run Tests**
```bash
cd <operator_path> && ${PYTHON_CMD} -m pytest test_*.py -v 2>&1
```

**Step 2.2: Tests Pass → Save as safe_version, proceed to PHASE 3**

Save current code as `safe_version_code` and also set as `best_version_code`.
Back up file to `$OUTPUT_DIR/iterations/triton_iter_0_safe.py`.

**Step 2.3: Tests Fail → Call MCP to Fix**

Call the `mcp__kernelgen-mcp__optimize_kernel` tool with the `check_result` parameter to fix:

```
Use MCP tool: mcp__kernelgen-mcp__optimize_kernel

Parameters:
- kernel_name: Operator name (extracted from filename or directory name)
- triton_code: Complete code of the current Triton implementation
- func_desc: Operator function description (optional)
- check_result: Dictionary containing error information, format:
  {
    "success": false,
    "error": "Error stack trace",
    "test_case": "Description of the failed test case"
  }

Returns:
- triton_code: Fixed Triton code
```

---

### PHASE 3: Performance Optimization Loop (max {max_iterations} iterations, default 10)

**Step 3.1: Run Benchmark**
```bash
cd <operator_path> && ${PYTHON_CMD} benchmark_*.py 2>&1
```

Parse benchmark output, extract PyTorch time, Triton time, and speedup for each input size.
Record current overall speedup, update `safe_version_speedup`.
If this is the first benchmark, also record as `initial_speedup`.

**Step 3.2: Check Exit Conditions**

- Target speedup reached → update `best_version`, proceed to PHASE 4
- Reached `max_iterations` → restore `best_version_code` to file, proceed to PHASE 4
- Current speedup > `best_version_speedup` → update `best_version_code` and `best_version_speedup`

**Step 3.3: Profiling Analysis (execute on first iteration and key iterations)** ⭐

Before the first optimization, and when speedup shows no improvement for 2 consecutive iterations, perform profiling analysis to precisely locate bottlenecks.

**Method**: Insert `torch.cuda.Event` timing points at key positions in the wrapper code, measuring:
1. **Index preprocessing time** (broadcast, flat_idx calculation, and other Python-side operations)
2. **Data preparation time** (clone, contiguous, dtype conversions, etc.)
3. **Kernel execution time** (pure Triton kernel portion)
4. **Post-processing time** (dtype back-conversion, etc.)

Record results as `profiling_breakdown`, format:
```
profiling_breakdown = {
    "index_prep_ms": 0.05,
    "data_prep_ms": 0.08,
    "kernel_ms": 0.01,
    "postprocess_ms": 0.02,
    "total_triton_ms": 0.16,
    "pytorch_baseline_ms": 0.02,
    "bottleneck": "data_prep"  # The phase with the largest proportion
}
```

This profiling result will be passed to MCP in Step 3.4, enabling it to precisely optimize the bottleneck.

**Step 3.4: Build Optimization Context Summary** ⭐

Before calling MCP, you must build a structured optimization context summary and pass it to MCP via the `func_desc` parameter.
This is a critical step to ensure MCP understands the current optimization state.

Summary format:

```
func_desc content template:

"Operator: {op_name}
Function: {brief description of operator function}

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
| ...        | ...         | ...         | ...     |

== Optimization History ==
{Iterate through optimization_log list, listing each entry:}
- Iter {i}: {action}, Result: {status}, Speedup: {speedup}x
- Iter {j}: {action}, Result: rolled_back (correctness failure)
- ...

== Bottleneck Analysis ==
{Analyze based on benchmark and profiling data: whether bottleneck is in wrapper or kernel, compute-bound or memory-bound, etc.}

== Optimization Constraints (Important!) ==
- Do not significantly increase kernel complexity or parameter count
- If bottleneck is in wrapper rather than kernel, focus on optimizing wrapper logic (reduce clone/contiguous/dtype conversions)
- Must follow the Triton compatibility constraints above, do not use unsupported dtype + atomic combinations
- Optimized code must maintain the same wrapper interface (input/output signatures unchanged)"
```

**Step 3.5: Call MCP to Get Optimized Code**

Call MCP tool to get optimized code:

```
Use MCP tool: mcp__kernelgen-mcp__optimize_kernel

Parameters:
- kernel_name: Operator name (required, extracted from filename or directory name, e.g., "relu", "softmax")
- triton_code: Complete code of current safe_version (required)
- func_desc: Optimization context summary built in Step 3.4 (required, including current state, benchmark data, optimization history, constraints)
- gpu: Target chip type (optional), supports:
  - "nvidia" - NVIDIA GPU
  - "haiguang" - Haiguang DCU
  - "moore" - Moore Threads
  - "tianshu" - Tianshu Zhixin
  - "huawei" - Huawei Ascend

Returns:
- triton_code: Fully optimized Triton code
```

The MCP server will, based on the provided context:
1. Understand current optimization progress and historical attempts, avoiding repetition of failed strategies
2. Analyze performance bottlenecks based on benchmark and profiling data
3. Generate targeted optimization code

**Step 3.6: MCP Result Usability Check** ⭐

Before writing to file, **must first check if the MCP-returned code is usable**:

1. **Basic syntax check**: Is the code valid Python? Does it contain complete kernel and wrapper functions?
2. **Complexity check**: Compared to the original code, if the new code's kernel parameter count increases more than 3x, or line count increases more than 5x, consider it over-complex and skip this iteration
3. **Interface compatibility check**: Is the wrapper function's input/output signature consistent with the original version?
4. **Known defect check**: Does it reference undefined variables? Does it use operations marked as unsupported in dtype_constraints?

If the check fails:
- Log entry: `{"iteration": N, "action": "MCP returned unusable code: {reason}", "status": "skipped"}`
- **Do not write to file**, proceed directly to next iteration (if remaining iterations available)
- In the next MCP call, explain in func_desc why the previous iteration was skipped, guiding MCP to generate a more reasonable solution

**Step 3.7: Apply Optimized Code**

Write the MCP-returned `triton_code` to the `*_triton.py` file.
Also back up the pre-optimization `safe_version_code` (memory + file).

**Step 3.8: Verify Correctness**
```bash
cd <operator_path> && ${PYTHON_CMD} -m pytest test_*.py -v 2>&1
```

Handle results by branch:

**Case A: Tests Pass**
- Log optimization: `{"iteration": N, "action": "MCP optimization", "correctness": true, "speedup": null, "status": "pending_benchmark"}`
- Update `safe_version_code` to current code
- Back up file to `$OUTPUT_DIR/iterations/triton_iter_{N}_safe.py`
- → Return to Step 3.1 to run benchmark and fill in speedup data

**Case B: Tests Fail → Call MCP to Fix (max 2 fix attempts)**

Do not roll back immediately! Try letting MCP fix it first:

```
Use MCP tool: mcp__kernelgen-mcp__optimize_kernel

Parameters:
- kernel_name: Operator name
- triton_code: Current failing code (not the rolled-back code)
- func_desc: Optimization context summary + additional note:
  "Note: The last optimization caused correctness failure. Please fix correctness while maintaining performance optimizations.
   Triton compatibility constraints: {dtype_constraints}
   Error type: {key info extracted from error message, e.g., 'bf16 doesn't support atomic_add'}"
- check_result: {
    "success": false,
    "error": "Error stack trace from pytest output",
    "test_case": "Name and parameters of the failed test case"
  }

Returns:
- triton_code: Fixed Triton code
```

After fix, run correctness tests again:
- Fix successful (tests pass) → log status="fixed_then_applied", update `safe_version_code`, return to Step 3.1
- Fix failed (both fix attempts failed) → **Roll back** to `safe_version_code` (restore from backup file), log status="rolled_back", proceed to next iteration

---

### PHASE 4: Generate Report

**Step 4.1: Generate Final Report** `final_report.md`

```markdown
# Triton Kernel Optimization Report (MCP)

## Summary
| Metric | Value |
|--------|-------|
| Operator | {op_name} |
| Initial Speedup | {initial}x |
| Final Speedup | {final}x |
| Best Speedup | {best_version_speedup}x |
| Total Iterations | {N} |
| Status | {SUCCESS/PARTIAL/FAILED} |

## Optimization History
| Iter | Action | Correctness | Speedup | Status |
|------|--------|-------------|---------|--------|
| 1    | {desc}  | PASS        | 1.05x   | applied |
| 2    | {desc}  | FAIL→FIX    | 1.12x   | fixed_then_applied |
| 3    | {desc}  | FAIL        | -       | rolled_back |
| ...  | ...    | ...         | ...     | ... |

## Performance Comparison (Best Version)
| Input Size | PyTorch (ms) | Triton (ms) | Speedup |
|------------|--------------|-------------|---------|
...

## Profiling Breakdown (if available)
| Phase | Time (ms) | Percentage |
|-------|-----------|------------|
| Index Prep | ... | ...% |
| Data Prep | ... | ...% |
| Kernel | ... | ...% |
| Postprocess | ... | ...% |
```

**Step 4.2: Output Summary**
```
============================================================
OPTIMIZATION COMPLETE (MCP)
============================================================
Operator: {op_name}
Status: {SUCCESS | PARTIAL | FAILED}
Initial → Final Speedup: {initial}x → {final}x
Output Directory: {OUTPUT_DIR}
============================================================
```

---

## MCP Tool Reference

### mcp__kernelgen-mcp__optimize_kernel

Optimize Triton kernel code. Can be used for performance optimization or error fixing.

**Input Parameters:**
```json
{
  "kernel_name": "Operator name, e.g., relu, softmax, add, etc.",
  "triton_code": "Complete code of the current Triton implementation",
  "func_desc": "Operator function description + optimization context summary (strongly recommended, see below)",
  "check_result": {
    "success": false,
    "error": "Error message (optional, for fix mode)"
  },
  "gpu": "Target chip type (optional)"
}
```

**Returns:**
```json
{
  "triton_code": "Optimized/fixed complete Triton code"
}
```

**Usage Scenarios:**
1. **Performance optimization**: Provide `kernel_name`, `triton_code`, `func_desc` (with optimization context summary)
2. **Error fixing**: Additionally provide `check_result` containing error information
3. **Fix after optimization**: Provide both `check_result` (correctness error) and `func_desc` (with optimization context), letting MCP fix correctness while maintaining performance optimizations

**About Passing Optimization Context via func_desc (Important):**

In the PHASE 3 optimization loop, `func_desc` is not just an operator function description — it should also include the complete optimization context summary.
This enables MCP to:
- Understand current optimization progress, avoiding repetition of failed strategies
- Locate performance bottlenecks based on benchmark and profiling data
- Follow Triton compatibility constraints, avoiding generation of incompatible code
- Make better decisions based on optimization history

See PHASE 3 Step 3.4 for the detailed format.

---

## Error Handling

If the MCP service is unavailable:
```
ERROR: kernelgen-mcp MCP server not available.
Please ensure the MCP server is configured and running.
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| SUCCESS | Target speedup achieved |
| PARTIAL | Correctness passed, speedup improved but did not reach target |
| FAILED_CORRECTNESS | Still cannot pass tests after 3 iterations |
| FAILED_MCP | MCP service error |
| NO_TEST | Test file missing |
