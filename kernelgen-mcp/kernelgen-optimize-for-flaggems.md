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


# FlagGems Operator Optimization & Integration Pipeline

Supports three optimization modes: built-in operator in-place optimization (PATH A), external directory operator optimization & integration (PATH B), and experimental operator in-place optimization (PATH C).

**Core Principle**: No new operator generation — only optimize existing operators. **Must record the initial speedup baseline before optimization** to quantify the optimization tool's effectiveness.

## Prerequisites

- `kernelgen-mcp` MCP server must be configured and running
- Python environment with `torch`, `triton`, and `flag_gems` installed (environment and paths are dynamically detected in Step 0)

## Arguments

- `$ARGUMENTS` - Space-separated parameters:
  1. **Operator name or source directory path** (required) — operator name (e.g., `relu`, `softmax`), external directory path, or experimental operator filename
  2. Target speedup ratio (optional, default 1.2)
  3. Maximum optimization iterations (optional, default 10)

**Routing Rules**:
- Input is an **existing directory path** → **PATH B** (external directory flow)
- Input matches an existing file under `experimental_ops/` (e.g., `index_put_triton_nvidia_CC`) → **PATH C** (experimental operator in-place optimization)
- Otherwise treated as a **built-in operator name**, check if `src/flag_gems/ops/{op_name}.py` exists:
  - Exists → **PATH A** (built-in operator in-place optimization)
  - Does not exist → Error `NOT_FOUND`

Examples:
- `/kernel-flagGems-optimize relu 1.2 3` — PATH A, built-in operator in-place optimization
- `/kernel-flagGems-optimize /path/to/index_put_cc_nvidia` — PATH B, external directory flow
- `/kernel-flagGems-optimize index_put_triton_nvidia_CC 1.4 5` — PATH C, experimental operator in-place optimization
- `/kernel-flagGems-optimize nonexistent_op` — Error NOT_FOUND

---

## FlagGems Project Path Conventions

All paths are relative to `${FLAGGEMS_ROOT}` (dynamically detected in Step 0):

| Type | Relative Path |
|------|--------------|
| Built-in operators directory | `src/flag_gems/ops/` |
| Experimental operators directory | `src/flag_gems/experimental_ops/` |
| Registration file | `src/flag_gems/experimental_ops/__init__.py` |
| Experimental unit tests | `experimental_tests/unit/` |
| Experimental performance tests | `experimental_tests/performance/` |
| Framework test directory | `tests/` |
| Framework benchmark directory | `benchmark/` |

## Naming Conventions (PATH B/C)

| File Type | Naming Format | Example |
|-----------|--------------|---------|
| Triton operator | `{op_name}_triton_{gpu_name}_CC.py` | `index_put_triton_nvidia_CC.py` |
| Unit test | `{op_name}_test_{gpu_name}_CC.py` | `index_put_test_nvidia_CC.py` |
| Benchmark | `{op_name}_benchmark_{gpu_name}_CC.py` | `index_put_benchmark_nvidia_CC.py` |

---

## Workflow

### PHASE 0: Environment Detection & Route Decision

**Step 0.0: Dynamic Environment Detection** ⭐

Two things need to be detected: **which Python environment has flag_gems installed**, and **the FlagGems project root directory**.

**Step 0.0a: Try Current Python**

```bash
python -c "
import torch, triton, flag_gems, os
fg_file = flag_gems.__file__
flaggems_root = os.path.abspath(os.path.join(os.path.dirname(fg_file), '..', '..'))
print('FLAGGEMS_ROOT=' + flaggems_root)
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('flag_gems=' + flag_gems.__version__)
print('device=' + str(flag_gems.device))
"
```

- If successful → `PYTHON_CMD=python`, extract `FLAGGEMS_ROOT`, continue
- If failed (import error) → proceed to Step 0.0b

**Step 0.0b: Scan Conda Environments**

When current Python lacks flag_gems, automatically scan conda environments:

```bash
for env in $(conda env list --json | python -c "import sys,json; print('\n'.join(json.load(sys.stdin)['envs']))"); do
  name=$(basename "$env")
  if conda run -n "$name" python -c "import flag_gems" 2>/dev/null; then
    echo "FOUND_ENV=$name"
    break
  fi
done
```

- Found → `PYTHON_CMD=conda run -n {found_env} python`
- Not found → proceed to Step 0.0c to auto-create environment

Re-execute the diagnostic script from 0.0a with the found `${PYTHON_CMD}` to extract `FLAGGEMS_ROOT`.

**Step 0.0c: Auto-create Environment (only if both 0.0a and 0.0b fail)**

No existing environment has flag_gems installed; automatically create a new environment and install all dependencies.

**1. Locate FlagGems Project Root Directory**

Search in the following order for a `pyproject.toml` containing `flag.gems` to confirm source location:

```bash
for candidate in "." "../FlagGems" "../../FlagGems"; do
  if [ -f "$candidate/pyproject.toml" ] && grep -q "flag.gems" "$candidate/pyproject.toml" 2>/dev/null; then
    echo "REPO_ROOT=$(cd "$candidate" && pwd)"
    break
  fi
done
```

- Found → record `REPO_ROOT`
- Not found → ask the user: `FlagGems project directory not found. Please provide the absolute path to the FlagGems repository root.`

**2. Detect CUDA Version**

```bash
nvidia-smi | grep -oP "CUDA Version: \K[\d.]+"
```

If `nvidia-smi` is unavailable, try fallback:
```bash
nvcc --version 2>/dev/null | grep -oP "release \K[\d.]+"
```

Extract the CUDA major version (e.g., `11.8`, `12.1`, `12.6`) for selecting the torch wheel.
If neither works, ask the user for the CUDA version.

**3. Create Conda Environment**

```bash
ENV_NAME="flaggems_env"
conda create -n ${ENV_NAME} python=3.10 -y
```

**4. Install PyTorch (select wheel based on CUDA version)**

| CUDA Version | Install Command |
|-------------|----------------|
| 11.8 | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 12.4+ | `conda run -n ${ENV_NAME} pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Uncertain | Ask the user to provide the correct torch install command |

Verify CUDA availability after installation:
```bash
conda run -n ${ENV_NAME} python -c "import torch; print('cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

- `torch.version.cuda` is `None` → CPU-only build, prompt user to reinstall with CUDA support
- `torch.cuda.is_available()` is `False` → CUDA compiled but no GPU, warn user that tests will fail

**5. Install Remaining Dependencies**

```bash
conda run -n ${ENV_NAME} pip install triton
conda run -n ${ENV_NAME} pip install -e ${REPO_ROOT}
conda run -n ${ENV_NAME} pip install pytest numpy scipy pyyaml packaging
```

If `pip install -e ${REPO_ROOT}` fails, check whether `${REPO_ROOT}` contains `pyproject.toml` or `setup.py`.
If not, error: `ERROR: ${REPO_ROOT} is not a valid FlagGems project directory.`

**6. Verify Installation**

```bash
conda run -n ${ENV_NAME} python -c "
import torch, triton, flag_gems, os
fg_file = flag_gems.__file__
flaggems_root = os.path.abspath(os.path.join(os.path.dirname(fg_file), '..', '..'))
print('FLAGGEMS_ROOT=' + flaggems_root)
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('flag_gems=' + flag_gems.__version__)
print('device=' + str(flag_gems.device))
"
```

- Success → `PYTHON_CMD=conda run -n ${ENV_NAME} python`, extract `FLAGGEMS_ROOT`
- Failure → error and abort: `ERROR: Auto-environment creation failed. Please check the error message and configure manually.`

**Step 0.0d: Record Environment Variables**

Finalize two variables for all subsequent steps:
- `PYTHON_CMD`: Full command to execute Python (e.g., `python` or `conda run -n xxx python`)
- `FLAGGEMS_ROOT`: FlagGems project root directory

All subsequent commands follow this pattern:
- `cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest ...`
- **Do not** hardcode conda environment names or absolute paths

**Step 0.1: Parse Arguments**

Parse from `$ARGUMENTS`:
- `first_arg`: First argument (operator name or directory path)
- `target_speedup`: Target speedup ratio (default 1.2)
- `max_iterations`: Maximum optimization iterations (default 10)

**Step 0.2: Route Decision**

```
IF first_arg is an existing directory:
    → PATH B (external directory flow, jump to PHASE 0B)
ELSE:
    # Check if it's an existing operator under experimental_ops
    # Supports with or without .py suffix
    exp_name = first_arg (strip .py suffix)
    IF file ${FLAGGEMS_ROOT}/src/flag_gems/experimental_ops/{exp_name}.py exists:
        → PATH C (experimental operator in-place optimization, jump to PHASE 0C)
    ELSE:
        op_name = first_arg
        IF file ${FLAGGEMS_ROOT}/src/flag_gems/ops/{op_name}.py exists:
            → PATH A (built-in operator in-place optimization, jump to PHASE 0A)
        ELSE:
            → Error: "NOT_FOUND: '{first_arg}' is neither a valid directory nor found in experimental_ops/ or ops/"
            → Abort
```

---

# ═══════════════════════════════════════════════════════════
# PATH A: Built-in Operator In-place Optimization
# ═══════════════════════════════════════════════════════════

### PHASE 0A: Locate & Backup

**Step 0A.1: Locate Files**

- Operator implementation: `${FLAGGEMS_ROOT}/src/flag_gems/ops/{op_name}.py`
- Read the code, understand the implementation pattern (`@pointwise_dynamic` / raw `@triton.jit` / code generation / hybrid)

**Step 0A.2: Verify pytest Mark Availability**

Check correctness tests:
```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest tests/ -m {op_name} --collect-only 2>&1
```
- If selected > 0 → `TEST_CMD_FLAG = -m`
- If selected == 0 → fallback `-k {op_name}`
- If both are 0 → error `NO_TEST`, abort

Similarly check benchmarks:
```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest benchmark/ -m {op_name} --collect-only 2>&1
```
- Found → `HAS_BENCHMARK = true`, record `BENCH_CMD_FLAG`
- Not found → `HAS_BENCHMARK = false`, warn

**Step 0A.3: Create Backup**

```bash
WORK_DIR="${FLAGGEMS_ROOT}/_optimize_backup/{op_name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"
cp ${FLAGGEMS_ROOT}/src/flag_gems/ops/{op_name}.py "$WORK_DIR/{op_name}_original.py"
```

**Step 0A.4: Initialize Version Management**

```
safe_version_code = None
safe_version_speedup = None
best_version_code = None
best_version_speedup = 0
initial_speedup = None          # ⭐ Pre-optimization baseline speedup
optimization_log = []
```

---

### PHASE 2A: Correctness Verification + Initial Benchmark Baseline ⭐

**Step 2A.1: Run Correctness Tests**
```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest tests/ {TEST_CMD_FLAG} {op_name} -v 2>&1
```

- Pass → `safe_version_code` = current code
- Fail → error `ORIGINAL_FAILED`, abort

**Step 2A.2: Record Initial Benchmark Baseline** ⭐

**Important**: Before any optimization, must run the benchmark first to get the initial speedup. This data serves as the baseline for evaluating optimization effectiveness.

```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest benchmark/ {BENCH_CMD_FLAG} {op_name} -v -s 2>&1
```

Parse output, record:
- `initial_speedup` = average speedup across all cases
- `initial_benchmark_details` = detailed data for each case (input size, torch/triton time, speedup)
- `best_version_speedup` = `initial_speedup`

Output initial baseline summary:
```
┌─────────────────────────────────────────┐
│ Initial Benchmark Baseline (Pre-opt)     │
│ Operator: {op_name}                      │
│ Average Speedup: {initial_speedup}x      │
│ Min Speedup: {min_speedup}x              │
│ Max Speedup: {max_speedup}x              │
│ Target Speedup: {target_speedup}x        │
└─────────────────────────────────────────┘
```

---

### PHASE 3A: Performance Optimization Loop (max {max_iterations} iterations)

**Step 3A.1: Run Benchmark**

```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest benchmark/ {BENCH_CMD_FLAG} {op_name} -v -s 2>&1
```

**Step 3A.2: Check Exit Conditions**

- Average speedup >= target → proceed to PHASE 5A
- Reached max_iterations → restore best_version_code, proceed to PHASE 5A
- Current speedup > best_version_speedup → update best_version

**Step 3A.3: Build Optimization Context Summary**

```
func_desc template:

"Operator: {op_name}
Function: {operator function description}
Mode: FlagGems built-in operator (in-place optimization)

== Initial Baseline ==
Pre-optimization speedup: {initial_speedup}x

== FlagGems Framework Info ==
This operator uses FlagGems framework APIs; optimizations must maintain compatibility.

== Current Optimization State ==
Iteration: #{N} (max {max_iterations})
Current speedup: {current_speedup}x (improvement over baseline: {improvement}%)
Target speedup: {target_speedup}x

== Benchmark Details ==
| Input Size | PyTorch (ms) | Triton (ms) | Speedup |
...

== Optimization History ==
...

== Bottleneck Analysis ==
..."
```

**Step 3A.4: Call MCP to Get Optimized Code**

```
MCP tool: mcp__kernelgen-mcp__optimize_kernel
Parameters: kernel_name, triton_code, func_desc, device: nvidia
```

**Step 3A.5: Apply Optimized Code** → write back to ops file

**Step 3A.6: Verify Correctness**
- Pass → update safe_version, return to 3A.1
- Fail → MCP fix (max 2 attempts), roll back if fix fails

---

### PHASE 5A: Final Verification

Correctness + performance verification. Restore backup if failed.

---

### PHASE 6A: Generate Report

The report must include:
- **Initial Speedup (pre-optimization baseline)**
- Final Speedup
- Improvement = (final - initial) / initial * 100%

```
============================================================
FLAGGEMS IN-PLACE OPTIMIZATION COMPLETE
============================================================
Operator: {op_name}
Status: {SUCCESS | PARTIAL | FAILED}
Initial Speedup (Baseline): {initial_speedup}x  ← pre-optimization
Final Speedup:              {final_speedup}x     ← post-optimization
Improvement:                +{improvement}%

Modified File: src/flag_gems/ops/{op_name}.py
Backup: _optimize_backup/{op_name}_{timestamp}/{op_name}_original.py
============================================================
```

---

# ═══════════════════════════════════════════════════════════
# PATH B: External Directory Operator Optimization & Integration
# ═══════════════════════════════════════════════════════════

### PHASE 0B: Initialization

**Step 0B.1: Parse Arguments**

Parse `operator_path`, `target_speedup`, `max_iterations` from `$ARGUMENTS`.
Extract `op_name`, `gpu_name`, `func_name` from directory/file names.

**Step 0B.2: Create Working Copy**

```bash
WORK_DIR="<operator_path>_flaggems_optimize_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR/iterations"
cp <operator_path>/*.py "$WORK_DIR/"
```

**Step 0B.3: Initialize Version Management**

```
safe_version_code = None
safe_version_speedup = None
best_version_code = None
best_version_speedup = 0
initial_speedup = None          # ⭐ Pre-optimization baseline
optimization_log = []
```

---

### PHASE 1B: Pre-checks

Verify required files (`*_triton.py`, `*_torch.py`, `test_*.py`, `benchmark_*.py`), read and understand source files.

---

### PHASE 2B: Correctness Verification + Initial Benchmark Baseline ⭐

**Step 2B.1: Run Tests** (max 3 fix attempts)
```bash
cd $WORK_DIR && ${PYTHON_CMD} -m pytest test_*.py -v 2>&1
```

**Step 2B.2: Record Initial Benchmark Baseline** ⭐

**Important**: After correctness passes and before optimization, must run the benchmark first to record the initial speedup.

```bash
cd $WORK_DIR && ${PYTHON_CMD} -m pytest benchmark_*.py -v -s 2>&1
```

Record `initial_speedup`, output initial baseline summary:
```
┌─────────────────────────────────────────┐
│ Initial Benchmark Baseline (Pre-opt)     │
│ Operator: {op_name}                      │
│ Average Speedup: {initial_speedup}x      │
│ Target Speedup: {target_speedup}x        │
└─────────────────────────────────────────┘
```

---

### PHASE 3B: Performance Optimization Loop (max {max_iterations} iterations)

Same optimization loop logic as PATH A, but operating on `*_triton.py` files in the working directory.

---

### PHASE 4B: FlagGems Integration

**Step 4B.1**: Place Triton operator → `${FLAGGEMS_ROOT}/src/flag_gems/experimental_ops/{op_name}_triton_{gpu_name}_CC.py`
**Step 4B.2**: Register in `__init__.py` (check to avoid duplicates)
**Step 4B.3**: Transform and place unit test → `${FLAGGEMS_ROOT}/experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py`
**Step 4B.4**: Transform and place benchmark → `${FLAGGEMS_ROOT}/experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py`

---

### PHASE 5B: Verification Under FlagGems Framework

```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py -v 2>&1
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py -v 2>&1
```

---

### PHASE 6B: Generate Report

The report must include comparison between the initial baseline speedup and the final speedup.

```
============================================================
FLAGGEMS OPTIMIZATION & INTEGRATION COMPLETE
============================================================
Operator: {op_name} | GPU: {gpu_name}
Status: {SUCCESS | PARTIAL | FAILED}
Initial Speedup (Baseline): {initial_speedup}x  ← pre-optimization
Final Speedup:              {final_speedup}x     ← post-optimization
Improvement:                +{improvement}%

FlagGems Files:
  Op:        src/flag_gems/experimental_ops/{op_name}_triton_{gpu_name}_CC.py
  Test:      experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py
  Benchmark: experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py
============================================================
```

---

# ═══════════════════════════════════════════════════════════
# PATH C: Experimental Operator In-place Optimization (NEW)
# ═══════════════════════════════════════════════════════════

Perform in-place optimization on existing experimental operators under `experimental_ops/`.
Directly use existing tests in `experimental_tests/unit/` and `experimental_tests/performance/`.

### PHASE 0C: Locate & Backup

**Step 0C.1: Parse Operator Information**

Parse from `first_arg` (e.g., `index_put_triton_nvidia_CC`):
- `exp_file_name`: Full filename (without .py)
- `op_name`: Base operator name (e.g., `index_put`) — extracted by removing `_triton_{gpu}_CC` from filename
- `gpu_name`: GPU name (e.g., `nvidia`) — extracted from filename

Corresponding file paths:
- Operator file: `${FLAGGEMS_ROOT}/src/flag_gems/experimental_ops/{exp_file_name}.py`
- Unit test: `${FLAGGEMS_ROOT}/experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py`
- Performance test: `${FLAGGEMS_ROOT}/experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py`

**Step 0C.2: Verify File Existence**

Check if all three files exist:
```bash
ls ${FLAGGEMS_ROOT}/src/flag_gems/experimental_ops/{exp_file_name}.py
ls ${FLAGGEMS_ROOT}/experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py
ls ${FLAGGEMS_ROOT}/experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py
```

- Operator file missing → error `NOT_FOUND`
- Test file missing → error `NO_TEST: Test file for {exp_file_name} not found`

**Step 0C.3: Read and Understand Code**

Read operator implementation, unit test, and benchmark code to understand:
- Operator function and implementation approach
- Kernel structure and potential optimization opportunities
- Test case coverage

**Step 0C.4: Create Backup**

```bash
WORK_DIR="${FLAGGEMS_ROOT}/_optimize_backup/{exp_file_name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"
cp ${FLAGGEMS_ROOT}/src/flag_gems/experimental_ops/{exp_file_name}.py "$WORK_DIR/{exp_file_name}_original.py"
```

**Step 0C.5: Initialize Version Management**

```
safe_version_code = None
safe_version_speedup = None
best_version_code = None
best_version_speedup = 0
initial_speedup = None          # ⭐ Pre-optimization baseline
optimization_log = []
```

---

### PHASE 2C: Correctness Verification + Initial Benchmark Baseline ⭐

**Step 2C.1: Run Correctness Tests**

```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py -v 2>&1
```

- Pass → `safe_version_code` = current code
- Fail → error `ORIGINAL_FAILED`, abort

**Step 2C.2: Record Initial Benchmark Baseline** ⭐

```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py -v 2>&1
```

Parse output, record:
- `initial_speedup` = average speedup across all cases
- `initial_benchmark_details` = detailed data for each case

Output initial baseline summary:
```
┌─────────────────────────────────────────┐
│ Initial Benchmark Baseline (Pre-opt)     │
│ Operator: {exp_file_name}                │
│ Average Speedup: {initial_speedup}x      │
│ Min Speedup: {min_speedup}x              │
│ Max Speedup: {max_speedup}x              │
│ Target Speedup: {target_speedup}x        │
└─────────────────────────────────────────┘
```

---

### PHASE 3C: Performance Optimization Loop (max {max_iterations} iterations)

**Step 3C.1: Run Benchmark**

```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py -v 2>&1
```

Parse output, record speedup. First run is same as `initial_speedup` (already recorded in 2C.2).

**Step 3C.2: Check Exit Conditions**

- Average speedup >= target → proceed to PHASE 5C
- Reached max_iterations → restore best_version_code and write back to operator file, proceed to PHASE 5C
- Current speedup > best_version_speedup → update best_version

**Step 3C.3: Build Optimization Context Summary**

```
func_desc template:

"Operator: {op_name} (experimental operator)
File: experimental_ops/{exp_file_name}.py
Function: {operator function description}

== Initial Baseline ==
Pre-optimization speedup: {initial_speedup}x

== Current Optimization State ==
Iteration: #{N} (max {max_iterations})
Current speedup: {current_speedup}x (improvement over baseline: {improvement}%)
Target speedup: {target_speedup}x
Best historical speedup: {best_version_speedup}x

== Benchmark Details ==
| Test Case | PyTorch (ms) | Triton (ms) | Speedup |
|-----------|-------------|-------------|---------|
| ...       | ...         | ...         | ...     |

== Optimization History ==
- Iter {i}: {action}, Result: {status}, Speedup: {speedup}x
...

== Bottleneck Analysis ==
{Analyze performance bottlenecks based on benchmark data}"
```

**Step 3C.4: Call MCP to Get Optimized Code**

```
MCP tool: mcp__kernelgen-mcp__optimize_kernel
Parameters:
- kernel_name: {op_name}
- triton_code: Complete code of the current operator file
- func_desc: Optimization context summary above
- device: {gpu_name}
```

**Step 3C.5: Apply Optimized Code**

**Directly write back** the MCP-returned code to `${FLAGGEMS_ROOT}/src/flag_gems/experimental_ops/{exp_file_name}.py`.

**Important**: MCP-returned code should be reviewed by CC for quality. If the code has obvious issues, CC should optimize it independently.

**Step 3C.6: Verify Correctness**

```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py -v 2>&1
```

- Pass → update safe_version, return to 3C.1
- Fail → MCP fix (max 2 attempts), roll back to safe_version if fix fails, continue to next iteration

---

### PHASE 5C: Final Verification

**Step 5C.1: Correctness Verification**
```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py -v 2>&1
```

**Step 5C.2: Performance Verification**
```bash
cd ${FLAGGEMS_ROOT} && ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py -v 2>&1
```

**Step 5C.3: Handle Failure**

If final verification fails:
- Restore backup: `cp "$WORK_DIR/{exp_file_name}_original.py" ${FLAGGEMS_ROOT}/src/flag_gems/experimental_ops/{exp_file_name}.py`
- Error `FAILED_FINAL_VERIFY`

---

### PHASE 6C: Generate Report

**Step 6C.1: Generate `final_report.md`**

```markdown
# FlagGems Experimental Op Optimization Report

## Summary
| Metric | Value |
|--------|-------|
| Operator | {exp_file_name} |
| GPU | {gpu_name} |
| Initial Speedup (Baseline) | {initial_speedup}x |
| Final Speedup | {final_speedup}x |
| Best Speedup | {best_version_speedup}x |
| Improvement | +{improvement}% |
| Total Iterations | {N} |
| Target Speedup | {target}x |
| Status | {SUCCESS/PARTIAL/FAILED} |

## Optimization History
| Iter | Action | Correctness | Speedup | vs Baseline | Status |
|------|--------|-------------|---------|-------------|--------|
| 0    | Baseline | ✓         | {init}x | —           | baseline |
| 1    | ...    | ...         | ...     | +{x}%       | ...    |
| ...  | ...    | ...         | ...     | ...         | ...    |

## Performance Comparison (Before vs After)
| Test Case | Before (ms) | After (ms) | Before Speedup | After Speedup | Change |
|-----------|-------------|------------|----------------|---------------|--------|
| ...       | ...         | ...        | ...            | ...           | ...    |
```

**Step 6C.2: Output Summary**

```
============================================================
FLAGGEMS EXPERIMENTAL OP OPTIMIZATION COMPLETE
============================================================
Operator: {exp_file_name}
GPU: {gpu_name}
Status: {SUCCESS | PARTIAL | FAILED}

Performance:
  Initial Speedup (Baseline): {initial_speedup}x  ← pre-optimization
  Final Speedup:              {final_speedup}x     ← post-optimization
  Improvement:                +{improvement}%

Modified File:
  src/flag_gems/experimental_ops/{exp_file_name}.py

Backup:
  _optimize_backup/{exp_file_name}_{timestamp}/{exp_file_name}_original.py

Verify Commands:
  cd ${FLAGGEMS_ROOT}
  ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_{gpu_name}_CC.py -v
  ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_{gpu_name}_CC.py -v

Restore Original:
  cp _optimize_backup/{exp_file_name}_{timestamp}/{exp_file_name}_original.py src/flag_gems/experimental_ops/{exp_file_name}.py
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

**Important**: MCP-returned code should be reviewed by CC for quality. If the code has obvious issues (invalid syntax, unreasonable approach), CC should optimize it independently or make corrections based on the MCP result.

---

## Error Handling

| Scenario | Action |
|----------|--------|
| MCP service unavailable | Error: `ERROR: kernelgen-mcp MCP server not available.` |
| Python environment missing dependencies | Error: `ERROR: Python environment missing required dependencies` |
| Operator not found in any known location | Error: `NOT_FOUND` |
| PATH A original code fails correctness | Error: `ORIGINAL_FAILED` |
| PATH A no test cases | Error: `NO_TEST` |
| PATH B source directory missing required files | Error with list of missing files |
| PATH C no corresponding test file | Error: `NO_TEST` |
| Correctness fails after 3 fix attempts | Error: `FAILED_CORRECTNESS` |
| FlagGems verification fails | Error: `FAILED_INTEGRATION` (PATH B) or `FAILED_FINAL_VERIFY` (PATH A/C) |

## Exit Codes

| Code | Meaning |
|------|---------|
| SUCCESS | Target speedup achieved and verification passed |
| PARTIAL | Speedup improved but did not reach target, verification passed |
| FAILED_CORRECTNESS | Correctness cannot pass during optimization phase |
| FAILED_INTEGRATION | PATH B: Optimization succeeded but FlagGems integration verification failed |
| FAILED_FINAL_VERIFY | PATH A/C: Final verification failed, original code restored |
| FAILED_MCP | MCP service error |
| NOT_FOUND | Operator not found in any known location |
| ORIGINAL_FAILED | Original code fails correctness tests |
| NO_TEST | No corresponding test cases found |
