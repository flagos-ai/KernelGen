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


# KernelGen Skill — Generate FlagGems Operators via MCP

You are an expert at generating GPU kernel operators for the FlagGems project using the `kernelgen-mcp` MCP service.

**Tool usage**: This skill relies on the following capabilities:
- **Shell**: execute shell commands (python, pip, pytest, etc.) via the Bash/shell tool
- **Read**: read files from the codebase
- **Write / Edit**: create or modify files
- **Grep**: search file contents
- **Glob**: find files by pattern
- **MCP tools**: `mcp__kernelgen-mcp__generate_kernel` and `mcp__kernelgen-mcp__optimize_kernel`

> **MCP Prerequisite Check**: If the user has not configured the kernelgen MCP service (i.e., MCP tools are unavailable or calls fail),
> immediately prompt the user to visit https://kernelgen.flagos.io/ to register and obtain the kernelgen MCP service URL and JWT Token,
> then complete the configuration following Step 0b instructions before retrying. Do not proceed with subsequent steps if MCP is not ready.

When you need user input or clarification, ask a question directly and wait for their reply.
Always use the appropriate built-in tool rather than outputting commands for the user to run manually.

## Execution Rules

1. **Follow the steps in order**. Never jump ahead.
2. **Never skip Step 2** (operator existence check) — always verify before generating.
3. **Do not generate or write any code before Step 4** (MCP call). Steps 0–3 are preparation only.
4. **Always confirm with the user before destructive actions**.
5. **Report progress to the user when entering each step**.
6. **Never fabricate repository files or paths**.
7. **CRITICAL — MCP is mandatory**: ALL operator code generation MUST go through the
   `mcp__kernelgen-mcp__generate_kernel` MCP tool. NEVER generate Triton kernels, PyTorch
   wrappers, or operator implementations yourself — even if the operator seems simple (e.g.,
   relu, abs). If MCP is not configured, not reachable, or fails after all retries in Step 4,
   STOP and report the issue to the user. Do NOT fall back to writing kernel code manually.
   The MCP service produces optimized, tested code that manual writing cannot match.

---

## FlagGems Project Layout Reference

The FlagGems project has three operator locations and corresponding test/benchmark locations:

### Operator Locations
| Location | Path | Description |
|---|---|---|
| **Core (src)** | `src/flag_gems/ops/<kernel_name>.py` | Mature, production-quality operators. Uses `pointwise_dynamic` decorators for pointwise ops. Registered in `_FULL_CONFIG` for aten dispatch. |
| **Experimental** | `src/flag_gems/experimental_ops/<kernel_name>.py` | New/experimental operators. Uses raw Triton pointer-based style (self-contained, no `pointwise_dynamic`). NOT registered in `_FULL_CONFIG`. |
| **Backend** | `src/flag_gems/runtime/backend/_<vendor>/ops/<kernel_name>.py` | Non-NVIDIA chip-specific operators. Vendor names: `_ascend` (Huawei), `_cambricon`, `_hygon` (DCU), `_mthreads` (Moore), `_iluvatar` (Tianshu), `_metax`, `_amd`, `_kunlunxin`, `_arm`, `_aipu`. |

### Test Locations
| Operator Location | Accuracy Tests | Performance Tests |
|---|---|---|
| **Core (src)** | `tests/test_<category>_ops.py` (append to existing file) | `benchmark/test_<category>_perf.py` (append to existing file) |
| **Experimental** | `experimental_tests/<kernel_name>_test.py` (standalone file) | Same file as accuracy test (includes benchmark function) |
| **Backend** | Backend-specific test directory (follow existing pattern per vendor) | Backend-specific benchmark directory |

### Test Style Conventions (CRITICAL)

**`tests/` folder style** — uses `flag_gems.use_gems()` context manager and calls ops via `torch.<op>()`:
```python
import flag_gems
from .accuracy_utils import gems_assert_close, to_reference, POINTWISE_SHAPES, FLOAT_DTYPES

def test_accuracy_<kernel_name>(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = <torch_call>(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.<kernel_name>(inp)
    gems_assert_close(res_out, ref_out, dtype)
```

**`experimental_tests/` folder style** — uses direct import from `flag_gems.experimental_ops`:
```python
import flag_gems
from flag_gems.experimental_ops.<kernel_name> import <kernel_name> as gems_op

def test_<kernel_name>_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.<kernel_name>(ref_x)
    with flag_gems.use_gems():
        act_out = gems_op(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)
```

**IMPORTANT**: Before writing a test, always read 1-2 existing test files in the **same folder** to match the exact style (imports, utility functions, assertion helpers, parametrize decorators, etc.). Never mix styles between folders.

### Decorator Conventions

**Core ops (`src/flag_gems/ops/`)** use FlagGems decorators:
- `@pointwise_dynamic(promotion_methods=...)` + `@triton.jit` — for pointwise ops
- The kernel receives **scalar elements** (not pointers), no `tl.load`/`tl.store`
- See `src/flag_gems/ops/abs.py`, `src/flag_gems/ops/sigmoid.py` for examples

**Experimental ops (`src/flag_gems/experimental_ops/`)** use raw Triton style:
- `@triton.jit` only — standard Triton pointer-based kernels
- The kernel receives **pointers**, uses `tl.load`/`tl.store`, `BLOCK_SIZE`, masks
- Self-contained, no `flag_gems.utils` imports needed
- See `src/flag_gems/experimental_ops/abs.py` for example

**When adapting MCP output**: The MCP always generates raw pointer-based Triton code.
- For **core ops**: MUST rewrite to `pointwise_dynamic` style — strip all pointer logic
- For **experimental ops**: Keep the MCP Triton code mostly as-is (it's already pointer-based)
- For **backend ops**: Follow the existing backend vendor patterns

---

## Step 0: Pre-flight — Environment & MCP Check

### 0a. Check Python Environment

Run the following diagnostic command:

```bash
python -c "import torch; import triton; import flag_gems; print('torch', torch.__version__); print('triton', triton.__version__); print('flag_gems', flag_gems.__version__); print('device', flag_gems.device)"
```

**If any import fails**, identify what's missing and use the shell tool to install it:

| Missing package | Install command |
|---|---|
| `torch` | Do NOT auto-install. Ask the user for the correct install command, as the CUDA wheel variant depends on their environment. |
| `triton` | `pip install triton` |
| `flag_gems` | `pip install -e .` from the repo root, or `pip install flag_gems` |
| `pytest` | `pip install pytest` |
| `numpy` | `pip install numpy` |
| `scipy` | `pip install scipy` |
| `pyyaml` | `pip install pyyaml` |
| `packaging` | `pip install packaging` |

If `torch` is missing or has no CUDA support, run a separate GPU diagnostic:

```bash
python -c "import torch; print('cuda_version:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available())"
```

Then diagnose:
- If `torch.version.cuda` is `None` → torch is a CPU-only build. Ask the user to reinstall with CUDA.
- If `torch.version.cuda` has a value but `torch.cuda.is_available() == False` → CUDA build but no GPU
  detected. Warn the user that GPU tests will fail and ask how to proceed.

If `flag_gems` is not installed, first verify `pyproject.toml` or `setup.py` exists in the current
working directory (to confirm we're at the repo root). If found, run `pip install -e .`. If not
found, ask the user for the correct repo root path.

After installing, re-run the diagnostic via the shell tool to confirm everything works. Only proceed when all imports succeed.

### 0b. Check MCP Availability

Verify that the `kernelgen-mcp` MCP server is configured and reachable.

Use the Read tool to read `.claude/settings.json` and look for an `mcpServers` entry whose key
contains `kernelgen` (case-insensitive). If the file does not exist, treat it as "MCP not configured".

**If the MCP server is NOT configured**, stop and print the following message to the user, then
wait for the user to provide the URL and JWT token:

```
The kernelgen-mcp service is not yet configured. Please follow these steps:

1. Visit https://kernelgen.flagos.io/ to register and obtain a JWT Token.
2. Add the following MCP configuration to `.claude/settings.json` (replace <YOUR_URL> and <YOUR_JWT_TOKEN> with actual values):

{
  "mcpServers": {
    "kernelgen-mcp": {
      "type": "sse",
      "url": "<YOUR_URL>",
      "headers": {
        "Authorization": "Bearer <YOUR_JWT_TOKEN>"
      }
    }
  }
}

3. After configuration is complete, please re-run this command.
```

After the user provides the URL and JWT, use the Edit tool (or Write tool if the file doesn't exist)
to write the configuration into `.claude/settings.json`, merging with any existing content. Then proceed.

**If the MCP server IS configured**, proceed to Step 1.

## Step 1: Understand the Operator Request

Parse the user's description to determine:
- `kernel_name`: operator name (e.g., `relu`, `softmax`, `gelu`, `layer_norm`).
  Prefer names exactly as defined in `torch.ops.aten` (e.g., `relu`, `silu`, `neg`),
  since registration uses the aten name.
- `torch_call`: the actual Python call to invoke this op on a tensor. This is NOT always
  `torch.<kernel_name>`. Determine it as follows:
  1. If `torch.<kernel_name>` exists (`hasattr(torch, '<kernel_name>')` is True) → use `torch.<kernel_name>`
  2. If not, check `torch.nn.functional.<kernel_name>` (e.g., `torch.nn.functional.layer_norm`)
  3. If not, check `torch.special.<kernel_name>` or `torch.linalg.<kernel_name>`
  4. As a last resort, use `torch.ops.aten.<kernel_name>.default`
  Do NOT use `torch._C._nn` — it is an internal API that can break across PyTorch versions.
  **Mandatory extra arguments**: If the chosen `torch_call` requires mandatory arguments beyond the
  input tensor(s), you MUST include them with sensible defaults. Common cases:
  - `torch.nn.functional.softmax(x, dim=-1)`
  - `torch.nn.functional.log_softmax(x, dim=-1)`
  - `torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])`
  - `torch.nn.functional.dropout(x, p=0.5, training=False)`
  - `torch.nn.functional.normalize(x, dim=-1)`
  When in doubt, check the existing tests in the repo or the PyTorch docs for required arguments.
  Omitting mandatory arguments (e.g., `torch.nn.functional.softmax(x)` without `dim`) will cause
  runtime errors.
  Store `<torch_call>` (including any mandatory extra arguments) — it will be used in tests and
  benchmarks wherever the reference op is needed.
- `func_desc`: what the operator does
- `func_type`: one of the following categories (aligned with FlagGems test structure).
  **Auto-infer from the operator signature when possible:**
  - If 1 tensor input, no dim arg → `unary_pointwise`
  - If operator has >=2 tensor inputs, no dim arg → `binary_pointwise`
  - If has a `dim` argument → `reduction`
  - If name contains `norm`, `softmax` (also matches `log_softmax`), `rmsnorm`, or `batchnorm` → `normalization`
  - If name is `mm`, `matmul`, `bmm`, `addmm` → `blas`
  - Otherwise → `other`
  Confirm the inferred `func_type` with the user if ambiguous. Categories:
  - `unary_pointwise` — single-input elementwise ops (relu, sigmoid, abs, etc.)
  - `binary_pointwise` — two-input elementwise ops (add, mul, etc.)
  - `reduction` — ops that reduce dimensions (sum, mean, max, etc.)
  - `normalization` — norm ops (layer_norm, batch_norm, group_norm, etc.)
  - `blas` — matrix operations (matmul, mm, bmm, etc.)
  - `other` — everything else (indexing, special, etc.)
- `arg_names`, `arg_type`, `arg_descs`, `output_arg_desc`: parameter information
- `target_device`: determine target device from user context:
  - If user mentions NVIDIA/英伟达/nv or doesn't specify → `"nvidia"` (default)
  - If user mentions 华为/昇腾/Ascend → `"huawei"`
  - If user mentions 海光/Hygon/DCU → `"haiguang"`
  - If user mentions 摩尔/Moore/摩尔线程 → `"moore"`
  - If user mentions 天数/Tianshu/天数智芯 → `"tianshu"`
  - If user mentions 寒武纪/Cambricon → `"cambricon"`

If the operator name is ambiguous, first check this common alias table before asking the user:

| User says | Correct `kernel_name` (torch.ops.aten) |
|---|---|
| swish | `silu` |
| negative / negate | `neg` |
| power | `pow` |
| square | `pow` (exp=2) |
| cube | `pow` (exp=3) |
| absolute | `abs` |
| multiply | `mul` |
| divide | `div` |
| clip | `clamp` |
| hard_swish | `hardswish` |
| hard_sigmoid | `hardsigmoid` |
| logarithm / ln | `log` |
| exponential | `exp` |
| hyperbolic_tangent | `tanh` |
| square_root | `sqrt` |
| cube_root | `cbrt` |
| inverse_sqrt | `rsqrt` |
| reciprocal / inverse | `reciprocal` |
| minimum | `min` |
| maximum | `max` |
| floor_divide | `floor_divide` |
| modulo / mod | `remainder` |

If the alias is not in this table, ask the user to clarify.

After determining `func_type`, derive the `<category>` value used for test/benchmark file paths:

| func_type | category (for file paths) |
|---|---|
| `unary_pointwise` | `unary_pointwise` |
| `binary_pointwise` | `binary_pointwise` |
| `reduction` | `reduction` |
| `normalization` | `norm` |
| `blas` | `blas` |
| `other` | `special` |

Store this `<category>` value — it determines which test and benchmark files to modify.

## Step 1.5: Determine Placement (NEW — CRITICAL DECISION)

Based on the user's request and the operator analysis, determine `<placement>`:

### Decision Rules

1. **Backend placement** (`backend`): If `target_device` is NOT `"nvidia"` (i.e., non-NVIDIA chip), the operator goes into the backend directory:
   - Operator: `src/flag_gems/runtime/backend/_<vendor>/ops/<kernel_name>.py`
   - Tests: Follow the existing backend vendor test pattern (read existing tests in that vendor's directory first)

2. **Core placement** (`core`): If the user explicitly asks to **modify/replace an existing core operator** in `src/flag_gems/ops/`, or the user specifically says "add to src/core":
   - Operator: `src/flag_gems/ops/<kernel_name>.py`
   - Tests: `tests/test_<category>_ops.py` (append to existing file)
   - Benchmarks: `benchmark/test_<category>_perf.py` (append to existing file)

3. **Experimental placement** (`experimental`) — **DEFAULT for new operators**:
   - New operators should go here unless the user explicitly requests core placement
   - Operator: `src/flag_gems/experimental_ops/<kernel_name>.py`
   - Tests: `experimental_tests/<kernel_name>_test.py` (standalone file)
   - Benchmarks: Included in the same test file

### Ask the user if ambiguous

If the placement is not obvious from context, ask:
- "This is a new operator. I recommend placing it in `experimental_ops/` (experimental). Would you prefer to place it in `src/flag_gems/ops/` (core) instead?"
- If the user mentions a non-NVIDIA chip, confirm: "I'll place this in the `_<vendor>` backend directory. Is that correct?"

Store `<placement>` as one of: `core`, `experimental`, `backend`.

## Step 2: Check Whether the Operator Already Exists

Before calling the MCP generator, **thoroughly search** the codebase for existing implementations:

1. **Core ops**: Use the Glob tool to check for `src/flag_gems/ops/<kernel_name>.py`
2. **Experimental ops**: Use the Glob tool to check for `src/flag_gems/experimental_ops/<kernel_name>.py`
3. **Backend ops**: If `<placement>` is `backend`, check `src/flag_gems/runtime/backend/_<vendor>/ops/<kernel_name>.py`
4. **Registration**: Use the Grep tool to search `src/flag_gems/ops/__init__.py` and `src/flag_gems/__init__.py` for the op name
5. **Aten registration**: Use the Grep tool to search for `torch.ops.aten.<kernel_name>` in the codebase (some registrations use the aten op reference instead of string names)
6. **Tests**: Use the Grep tool to search `tests/test_*_ops.py` and `experimental_tests/<kernel_name>_test.py` for the op name
7. **Benchmarks**: Use the Grep tool to search `benchmark/test_*_perf.py` for the op name in `forward_operations`

### If the operator already exists

Present findings to the user and ask them to choose one of the following:

**Option A — Skip generation**: The operator already exists; do nothing.

**Option B — Replace existing**: Overwrite the current implementation with MCP-generated code
(adapted to FlagGems conventions). The file stays in its **current location** (core stays in core, experimental stays in experimental).

**Option C — Create experimental variant (side-by-side)**: Generate the operator under the same
name in `experimental_ops/` so it coexists with the original. This will:
  - Create `src/flag_gems/experimental_ops/<kernel_name>.py`
  - Create a standalone test in `experimental_tests/<kernel_name>_test.py`
  - Include a perf benchmark in that same test file
  - Do **NOT** register it in `_FULL_CONFIG` (it won't override the aten dispatch)

Only proceed to Step 3 after the user has made a choice.

### If the operator does NOT exist

Proceed directly to Step 3. Use `<placement>` determined in Step 1.5.

## Step 3: Research Context (flagos_wiki)

Before calling the MCP generator, use the Read tool to gather reference materials that improve generation quality:

1. **Read similar operator code** from the codebase. Choose based on `<placement>`:
   - **Core placement** — read from `src/flag_gems/ops/`:
     - Unary pointwise → read `src/flag_gems/ops/abs.py` or `src/flag_gems/ops/sigmoid.py`
     - Binary pointwise → read `src/flag_gems/ops/add.py`
     - Reduction → read `src/flag_gems/ops/sum.py` or `src/flag_gems/ops/mean.py`
     - Normalization → read `src/flag_gems/ops/layer_norm.py`
     - BLAS → read `src/flag_gems/ops/mm.py`
   - **Experimental placement** — read from `src/flag_gems/experimental_ops/`:
     - Read `src/flag_gems/experimental_ops/abs.py` or a similar experimental op
   - **Backend placement** — read existing ops in the target vendor directory

2. **Read the test pattern** from the matching test location:
   - **Core**: read `tests/test_<category>_ops.py` — note the import style, utility functions, parametrize patterns
   - **Experimental**: read an existing `experimental_tests/<some_op>_test.py` — note the direct import style, `to_reference()`, `GenericBenchmark` usage
   - **Backend**: read existing tests in the backend vendor directory

3. **Note the decorators used** in similar operators:
   - Core pointwise ops use `@pointwise_dynamic(promotion_methods=...)` + `@triton.jit`
   - Experimental ops use `@triton.jit` only (raw pointer-based)
   - Some ops may use other decorators like `@triton.autotune` — note these

4. **If replacing an existing op** (Option B), read the current implementation so the new version
   can be compared / improved upon.

5. Collect all findings and **summarize into concise notes** (not full file contents) to pass as
   the `flagos_wiki` parameter. For example:
   - `"abs.py uses pointwise_dynamic with DEFAULT promotion, returns tl.abs(x)"`
   - `"sigmoid.py uses INT_TO_FLOAT promotion, returns 1/(1+tl.exp(-x))"`
   - `"test pattern: @pytest.mark.xxx, POINTWISE_SHAPES, FLOAT_DTYPES, gems_assert_close"`
   - `"experimental test style: from flag_gems.experimental_ops.abs import abs as gems_abs, uses torch.ops.aten.abs for reference"`

   If the `mcp__kernelgen-mcp__generate_kernel` tool supports a `flagos_wiki` (or similarly named
   `context`/`references`) parameter, pass the collected notes. If the MCP call fails with an
   error containing `unexpected argument`, `unknown field`, `invalid schema`, or similar, retry
   the call without the `flagos_wiki` field.

## Step 4: Call kernelgen-mcp

Invoke `mcp__kernelgen-mcp__generate_kernel` with the parameters gathered above, including the
`flagos_wiki` list for reference context.

If `<placement>` is `backend` and target device is not nvidia, pass the `device` parameter to the
MCP call (e.g., `device="huawei"` for Ascend).

**Set the iteration counter**: `iteration_count = 1`. This tracks total MCP calls for the final report.

The MCP returns four code blocks:
- `torch_code` — PyTorch reference implementation
- `triton_code` — Triton kernel implementation
- `test_func_code` — accuracy test code
- `benchmark_func_code` — performance benchmark code

## Step 4.5: MCP Report Review

If the MCP response includes test results (accuracy test reports, benchmark/speedup reports),
present them to the user **in full** before proceeding to local code adaptation:

```
=== MCP Generation Test Report ===

Accuracy Test Results:
<paste the COMPLETE accuracy test output from the MCP response — do not summarize or truncate>

Performance Benchmark Results:
<paste the COMPLETE benchmark/speedup output from the MCP response — do not summarize or truncate>
```

Then ask the user:

> The MCP service has returned the above test reports from its generation environment.
> Would you like to:
>
> **A — Skip local testing**: Trust the MCP reports and proceed directly to code placement
> and summary. No local tests will be run.
>
> **B — Run local tests**: Run accuracy and performance tests on your local machine as well
> to verify the results in your specific environment.

Wait for the user's choice before proceeding:
- If **A (skip local testing)**: After placing code in Step 5, skip Steps 6 and 7 (local
  accuracy tests and benchmarks) and proceed directly to Step 8 (Summary). Use the MCP
  report numbers for the summary.
- If **B (run local tests)**: Proceed normally through Steps 5 → 6 → 7 → 8. Before
  running local tests, perform the **Chip Compatibility Check** described in Step 5.5c.

**If the MCP response does NOT include test reports** (only code blocks), proceed normally
to Step 5 — local testing will be required.

## Step 5: Adapt and Place Code into the FlagGems Project

This is the most critical step. The generated code must be transformed to match FlagGems conventions.
The exact placement and transformation depends on `<placement>`.

---

### Placement: Core (`src/flag_gems/ops/`)

#### 5a. Operator Implementation → `src/flag_gems/ops/<kernel_name>.py`

**Transformation rules for pointwise ops**: The MCP generator will almost always output raw
pointer-based Triton code. For core pointwise ops, you MUST **always** rewrite it into the
`pointwise_dynamic` elementwise style:
- The `@triton.jit` kernel function receives **scalar elements** (not pointers)
- `pointwise_dynamic` handles all pointer arithmetic, tiling, and masking automatically
- The kernel only contains the elementwise math logic (e.g., `return tl.maximum(x, 0)`)
- **Remove ALL** `tl.load()`, `tl.store()`, pointer arithmetic (`ptr + offsets`), mask logic,
  and BLOCK_SIZE parameters — these are incompatible with `pointwise_dynamic`
- Only keep the pure math expression that transforms input element(s) to output element(s)

Examples of correct `pointwise_dynamic` kernels (keep ONLY the scalar `tl.*` math):
```python
# relu: return tl.maximum(x, 0)
# exp:  return tl.exp(x)
# silu: return x * tl.sigmoid(x)
# add:  return x + y
```

Do NOT keep `offsets`, `mask`, `n_elements`, `BLOCK_SIZE`, or any tiling logic.

**CRITICAL**: The MCP Triton code MUST NOT be copied directly for core ops. Always rewrite it
following FlagGems `pointwise_dynamic` conventions. Never trust the raw MCP code — it will
almost always be pointer-style which is incompatible with `pointwise_dynamic`.

Use the Write tool (new file) or Edit tool (replacing existing) to create the operator file.

**Unary pointwise template:**
```python
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=<PROMOTION_METHODS>)
@triton.jit
def <kernel_name>_forward(x):
    return ...


def <kernel_name>(A):
    logger.debug("GEMS <KERNEL_NAME> FORWARD")
    return <kernel_name>_forward(A)
```

**Binary pointwise template:**
```python
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=<PROMOTION_METHODS>)
@triton.jit
def <kernel_name>_forward(x, y):
    return ...


def <kernel_name>(A, B):
    logger.debug("GEMS <KERNEL_NAME> FORWARD")
    return <kernel_name>_forward(A, B)
```

Choose the correct template based on `func_type`. For non-pointwise ops (reduction, BLAS,
normalization), do NOT use either template — instead read a similar existing op and follow
its pattern.

**For in-place variants**: Read an existing in-place operator (e.g., `relu.py`) to find the
exact call pattern used in this repo, then follow it. Do NOT assume the in-place parameter
name — it may be `out0=A`, `out=A`, or another convention. Always verify by reading an existing
in-place operator first.

Key conventions:
- Use `pointwise_dynamic` decorator from `flag_gems.utils` for pointwise ops
- Use `triton.jit` decorator
- The kernel function takes raw tensor elements (not pointers) — `pointwise_dynamic` handles the boilerplate
- **Unary ops**: kernel has one parameter `(x)`, e.g. `def relu_forward(x)`
- **Binary ops**: kernel has two parameters `(x, y)`, e.g. `def add_forward(x, y)`
- Promotion methods — **always read a similar existing operator first** and copy its promotion
  method. If no similar operator exists, use these guidelines as fallback:
  - `INT_TO_FLOAT`: ops that always produce float output (exp, log, sigmoid, tanh, sqrt, etc.)
  - `COMPLEX_TO_FLOAT`: ops that reduce complex to real (abs)
  - `DEFAULT`: ops that preserve input dtype (relu, neg, add, mul, clamp, etc.)
  - **Template placeholder** `<PROMOTION_METHODS>` must expand to the full list:
    - Unary ops: `[(0, "DEFAULT")]` or `[(0, "INT_TO_FLOAT")]`
    - Binary ops: `[(0, "DEFAULT"), (1, "DEFAULT")]` or `[(0, "INT_TO_FLOAT"), (1, "INT_TO_FLOAT")]`
  - **When in doubt, prefer reading the repo** — promotion behavior is subtle and repo-specific.
- The wrapper function takes `A` as the input tensor parameter name (NOT `self`)
- Include `logger.debug("GEMS <NAME> FORWARD")` in the wrapper
- For non-pointwise ops (reduction, BLAS, normalization, etc.), follow the specific pattern of similar existing ops — do NOT force the `pointwise_dynamic` pattern on them

#### 5b. Register the Operator

Use the Edit tool to make these changes:

1. **`src/flag_gems/ops/__init__.py`**: Add import and `__all__` entry (in alphabetical order):
   ```python
   from flag_gems.ops.<kernel_name> import <kernel_name>, <kernel_name>_
   ```

2. **`src/flag_gems/__init__.py`**: Add to `_FULL_CONFIG` tuple (in alphabetical order).
   **IMPORTANT**: First read the existing `_FULL_CONFIG` entries to match the exact registration
   format used by this repo. Different versions may use different styles:
   - String style: `("relu", relu)`
   - Aten op style: `(torch.ops.aten.relu.default, relu)`
   Copy the pattern from existing entries. Do NOT guess the format.
   Insert the new entry in **alphabetical order** — do not change the order of existing entries.

#### 5c. Accuracy Test → `tests/test_<category>_ops.py`

Do NOT create a new test file. Use the Edit tool to add the test function to the appropriate existing test file:
- Unary pointwise: `tests/test_unary_pointwise_ops.py`
- Binary pointwise: `tests/test_binary_pointwise_ops.py`
- Reduction: `tests/test_reduction_ops.py`
- BLAS: `tests/test_blas_ops.py`
- Norm: `tests/test_norm_ops.py`
- Special: `tests/test_special_ops.py`

**IMPORTANT — Match existing style**: Before writing the test, **read at least 2 existing test
functions** in the target file. Ensure you match:
- Import style (from `.accuracy_utils import ...`)
- The `flag_gems.use_gems()` context manager pattern
- Assertion function (`gems_assert_close` or `gems_assert_equal`)
- `to_reference()` usage pattern
- Parametrize decorators (`POINTWISE_SHAPES`, `FLOAT_DTYPES`, etc.)

Follow the existing test pattern:
```python
@pytest.mark.<kernel_name>
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_<kernel_name>(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = <torch_call>(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.<kernel_name>(inp)

    gems_assert_close(res_out, ref_out, dtype)
```

Key conventions:
- Use `flag_gems.device` not `"cuda"`
- Use `to_reference()` to create reference tensors
- Use `flag_gems.use_gems()` context manager for the FlagGems call
- Use `gems_assert_close` or `gems_assert_equal` for comparison
- Mark with `@pytest.mark.<kernel_name>`
- If the marker `<kernel_name>` is not already registered, find where markers are defined
  (check `pytest.ini`, `pyproject.toml`, or `setup.cfg`) and add it.
- Use `POINTWISE_SHAPES`, `FLOAT_DTYPES` etc. from `accuracy_utils`

#### 5d. Performance Benchmark → `benchmark/test_<category>_perf.py`

Do NOT create a new benchmark file. Read the target benchmark file first, then use the Edit tool
to append the new tuple into the existing `forward_operations` list (find the list definition and
insert in alphabetical order). Follow the exact tuple format already used in the file.

---

### Placement: Experimental (`src/flag_gems/experimental_ops/`)

This is the **default for new operators**.

#### 5a. Operator Implementation → `src/flag_gems/experimental_ops/<kernel_name>.py`

Use the Write tool to create the file. Use the **raw Triton pointer-based style** (standard for experimental ops).
Keep the MCP-generated Triton code mostly as-is but ensure it:
- Is self-contained (no `flag_gems.utils` imports needed, no `pointwise_dynamic`)
- Has proper `@triton.jit` kernel with pointer args, mask, BLOCK_SIZE
- **Has a Python wrapper function** that computes the grid and launches the kernel
- Exports the wrapper function named `<kernel_name>` (and optionally `<kernel_name>_out`)

**Ensure all kernel meta parameters (BLOCK_SIZE, etc.) are passed** — either via
`@triton.autotune` or as explicit keyword arguments.

Example structure (reference `src/flag_gems/experimental_ops/abs.py`):
```python
import torch
import triton
import triton.language as tl


@triton.jit
def _<kernel_name>_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = ...  # elementwise computation
    tl.store(out_ptr + offsets, y, mask=mask)


def <kernel_name>(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _<kernel_name>_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```

#### 5b. Registration (limited)

For experimental ops, do **NOT** modify `src/flag_gems/__init__.py` or `_FULL_CONFIG`.
Optionally add an entry to `src/flag_gems/experimental_ops/__init__.py` if needed.

#### 5c. Standalone Test + Benchmark → `experimental_tests/<kernel_name>_test.py`

Use the Write tool to create a single self-contained test file.

**IMPORTANT — Match experimental_tests/ style**: Before writing the test, **read 1-2 existing
test files** in `experimental_tests/` (e.g., `abs_test.py`, `sigmoid_test.py`). Match their exact style:
- Direct import: `from flag_gems.experimental_ops.<kernel_name> import <kernel_name> as gems_<kernel_name>`
- Reference call: `torch.ops.aten.<kernel_name>(ref_x)` (not `torch.<kernel_name>`)
- `to_reference()` with `TO_CPU` support
- `GenericBenchmark` from `benchmark.performance_utils` for perf tests
- `triton.testing.do_bench` for inline benchmarks

Follow this pattern (adapted from existing experimental tests):
```python
# <KERNEL_NAME> operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.<kernel_name> import <kernel_name> as gems_<kernel_name>

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.<kernel_name>
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_<kernel_name>_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.<kernel_name>(ref_x)
    with flag_gems.use_gems():
        act_out = gems_<kernel_name>(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.<kernel_name>
def test_perf_aten_<kernel_name>():
    def <kernel_name>_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    bench = GenericBenchmark(
        input_fn=<kernel_name>_input_fn,
        op_name="<kernel_name>",
        torch_op=torch.ops.aten.<kernel_name>,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
```

---

### Placement: Backend (`src/flag_gems/runtime/backend/_<vendor>/`)

#### 5a. Operator Implementation

Read existing operators in `src/flag_gems/runtime/backend/_<vendor>/ops/` to understand the
exact patterns used by this vendor. Follow the same style.

#### 5b. Registration

Follow the vendor's existing registration pattern (check `__init__.py` in the vendor directory).

#### 5c. Tests

Follow the vendor's existing test patterns. Read existing test files in the vendor's test
directory before writing new ones.

---

## Step 5.5: Pre-test Validation

Before running the full test suite, perform two quick checks:

### 5.5a. Lint / Format Check

If the repo uses linting tools (ruff, black, isort, flake8), run them on the changed files to
catch style issues early. Use the shell tool:

```bash
# Adjust path based on <placement>
python -m ruff check <operator_file_path> --fix
python -m ruff format <operator_file_path>
```

If `ruff` is not installed, check for other formatters in `pyproject.toml` or `setup.cfg` and
use those. If no linter is configured, skip this step.

### 5.5b. Triton Compile Smoke Test

Triton kernel compile errors only surface on first invocation (JIT compilation).

**For Core placement:**
```bash
python -c "
import torch, flag_gems
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for shape in [(1,), (4,), (128,), (4, 128), (0,)]:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device) if shape == (0,) else torch.randn(shape, dtype=dtype, device=flag_gems.device)
        with flag_gems.use_gems():
            out = <SMOKE_TEST_CALL>  # e.g. torch.relu(inp)
        print(f'Smoke test passed: {dtype}, shape={shape}')
"
```

**For Experimental placement:**
```bash
python -c "
import torch, flag_gems
from flag_gems.experimental_ops.<kernel_name> import <kernel_name>
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for shape in [(1,), (4,), (128,), (4, 128), (0,)]:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device) if shape == (0,) else torch.randn(shape, dtype=dtype, device=flag_gems.device)
        out = <kernel_name>(inp)
        print(f'Smoke test passed: {dtype}, shape={shape}')
"
```

The `(0,)` shape tests the empty-tensor edge case (numel==0), which often causes kernel crashes.

If this fails, read the error and fix the kernel before proceeding to Step 6.

### 5.5c. Chip Compatibility Check (before local testing)

Before running local tests, verify that the local hardware matches the operator's
`target_device` (determined in Step 1). Use the Bash tool:

```bash
python - <<'PY'
import subprocess
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"nvidia: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA device available")
except Exception:
    print("torch not available")
try:
    r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
    if r.returncode == 0 and "NPU" in r.stdout:
        print("huawei: Ascend NPU detected")
except Exception:
    pass
try:
    r = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
    if r.returncode == 0:
        kind = "haiguang" if "hygon" in r.stdout.lower() else "amd"
        print(f"{kind}: ROCm device detected")
except Exception:
    pass
PY
```

Compare detected hardware with `target_device`:

| Target Device | Required Local Hardware |
|---|---|
| `nvidia` (default) | NVIDIA GPU |
| `huawei` | Huawei Ascend NPU (`npu-smi`) |
| `haiguang` | Hygon DCU (`rocm-smi` + Hygon) |
| `moore` | Moore Threads GPU |
| `tianshu` | Tianshu GPU |
| `cambricon` | Cambricon MLU |

**If the local hardware does NOT match the target device**, warn the user:

```
⚠️ Hardware mismatch detected:
  - Operator target device: <target_device>
  - Local hardware: <detected_hardware>

Running tests for a <target_device> operator on <detected_hardware> hardware will likely
fail or produce incorrect results. Please choose:

  A — Skip local testing: Do not run local tests; rely on MCP test reports (if available).
  B — Proceed anyway: Run local tests despite the mismatch (results may be unreliable).
```

If the user chooses A, skip Steps 6 and 7 and proceed to Step 8 (Summary).
If the user chooses B, proceed with local tests but note the mismatch in the final report.

## Step 6: Run Accuracy Tests

Run the accuracy tests for the newly added operator.
Assume the current working directory is the repository root.

**For Core placement:**
```bash
python -m pytest tests/test_<category>_ops.py -m <kernel_name> -v
```

**For Experimental placement:**
```bash
python -m pytest experimental_tests/<kernel_name>_test.py -v -k "test_<kernel_name>"
```

**For Backend placement:**
```bash
# Use the vendor's test running convention (read existing CI scripts or Makefile first)
python -m pytest <backend_test_path> -m <kernel_name> -v
```

Report the results to the user. If tests fail, follow the **error classification and retry
protocol** below. This protocol strictly limits self-fix attempts to prevent infinite loops.

### Error Classification

First, classify the error into one of two categories:

**Category A — Compilation / Import errors** (model may attempt 1 self-fix):
- `ImportError`, `ModuleNotFoundError` — wrong import path or missing module
- `SyntaxError` — Python syntax issue
- `TritonCompilationError`, `CompilationError` — Triton kernel compile failure
- `NameError` — undefined variable or function
- `TypeError` in function call signature — wrong number/type of arguments
- `AttributeError` — wrong attribute access on a module or object

**Category B — Algorithm / Numerical accuracy errors** (do NOT self-fix):
- `AssertionError` from `torch.testing.assert_close` or `gems_assert_close` — numerical mismatch
- Wrong output values (results don't match reference)
- Shape mismatch in output tensors
- NaN or Inf in output
- Any error that indicates the kernel logic is incorrect

### Retry Protocol (strictly follow this order)

**Step 6a. (Category A only) Self-fix — maximum 1 attempt:**
If the error is Category A (compilation/import), you may attempt exactly ONE fix:
1. Use the Read tool to examine the error traceback carefully.
2. Apply a targeted fix using the Edit tool (e.g., fix import path, fix syntax, fix argument name).
3. Use the shell tool to re-run the tests.
4. If the test **passes** → proceed to Step 7.
5. If the test **still fails** → proceed to Step 6b. Do NOT attempt a second self-fix.

**If the error is Category B (algorithm/accuracy), skip Step 6a entirely** — go directly
to Step 6b. Do NOT attempt to fix algorithm logic yourself, as this typically leads to
an endless fix-retry loop without converging. The MCP service has better optimization
capabilities for these issues.

**Step 6b. MCP re-generation — pass error context to generate_kernel:**
Re-call `mcp__kernelgen-mcp__generate_kernel` with the **same parameters as Step 4**,
but add the error information to `flagos_wiki` as additional hints.
**Increment**: `iteration_count += 1`.
Keep `flagos_wiki` concise — maximum 10 items total. If retrying multiple times, replace
earlier error entries rather than appending, to avoid bloating the prompt.
Replace the kernel code with the new MCP output, re-run tests.
- If tests **pass** → proceed to Step 7.
- If tests **still fail** → proceed to Step 6c.

**Step 6c. MCP optimization — pass error context to optimize_kernel:**
Try `mcp__kernelgen-mcp__optimize_kernel` with the current kernel code and the
`check_result` parameter containing the error traceback.
**Increment**: `iteration_count += 1`.
Replace the kernel code with the optimized output, re-run tests.
- If tests **pass** → proceed to Step 7.
- If tests **still fail** → proceed to Step 6d.

**Step 6d. Stop and report:**
Do not keep retrying. Report the failure to the user with:
- The specific test failures and error messages
- Your analysis of what might be wrong
- Suggestion to try with different `func_type` or additional `flagos_wiki` hints

## Step 7: Run Performance Benchmark

Run the performance benchmark.

**For Core placement:**
```bash
python -m pytest benchmark/test_<category>_perf.py -m <kernel_name> -v
```

**For Experimental placement:**
```bash
python -m pytest experimental_tests/<kernel_name>_test.py -v -k "perf"
```

Look for lines in the output containing keywords like `speedup`, `latency`, `gems`, `torch`, or
timing values. Use regex patterns to extract numbers.

**Do not just copy the raw table** — always compute and report actual speedup ratios.

If a speedup metric is printed directly, extract and report it. Otherwise compute:
`speedup = torch_latency / gems_latency` (>1.0 means gems is faster).

**Note**: Triton kernels are JIT-compiled on first invocation, which adds significant overhead.
If benchmark results look unreasonably slow on the first config, check whether the benchmark
framework includes a warmup phase. If not, the first data point may be an outlier — **if the
first timing sample is >5x larger than the median of remaining samples, exclude it** from the
average speedup calculation and note "first sample excluded (Triton JIT compile overhead)" in
the report.

## Step 8: Summary

Provide a clear summary to the user with **exact numbers** extracted from test and benchmark
output:

```
=== KernelGen Operator Generation Report ===

Operator Name: <kernel_name>
Placement: Core / Experimental / Backend (_<vendor>)
Generation Mode: New / Replace Existing / Experimental Variant
Operator Type: <func_type>
Target Device: <target_device>

File Changes:
  - [New/Modified] <operator_file_path>
  - [New/Modified] <test_file_path>
  - [Modified] <benchmark_file_path> (if applicable)
  - [Modified] src/flag_gems/ops/__init__.py (if core)
  - [Modified] src/flag_gems/__init__.py (if core)

Decorator Style: pointwise_dynamic / raw triton.jit / other
Test Style: use_gems() + torch.<op>() / direct import from experimental_ops

Accuracy Tests: <N> passed, <M> failed (total <N+M> test cases)
  Pass Rate: <N/(N+M)*100>%
  Failed Cases: <list failed test names if any, or "None">

Performance Benchmark:
  Avg Speedup: <X.XX>x vs PyTorch reference
  Best Speedup: <X.XX>x (shape=<S>, dtype=<D>)
  Worst Speedup: <X.XX>x (shape=<S>, dtype=<D>)
  (If benchmark was not run or failed, write "Incomplete" and explain why)

MCP Iterations: <iteration_count> (write 1 if first generation passed)

Performance Analysis:
  <Assess based on average speedup:
   - Speedup > 5.0x → "Extreme fusion success, typical for fused multi-op kernels"
   - Speedup > 3.0x → "Excellent compute-bound optimization, significant kernel fusion benefit"
   - Speedup > 2.0x → "Compute-bound optimization effective, significant speedup achieved"
   - Speedup 1.2x ~ 2.0x → "Moderate speedup, kernel likely balanced between compute and memory"
   - Speedup 0.5x ~ 1.2x → "Kernel likely memory-bound, limited optimization headroom"
   - Speedup < 0.5x → "Kernel slower than PyTorch reference — likely suboptimal memory access pattern">

Issues and Fixes: (if any)
```

**After presenting the summary, check the speedup:**

- **If average speedup < 0.5x** (kernel slower than PyTorch), proactively warn the user.
- **If average speedup is 0.5x ~ 1.2x**, suggest optimization.
- **If average speedup > 1.2x**, no action needed — report the result normally.

## Step 9: Post-Completion Actions

After the summary is presented (whether based on MCP reports or local test results), perform
the following two checks.

### 9a. Chat Tool Notification

If the user's task was received via a chat tool (e.g., Feishu/飞书, Discord, Slack, Teams,
DingTalk/钉钉, WeChat Work/企业微信, Telegram, or any similar messaging platform), or if
the user mentions sending results to a client or team, ask:

> The operator generation is complete. Would you like me to prepare a message to send the
> generated files or summary to your client/team via **<chat_tool_name>**?

If the user confirms:
1. Prepare a concise summary message including: operator name, file paths, accuracy pass
   rate, speedup metrics, and any issues.
2. If the chat tool has a CLI or API integration available in the environment, use it to
   send the message directly.
3. If no CLI/API is available, format the message as copy-paste ready text for the user.
4. If files need to be sent, list the exact file paths the user should attach.

### 9b. Git Repository PR Submission

Check whether the current project is managed by a git-based hosting service:

```bash
git remote -v 2>/dev/null | head -5
```

Detect the hosting platform from the remote URL:
- `github.com` → GitHub
- `gitlab.com` (or self-hosted GitLab) → GitLab
- `gitee.com` → Gitee
- `bitbucket.org` → Bitbucket

**If a git hosting platform is detected**, ask the user:

> This project is hosted on **<platform>**. Would you like me to automatically create a
> Pull Request with the generated operator code?

**If the user confirms**, follow the standard PR creation workflow:

1. Create a new branch: `kernelgen/<kernel_name>`
2. Stage all changed/new files related to this operator generation (never use `git add -A`)
3. Create a commit with a descriptive message
4. Push the branch to the remote with `-u` flag
5. Create the PR using the platform's CLI tool:

   **For GitHub** (`gh`):
   ```bash
   gh pr create --title "Add <kernel_name> operator via KernelGen" --body "$(cat <<'EOF'
   ## Summary
   Add `<kernel_name>` operator (<func_type>) generated via KernelGen MCP service.

   ## Changes
   - [New/Modified] `<kernel_file_path>` — Triton kernel implementation
   - [New/Modified] `<test_file_path>` — Accuracy tests
   - [New/Modified] `<benchmark_file_path>` — Performance benchmarks
   - [Modified] `<registration_files>` — Operator registration (if applicable)

   ## Test Results
   ### Accuracy
   - **Pass rate**: <N>/<total> (<percentage>%)
   - **Failed cases**: <list or "None">

   ### Performance
   - **Avg speedup**: <X.XX>x vs PyTorch reference
   - **Best**: <X.XX>x | **Worst**: <X.XX>x

   ## Generation Details
   - **MCP iterations**: <iteration_count>
   - **Target device**: <target_device>
   - **Operator type**: <func_type>
   - **Placement**: <core / experimental / backend>

   ---
   Generated by KernelGen MCP skill

   EOF
   )"
   ```

   **For Gitee / GitLab**: Provide the PR/MR creation URL and prepared description.

6. Return the PR URL to the user.

**If PR creation fails**, report the error, provide the branch name, and suggest manual PR
creation with the prepared description.

**If the user declines**, do nothing — changes remain as local files.

## Important Notes

- **`kernel_name` must match the `torch` API name** (e.g., `silu` not `swish`, `neg` not `negate`).
  The accuracy test calls `torch.<kernel_name>(...)`, so a mismatch will cause test failures.
- **Never overwrite existing operators** without explicit user permission (Step 2 choice)
- **Placement determines decorator style**:
  - Core ops → `pointwise_dynamic` for pointwise, follow existing patterns for others
  - Experimental ops → raw `@triton.jit` pointer-based style
  - Backend ops → follow vendor-specific patterns
- **Placement determines test style**:
  - Core (`tests/`) → `flag_gems.use_gems()` + `torch.<op>()`
  - Experimental (`experimental_tests/`) → direct `from flag_gems.experimental_ops.<op> import <op>`
  - Backend → vendor-specific test patterns
- **Default placement for new operators is experimental** — only use core if user explicitly requests it
- **Follow alphabetical ordering** when adding entries to `__init__.py` and `__all__`
- **Match the existing code style** exactly — read existing files first and copy the pattern
  (imports, logging, naming conventions, registration format, test patterns)
- **Speedup formula**: `speedup = torch_latency / gems_latency` (values > 1.0 mean gems is faster)
- **Always use built-in tools** (shell, Read, Write, Edit, Grep, Glob) instead of
  outputting commands or code snippets for the user to execute manually
