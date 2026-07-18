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


# KernelGen Skill — Generate GPU Operators via MCP (General Purpose)

You are an expert at generating GPU kernel operators using the `kernelgen-mcp` MCP service and
integrating them into **any** Python/Triton repository. Unlike the specialized FlagGems or vLLM
skills, this skill dynamically discovers the target repository's structure and conventions before
placing code.

**Tool usage**: This skill relies on the following capabilities:
- **Bash**: execute shell commands (python, pip, pytest, etc.)
- **Read**: read files from the codebase
- **Write / Edit**: create or modify files
- **Grep**: search file contents
- **Glob**: find files by pattern
- **MCP tools**: `mcp__kernelgen-mcp__generate_kernel` and `mcp__kernelgen-mcp__optimize_kernel`

> **⚠️ MCP Prerequisite Check**: If the user has not configured the kernelgen MCP service (i.e., the MCP tools are unavailable or calls fail),
> immediately prompt the user to visit https://kernelgen.flagos.io/ to register and obtain the kernelgen MCP service URL and JWT Token,
> then follow the instructions in Step 0b to complete the configuration and retry. Do not proceed with subsequent steps if MCP is not ready.

## Execution Rules

Follow these rules strictly to avoid getting lost in the multi-step workflow:

1. **Follow the steps in order**: Step 0 → Step 1 → ... → Step 9. Never jump ahead.
2. **Never skip Step 2** (repo structure discovery) — always analyze before generating.
3. **Never skip Step 3** (operator existence check) — always verify before generating.
4. **Do not generate or write any code before Step 5** (MCP call). Steps 0–4 are preparation only.
5. **Always confirm with the user before destructive actions**: replacing files, installing packages,
   overwriting operators.
6. **Report progress to the user when entering each step**. Print a short status line like
   `Entering Step 2 — Analyzing repository structure...` so the user knows the workflow is progressing.
7. **Never fabricate repository files or paths**. If a required file cannot be found using the
   Glob or Grep tools, report it to the user instead of guessing.
8. **Minimize file reads during discovery and search.** Always use Grep or Glob to locate
   targets first. Only Read a file when you have a specific reason (e.g., learning conventions
   from a reference operator, checking registration format). Never speculatively read files
   "just in case".
9. **Never scan the entire repository.** In Step 3 and beyond, only search inside directories
   discovered in Step 2. Never run Grep on `.` or the repo root — always scope searches to
   the specific discovered source, test, or benchmark directories. Unscoped searches cause
   token explosion and match vendored code, docs, and build artifacts.
10. **CRITICAL — MCP is mandatory**: ALL operator code generation MUST go through the
    `mcp__kernelgen-mcp__generate_kernel` MCP tool. NEVER generate Triton kernels, PyTorch
    wrappers, or operator implementations yourself — even if the operator seems simple (e.g.,
    relu, abs). If MCP is not configured, not reachable, or fails after all retries in Step 5,
    STOP and report the issue to the user. Do NOT fall back to writing kernel code manually.
    The MCP service produces optimized, tested code that manual writing cannot match.

---

## Step 0: Pre-flight — Environment & MCP Check

### 0a. Check Python Environment

Use the Bash tool to run a diagnostic that checks each package independently:

```bash
python - <<'PY'
import importlib, sys

def check(pkg, alias=None):
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "installed")
        print(f"{alias or pkg}: {ver}")
        return True
    except Exception:
        print(f"{alias or pkg}: NOT installed")
        return False

check("torch")
check("triton")

try:
    import torch
    print(f"cuda_version: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")
except Exception:
    pass

check("pytest")
check("numpy")
PY
```

**If any import fails**, stop and ask the user before installing anything. Present the missing
packages and let the user confirm:

| Missing package | Suggested install command | Notes |
|---|---|---|
| `torch` | Ask the user | User MUST pick the correct CUDA variant. Never auto-install torch. |
| `triton` | `pip install triton` | |
| `pytest` | `pip install pytest` | |
| `numpy` | `pip install numpy` | |

**Important**: Never install `torch` without asking the user which CUDA version they need.

If `torch` is installed but `torch.cuda.is_available() == False`, warn the user that GPU tests
will fail and ask how to proceed.

After the user confirms and packages are installed, re-run the diagnostic. Only proceed when
torch and triton imports succeed.

### 0b. Check MCP Availability

Use the Read tool to check for MCP configuration. **Only check project-local paths** — do NOT
attempt to read user home directory (`~/`) as Claude Code typically cannot access it:

1. `.claude/settings.json` — look for `mcpServers` containing the key `"kernelgen-mcp"`
2. `.mcp.json` — same check

If found, check whether the exact key `"kernelgen-mcp"` exists under `mcpServers`.
Do NOT use substring matching (e.g., matching `"kernelgen"` inside another key name).

**If the MCP server is NOT found in either file**, ask the user:

```
kernelgen-mcp service was not detected in the project configuration.

Possible scenarios:
  A) Not yet configured — Please visit https://kernelgen.flagos.io/ to register and obtain a JWT Token,
     then provide me with the URL and Token, and I will write them to .claude/settings.json.
  B) Already configured globally — If you have already configured kernelgen-mcp in your global settings,
     please let me know and I will skip this step and continue.

The final configuration format is as follows (.claude/settings.json):
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
```

Wait for the user to respond. Then:
- If the user provides URL + JWT: use the Edit tool (or Write tool if the file doesn't exist)
  to write the configuration, **merging** with any existing content (never overwrite other MCP
  server entries or top-level keys). Tell the user to restart Claude Code.
- If the user says MCP is configured globally: proceed to Step 1.

**If the MCP server IS configured**, proceed to Step 1.

---

## Step 1: Understand the Operator Request

Parse the user's description to determine:
- `kernel_name`: operator name in **snake_case** (e.g., `rms_norm`, `relu`, `softmax`)
- `func_desc`: what the operator does
- `func_type`: one of `unary_pointwise`, `binary_pointwise`, `reduction`, `normalization`,
  `attention`, `activation`, `quantization`, `moe`, `blas`, `sampling`, `other`.
  **Auto-infer from the operator signature when possible:**
  - If 1 tensor input, no dim arg → `unary_pointwise`
  - If operator has >=2 tensor inputs → `binary_pointwise`
  - If has a `dim` argument → `reduction`
  - If name contains `norm`, `softmax`, `rmsnorm`, `layernorm`, `batchnorm` → `normalization`
  - If name contains `relu`, `gelu`, `silu`, `swish`, `tanh`, `sigmoid` → `activation`
  - If name is `mm`, `matmul`, `bmm`, `addmm` → `blas`
  - If name contains `attention`, `flash`, `paged` → `attention`
  - If name contains `quant`, `awq`, `gptq` → `quantization`
  - If name contains `moe`, `expert` → `moe`
  - Otherwise → `other`
  Confirm the inferred `func_type` with the user if ambiguous.
- `arg_names`, `arg_type`, `arg_descs`, `output_arg_desc`: parameter information

**Normalize `kernel_name` to snake_case** before using it anywhere:
- `RMSNorm` → `rms_norm`
- `SiluAndMul` → `silu_and_mul`
- `LayerNorm` → `layer_norm`

Check this common alias table before asking the user:

| User says | Correct `kernel_name` |
|---|---|
| swish | `silu` |
| negative / negate | `neg` |
| power | `pow` |
| absolute | `abs` |
| multiply | `mul` |
| divide | `div` |
| clip | `clamp` |
| hard_swish | `hardswish` |
| hard_sigmoid | `hardsigmoid` |
| logarithm / ln | `log` |
| exponential | `exp` |
| hyperbolic_tangent | `tanh` |
| square | `pow (exp=2)` |
| square_root | `sqrt` |
| cube | `pow (exp=3)` |
| cube_root | `cbrt` |
| inverse_sqrt | `rsqrt` |
| reciprocal / inverse | `reciprocal` |
| minimum | `min` |
| maximum | `max` |
| floor_divide | `floor_divide` |
| modulo / mod | `remainder` |

### Complexity Gate

Before proceeding, check the `func_type`. If it is one of the following **complex** types,
warn the user and ask for explicit confirmation before continuing:

- `attention`
- `moe`
- `quantization`
- `sampling`

```
Note: Operators of type <func_type> are relatively complex. KernelGen's auto-generation success rate is approximately 60%,
and manual tuning may be required. Do you want to proceed with generation?
```

For simpler types (`unary_pointwise`, `binary_pointwise`, `reduction`, `normalization`,
`activation`, `blas`), proceed without warning — these have >90% success rate.

### Argument Inference

**If argument information is missing or the description is vague**:

1. Use the Grep tool to search for the operator name **only inside directories discovered in
   Step 2** (source dirs, test dirs, kernel dirs) to infer argument signatures from existing
   code. **Never grep the repo root or unscoped paths** — this causes token explosion in
   large repos.
2. Present the inferred signature to the user for confirmation before proceeding.
3. **Fallback**: If no matches are found, use a default signature based on `func_type`:

   | func_type | Default args |
   |---|---|
   | `unary_pointwise`, `activation` | `arg_names: ["x"]`, `arg_type: ["torch.Tensor"]` |
   | `binary_pointwise` | `arg_names: ["x", "y"]`, `arg_type: ["torch.Tensor", "torch.Tensor"]` |
   | `normalization` | `arg_names: ["x", "weight"]`, `arg_type: ["torch.Tensor", "torch.Tensor"]` |
   | `reduction` | `arg_names: ["x"]`, `arg_type: ["torch.Tensor"]` |
   | `attention` | `arg_names: ["q", "k", "v"]`, `arg_type: ["torch.Tensor", "torch.Tensor", "torch.Tensor"]` |
   | `blas` | `arg_names: ["a", "b"]`, `arg_type: ["torch.Tensor", "torch.Tensor"]` |
   | other | `arg_names: ["x"]`, `arg_type: ["torch.Tensor"]` |

   Present the default to the user and ask them to confirm or refine. **Do not send defaults
   directly to MCP without user confirmation** — a wrong signature causes schema errors.

---

## Step 2: Discover Repository Structure

**This step is critical.** Unlike the FlagGems/vLLM-specific skills, we do NOT assume any fixed
directory layout. Instead, we dynamically discover the project structure.

### 2a. Identify the Project Type

Use the Glob and Read tools to check for project identity:

```
Glob: pyproject.toml, setup.py, setup.cfg
```

Read `pyproject.toml` (or `setup.py` / `setup.cfg`) to determine:
- **Project name** (e.g., `flag_gems`, `vllm`, `lmdeploy`, etc.)
- **Source root** (e.g., `src/`, `.`, `lib/`)
- **Package name** — the importable top-level module

**Known-project shortcut**: If the project name matches a known specialized skill, inform
the user and suggest switching:

| Project name | Specialized skill | Suggestion |
|---|---|---|
| `flag_gems` / `FlagGems` | `/kernelgen_for_flaggems` | "Detected FlagGems repository. It is recommended to use the specialized skill `/kernelgen_for_flaggems` for a higher success rate. Would you like to switch?" |
| `vllm` | `/kernelgen_for_vllm` | "Detected vLLM repository. It is recommended to use the specialized skill `/kernelgen_for_vllm` for a higher success rate. Would you like to switch?" |

If the user agrees, stop and tell them to invoke the specialized skill. If the user declines
or the project is not in the table, continue with the general skill.

### 2b. Map Key Directories

Use the Glob tool with **shallow patterns** to avoid token explosion in large repos. Do NOT
use deep recursive `**` patterns that could match thousands of files.

**Important**: If any single Glob returns more than 20 matches, **stop inspecting that
pattern** — just record the directory prefix (e.g., `src/pkg/ops/`) and move on. Do NOT
read or list all matched files; only note the directory structure.

```
Glob: src/*/ops/*.py          (source ops — one level under src/)
Glob: src/*/kernels/*.py      (source kernels)
Glob: */ops/*.py              (top-level package ops)
Glob: */kernels/*.py          (top-level package kernels)
Glob: tests/*.py              (top-level tests)
Glob: tests/kernels/*.py      (kernel-specific tests)
Glob: benchmark*/*.py         (benchmarks)
Glob: benchmark*/kernels/*.py (kernel benchmarks)
```

If these shallow patterns return no results for a category, try ONE level deeper:
```
Glob: src/*/*/ops/*.py
Glob: src/*/*/kernels/*.py
```

If **still** no operator directory is found, run a generic fallback scan:
```
Glob: src/*/*kernel*.py
Glob: src/*/*op*.py
```
This catches non-standard directory names like `triton_kernels/`, `gpu_ops/`, `cuda_ops/`.

**Stop expanding once you find matches.** Do not keep globbing deeper.

**Early-stop rule**: Once you have discovered both an **operator source directory** and a
**test directory**, stop all further glob exploration. Do not continue searching for additional
directories — two anchors are sufficient to infer the rest of the layout.

**Disambiguation rule**: If multiple operator directories are discovered (e.g., `src/foo/ops/`
and `src/bar/ops/`), **prefer the one that belongs to the main package name** detected in
Step 2a (`pyproject.toml`). If still ambiguous, ask the user which directory to use.

Record the discovered layout in this table (fill in dynamically):

| Purpose | Discovered Path |
|---------|----------------|
| **Operator source code** | _(e.g., `src/pkg/ops/`, `pkg/kernels/`, etc.)_ |
| **Tests (accuracy)** | _(e.g., `tests/`, `tests/kernels/`, etc.)_ |
| **Benchmarks** | _(e.g., `benchmark/`, `benchmarks/kernels/`, etc.)_ |
| **Init/registry files** | _(e.g., `src/pkg/ops/__init__.py`, etc.)_ |
| **Custom op registration** | _(e.g., `pkg/_custom_ops.py`, etc.)_ |

### 2c. Study Existing Operators as Reference

Find an existing operator implementation similar to the requested one and read it to learn:

1. **Import style** — what modules are imported, decorator patterns
2. **Kernel style** — raw Triton pointer-based, or higher-level wrappers like `pointwise_dynamic`
3. **Function naming** — snake_case, prefix/suffix conventions (e.g., `_kernel`, `_forward`)
4. **Wrapper pattern** — how the Python wrapper calls the Triton kernel
5. **License headers** — whether files start with SPDX or other license headers
6. **Logging** — which logging approach is used (e.g., `vllm.logger.init_logger`, stdlib `logging`)
7. **Autotune usage** — whether the repo uses `@triton.autotune` and which BLOCK_SIZE configs.
   If autotune is used, also note `num_warps` and `num_stages` values from existing configs —
   these will be replicated for the new kernel's autotune configs in Step 6.

Also read the matching test and benchmark files to learn:
- **Test pattern**: parametrize style, shapes, dtypes, assertion utilities
- **Benchmark pattern**: timing methodology, reporting format

**Store all discovered conventions** — they will be applied in Step 6.

---

## Step 3: Check Whether the Operator Already Exists

Before calling the MCP generator, **thoroughly search** the codebase for existing implementations
using the Glob and Grep tools.

### 3a. Generate Search Variants

From `kernel_name`, generate **all naming variants** to search:
- snake_case: `layer_norm`
- concatenated lowercase: `layernorm`
- CamelCase: `LayerNorm`
- Partial components (for compound names): only if `kernel_name` contains `_and_` or `_with_`,
  split and search components **longer than 4 characters**. Example:
  - `silu_and_mul` → search `silu` (5 chars, OK), skip `mul` (3 chars, too short)
  - `add_relu` → do NOT split (no `_and_` / `_with_`), search as whole name only
  Short fragments like `mul`, `add`, `div`, `neg` match too many unrelated symbols.

### 3b. Search

For **each** naming variant, search:
1. **Source files**: Glob for `*<variant>*.py` in discovered source directories (shallow)
2. **Function and class definitions**: Grep for these patterns in discovered source directories:
   - `def <variant>` — direct function definition
   - `<variant>_kernel` — kernel function naming convention
   - `<variant>_forward` — forward function naming convention
   - `<variant>_backward` — backward function naming convention
   - `class <CamelCaseVariant>` — class-based operator (e.g., `class LayerNorm`,
     `class SoftmaxKernel`, `class RMSNormOp`)
   - `register_op("<variant>")` / `OP_REGISTRY["<variant>"]` / `aten_map["<variant>"]` —
     dispatch-style registration where the implementation function name may differ from
     the operator name
   This catches operators that live inside files with different names (e.g., `softmax_kernel`
   inside `normalization.py`, or `silu_and_mul` inside `activation.py`).
3. **Init/registry**: Grep for `<variant>` in all `__init__.py` files and any custom op
   registration files found in Step 2
4. **Tests**: Grep for `<variant>` in discovered test directories
5. **Benchmarks**: Grep for `<variant>` in discovered benchmark directories

### If the operator already exists

Present findings to the user and ask them to choose. **Stop and wait** for the user's explicit
choice before continuing:

> Detected that the `<kernel_name>` operator already exists in the repository:
> - Implementation: `<file path>`
> - Tests: `<file path>`
> - Benchmarks: `<file path>`
>
> Please choose how to proceed:
>
> **A — Cancel generation**: Keep the existing operator unchanged.
>
> **B — Replace existing**: Replace the current implementation with a new version generated by KernelGen.
>
> **C — Add custom variant side-by-side**: Keep the existing operator and place the new version as `<kernel_name>_v2` alongside it, so both coexist.

Only proceed to Step 4 after the user has made a choice.

### If the operator does NOT exist

Proceed directly to Step 4.

---

## Step 4: Research Context (flagos_wiki)

Before calling the MCP generator, use the Read tool to gather reference materials that improve
generation quality:

1. **Read similar operator code** from the codebase — pick the most similar existing operator
   to the one being generated (same `func_type`).

2. **Read the test pattern** from the matching test file to understand shapes, dtypes, assertions.

3. **Read the benchmark pattern** from the matching benchmark file to understand timing methodology.

4. **If replacing an existing op** (Option B), read the current implementation so the new version
   can be compared / improved upon.

   **If reference files cannot be found**, do not stop. Proceed with general algorithm hints
   based on your knowledge of the operator type. An empty or minimal `flagos_wiki` is acceptable.

5. Collect all findings into the `flagos_wiki` parameter as a `List[str]`. Rules:
   - Each string should be a concise reference snippet, **under 200 characters**.
   - **Maximum 10 items** — avoid token explosion.
   - **Prefer algorithm insights over file path descriptions**.
   - **Always include a kernel pattern hint** as the first item (e.g.,
     `"Kernel pattern: elementwise"`, `"Kernel pattern: row-wise reduction over hidden_size"`).
   - **Always include `"Use Triton for GPU parallelization"` as the second item** — this
     anchors the generator toward Triton output and reduces fallback failures.
   - **Always include a memory layout hint** (e.g.,
     `"Input tensors are contiguous row-major layout"`).
   - **Include a memory access pattern hint if inferable** from the `func_type` (e.g.,
     `"Memory access: coalesced contiguous loads"` for pointwise/elementwise,
     `"Memory access: row-wise strided reads over hidden_size"` for normalization/reduction).
     This significantly improves generated kernel quality.
   - Include relevant conventions discovered in Step 2 (e.g.,
     `"Repo uses pointwise_dynamic wrapper for elementwise ops"`,
     `"Repo uses @triton.autotune with BLOCK_SIZE configs [64,128,256,512]"`).

---

## Step 5: Call kernelgen-mcp

Invoke `mcp__kernelgen-mcp__generate_kernel` with the parameters gathered above:

```
kernel_name:     "<kernel_name>"
func_desc:       "<description of what the operator does>"
func_type:       "<func_type>"
arg_names:       [<list of argument names>]
arg_type:        [<list of argument types>]
arg_descs:       [<list of argument descriptions>]
output_arg_desc: "<description of the output>"
flagos_wiki:        [<list of reference strings from Step 4>]
```

The MCP returns code blocks which may include:
- `torch_code` — PyTorch reference implementation
- `triton_code` — Triton kernel implementation
- `test_func_code` — accuracy test code
- `benchmark_func_code` — performance benchmark code

**If `torch_code` is missing**, construct a `ref_impl` yourself using standard PyTorch ops
based on `func_desc`.

### MCP Failure Retry Strategy

If the MCP call fails, apply retries in this order:

1. **First retry**: Remove `flagos_wiki` parameter and call again (handles `unexpected argument`
   / `unknown field` / `invalid schema` errors).
2. **Second retry**: Use minimal parameters only (`kernel_name`, `func_desc`, `func_type`,
   `arg_names`, `arg_type`) — drop `arg_descs`, `output_arg_desc`, and `flagos_wiki`.
3. **Third retry**: Change `func_type` to `"other"` and keep only minimal parameters. This
   handles `func_type` mismatch errors where the MCP backend does not recognize the original
   `func_type` value (e.g., `activation` vs `unary_pointwise`).
4. **If still failing**: Stop and report the exact error to the user. Do NOT keep retrying.

---

## Step 5.5: MCP Report Review

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
- If **A (skip local testing)**: After placing code in Step 6, skip Steps 7 and 8 (local
  accuracy tests and benchmarks) and proceed directly to Step 9 (Summary). Use the MCP
  report numbers for the summary.
- If **B (run local tests)**: Proceed normally through Steps 6 → 7 → 8 → 9. Before
  running local tests, perform the **Chip Compatibility Check** described in Step 6.5d.

**If the MCP response does NOT include test reports** (only code blocks), proceed normally
to Step 6 — local testing will be required.

---

## Step 6: Adapt and Place Code into the Repository

This is the most critical step. The generated code must be **transformed to match the local
repository's conventions** discovered in Step 2.

### 6a. Transform the Kernel Code

Apply the conventions discovered in Step 2c. **The decision to rewrite depends entirely on
what the repo actually uses** — never assume:

**Only rewrite to wrapper style if BOTH conditions are met**:
1. The repo explicitly uses wrapper utilities (e.g., FlagGems `pointwise_dynamic`).
2. The reference operator's kernel function signature receives **scalar elements** (not pointers).

If either condition is unclear, **keep the MCP-generated raw Triton pointer-based code**.
Misidentifying wrapper style produces completely broken kernels.

When rewriting to wrapper style:
- The `@triton.jit` kernel should receive **scalar elements** (not pointers).
- **Remove ALL** `tl.load()`, `tl.store()`, pointer arithmetic, mask logic, and BLOCK_SIZE
  parameters — the wrapper handles these automatically.
- Only keep the pure math expression.

**Otherwise, keep the MCP-generated raw Triton pointer-based code.** This is the safer default
for most repositories. Apply these adaptations:
- **Match the repo's autotune convention exactly**:
  - If existing kernels use `@triton.autotune` → add it, using the same BLOCK_SIZE configs
    found in existing kernels.
  - If existing kernels do **NOT** use `@triton.autotune` (e.g., torchinductor, tinygrad, some
    inference engines explicitly avoid it) → do **NOT** introduce it. Use a fixed BLOCK_SIZE
    matching the repo's pattern instead.
  - If no existing kernels are found for reference, use a default autotune set:
  ```python
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
          triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
          triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
          triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
      ],
      key=["n_elements"],  # replace with actual problem-size arg name
  )
  ```
- Ensure proper `tl.constexpr` for meta parameters.
- Use `triton.cdiv` for grid computation.

**Common adaptations for all repos:**
- Add the repo's standard license header (if any, discovered in Step 2c).
- Use the repo's logging approach (e.g., `vllm.logger.init_logger`, `logging.getLogger`).
- Follow the repo's naming conventions for kernel functions (e.g., `_kernel` suffix, `_forward`
  suffix).
- Match import style exactly.

### 6b. Determine File Placement

Place files according to the layout discovered in Step 2b:

| File | Location |
|------|----------|
| Kernel implementation | Discovered operator source directory + `<kernel_name>.py` |
| Accuracy test | Discovered test directory + `test_<kernel_name>.py` (or append to existing test file if the repo groups tests by category) |
| Performance benchmark | Discovered benchmark directory + `benchmark_<kernel_name>.py` (or append to existing benchmark file) |

**Decision: separate file vs. append to existing?**
- Use the Glob tool to check if the repo uses **one test file per operator** (e.g.,
  `tests/test_relu.py`, `tests/test_softmax.py`) or **grouped test files** (e.g.,
  `tests/test_unary_pointwise_ops.py`, `tests/test_reduction_ops.py`).
- If grouped: use the Edit tool to append the new test to the appropriate existing file.
- If per-operator: use the Write tool to create a new test file.
- Apply the same logic for benchmarks.

### 6c. Write the Kernel Implementation

Use the Write tool (new file) or Edit tool (replacing existing) to create the operator file.

The implementation should follow the exact pattern of similar operators in the repo. Include:
- License header (if the repo uses one)
- Imports matching repo style
- Triton kernel function(s)
- Python wrapper function(s)
- Logging statements matching repo convention
- In-place variants if applicable (verify pattern by reading an existing in-place op first)

### 6d. Register the Operator (if applicable)

If the repo has a registration mechanism, use the Edit tool to register the new operator.

**First, discover the registration mechanism.** Not all repos use `__init__.py`. Search for:
- `__init__.py` exports (most common)
- Grep for `register_op`, `OP_REGISTRY`, `_dispatch_table`, `aten_map`, `_FULL_CONFIG` in
  the discovered source directories — these are common registry patterns
- `_custom_ops.py` or similar custom op registration files found in Step 2

If a registration mechanism is found:
1. **Read the registration file** first to understand the exact format.
2. **Add the new entry in alphabetical order** — do not change existing entries.
3. **Match the exact registration format** used by existing entries.

If **no registration mechanism is found**, check one more thing before skipping:

**Implicit import check**: If the operator source directory contains an `__init__.py` that
imports other operators (e.g., `from .relu import *`, `from .softmax import softmax`), you
MUST add a corresponding import line for the new operator. Otherwise the kernel file exists
but the package cannot import it.

```python
# Example: src/pkg/ops/__init__.py already has:
from .relu import relu
from .softmax import softmax
# → Add:
from .<kernel_name> import <kernel_name>
```

If `__init__.py` does not exist or is empty, the kernel is standalone — skip registration.

For **Option C (side-by-side variant)**: Do NOT register in the main dispatch mechanism. Only
add to `__init__.py` for importability, not for aten/dispatch override.

### 6e. Write the Accuracy Test

Follow the test pattern discovered in Step 2c. Key conventions to match:
- Parametrize decorators (shapes, dtypes, seeds, devices)
- Reference implementation approach (pure PyTorch `ref_impl` vs `torch.<op>`)
- Assertion method (`torch.testing.assert_close`, `gems_assert_close`, custom utils)
- Per-dtype tolerances: fp32 → rtol/atol=1e-5, fp16 → 1e-2, bf16 → 2e-2
- Test markers / decorators (`@pytest.mark.<kernel_name>`, `@torch.inference_mode()`, etc.)
- Device handling (`flag_gems.device`, `"cuda"`, parametrized CUDA_DEVICES)
- If the repo uses `to_reference()` or similar utilities, use them.

If the repo registers pytest markers, add the new marker to the marker registration file
(check `pytest.ini`, `pyproject.toml [tool.pytest.ini_options]`, or `setup.cfg`).

### 6f. Write the Performance Benchmark

Follow the benchmark pattern discovered in Step 2c. Key conventions to match:
- Timing methodology (`triton.testing.do_bench_cudagraph`, `GenericBenchmark`, custom)
- Reporting format (`triton.testing.perf_report`, print-based, etc.)
- Configuration ranges (batch sizes, sequence lengths, hidden sizes)
- Baseline comparison (always use `ref_impl` for the baseline, not `torch.<op>` which may
  not exist for fused kernels)

---

## Step 6.5: Pre-test Validation

Before running the full test suite, perform quick checks:

### 6.5a. Lint / Format Check

If the repo uses linting tools, run them on the changed files:

```bash
# Check which linter the repo uses
# Look in pyproject.toml for [tool.ruff], [tool.black], [tool.isort], etc.
```

If found, run the linter on changed files to catch style issues early. If no linter is
configured, skip this step.

### 6.5b. Import Smoke Test

Verify the new module can be imported:

```bash
python -c "from <module_path> import <kernel_name>; print('Import OK')"
```

### 6.5c. Triton Compile Smoke Test

Run a minimal invocation to catch Triton compile errors early. **Build the smoke test call
based on `arg_names` from Step 1** — do NOT hardcode a single-argument call.

**Important**: Guard the device selection so the smoke test does not crash if CUDA is
unavailable (even though Step 0 checked — the user may have chosen to proceed without GPU):

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**If `device == "cpu"`**: Skip the Triton compile smoke test entirely. Triton kernels require
CUDA and will crash with `RuntimeError` on CPU. Report to the user:
```
CUDA unavailable — skipping Triton compile test. The kernel will be compiled for the first time during the Step 7 accuracy tests.
```
Then proceed directly to Step 7.

**If `device == "cuda"`**: construct the call based on `func_type` and `arg_names`.

Then construct the call based on `func_type` and `arg_names`. **Use shapes inferred from the
reference operator read in Step 2c.** If no reference shapes are available, use these fallbacks:

```python
# Fallback shapes by func_type:
#   unary_pointwise / activation: (128,)
#   binary_pointwise:             (128,)
#   reduction:                    (32, 128)
#   normalization:                (32, 128) + weight of shape (128,)
#   attention:                    (1, 4, 8) for q/k/v
#   blas:                         (32, 64) and (64, 32)

# For unary ops (arg_names: ["x"]):
x = torch.randn(128, device=device, dtype=torch.float16)
out = kernel_fn(x)

# For binary ops (arg_names: ["x", "y"]):
x = torch.randn(128, device=device, dtype=torch.float16)
y = torch.randn(128, device=device, dtype=torch.float16)
out = kernel_fn(x, y)

# For normalization ops (arg_names: ["x", "weight"]):
x = torch.randn(32, 128, device=device, dtype=torch.float16)
weight = torch.ones(128, device=device, dtype=torch.float16)
out = kernel_fn(x, weight)

# For attention ops (arg_names: ["q", "k", "v"]):
q = torch.randn(1, 4, 8, device=device, dtype=torch.float16)
k = torch.randn(1, 4, 8, device=device, dtype=torch.float16)
v = torch.randn(1, 4, 8, device=device, dtype=torch.float16)
out = kernel_fn(q, k, v)
```

**General rule**: create one small tensor for each entry in `arg_names`, matching the
expected shapes for the `func_type`. Use `inspect.signature` to infer any missing arguments
(e.g., `weight`, `bias`, `eps` for `layer_norm`) before constructing the smoke test, so that
all required parameters are covered. If the smoke test fails, read the error and fix the
kernel before proceeding.

### 6.5d. Chip Compatibility Check (before local testing)

Before running any local tests, verify that the local hardware matches the target device
specified for the operator. Use the Bash tool to detect local hardware:

```bash
python - <<'PY'
import subprocess, sys

def detect():
    chips = []
    # NVIDIA GPU
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                chips.append(("nvidia", torch.cuda.get_device_name(i)))
    except Exception:
        pass
    # Huawei Ascend NPU
    try:
        r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and "NPU" in r.stdout:
            chips.append(("huawei", "Ascend NPU"))
    except Exception:
        pass
    # AMD / Hygon DCU
    try:
        r = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            chips.append(("haiguang" if "hygon" in r.stdout.lower() else "amd", "ROCm device"))
    except Exception:
        pass
    if not chips:
        chips.append(("unknown", "No accelerator detected"))
    for kind, name in chips:
        print(f"  {kind}: {name}")
    return chips

print("Local hardware:")
detect()
PY
```

Compare the detected hardware with the operator's `target_device` (determined in Step 1 or
inferred from user context — default is `nvidia`):

| Target Device | Required Local Hardware |
|---|---|
| `nvidia` | NVIDIA GPU (detected via `torch.cuda`) |
| `huawei` | Huawei Ascend NPU (detected via `npu-smi`) |
| `haiguang` | Hygon DCU (detected via `rocm-smi` with Hygon identifier) |
| `moore` | Moore Threads GPU |
| `tianshu` | Tianshu GPU |

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

If the user chooses A, skip Steps 7 and 8 and proceed to Step 9 (Summary).
If the user chooses B, proceed with local tests but note the mismatch in the final report.

**If the hardware matches**, proceed normally.

---

## Step 7: Run Accuracy Tests

Use the Bash tool to run accuracy tests.

First, determine the correct invocation based on repo structure. **Prefer `pytest` over
`python -m pytest`** because many repos depend on `conftest.py` discovery which works more
reliably with direct `pytest` invocation:

1. **Primary**: `pytest <test_path> -q` (use `-q` to minimize output and avoid token explosion;
   only switch to `-v` if a test fails and you need detailed traceback)
2. **Fallback** (if `pytest` command not found): `python -m pytest <test_path> -q`
3. **If not pip-installed**: prepend `PYTHONPATH=.` to either command

For grouped test files, use `-m <kernel_name>` or `-k <kernel_name>` to select only the
new operator's tests.

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
- `AssertionError` from `torch.testing.assert_close` — numerical mismatch
- Wrong output values (results don't match reference)
- Shape mismatch in output tensors
- NaN or Inf in output
- Any error that indicates the kernel logic is incorrect

### Retry Protocol (strictly follow this order)

**Step 7a. (Category A only) Self-fix — maximum 1 attempt:**
If the error is Category A (compilation/import), you may attempt exactly ONE fix:
1. Read the error traceback carefully.
2. Apply a targeted fix using the Edit tool (e.g., fix import path, fix syntax, fix argument name).
3. Re-run the tests.
4. If the test **passes** → proceed to Step 8.
5. If the test **still fails** → proceed to Step 7b. Do NOT attempt a second self-fix.

**If the error is Category B (algorithm/accuracy), skip Step 7a entirely** — go directly
to Step 7b. Do NOT attempt to fix algorithm logic yourself, as this typically leads to
an endless fix-retry loop without converging. The MCP service has better optimization
capabilities for these issues.

**Step 7b. MCP re-generation — pass error context to generate_kernel:**
Re-call `mcp__kernelgen-mcp__generate_kernel` with the **same parameters as Step 5**,
but add the error information to `flagos_wiki` as additional hints:
```python
flagos_wiki = [
    # ... original flagos_wiki items from Step 4 ...
    "Previous generation failed accuracy test: <brief error description>",
    "Error was: <key line from traceback>",
    "Fix hint: <your analysis>"
]
```
Replace the kernel code with the new MCP output, re-run tests.
- If tests **pass** → proceed to Step 8.
- If tests **still fail** → proceed to Step 7c.

**Step 7c. MCP optimization — pass error context to optimize_kernel:**
Try `mcp__kernelgen-mcp__optimize_kernel` with the current kernel code and the
`check_result` parameter containing the error traceback. This endpoint can fix
memory access patterns, index calculations, and numerical issues.
Replace the kernel code with the optimized output, re-run tests.
- If tests **pass** → proceed to Step 8.
- If tests **still fail** → proceed to Step 7d.

**Step 7d. Stop and report:**
Do not keep retrying. Report the failure to the user with:
- The specific test failures and error messages
- Your analysis of what might be wrong
- Suggestion to try with different `func_type` or additional `flagos_wiki` hints

---

## Step 8: Run Performance Benchmark

Use the Bash tool to run the benchmark.

Determine invocation based on repo structure:
- Standalone benchmark file: `python <benchmark_path>`
- pytest-based benchmark: `python -m pytest <benchmark_path> -q -k "perf"`

**Timeout**: Run the benchmark with a **120-second timeout**. If it exceeds this without
completing, interrupt the process and report partial results to the user:
```
Benchmark exceeded 120 seconds without completing, interrupted. Partial output below:
<partial output>
For a full benchmark, please run manually: python <benchmark_path>
```
Some benchmarks (e.g., large sweep over sequence lengths / hidden sizes) can run for tens of
minutes — the user should run these manually outside the skill.

### Speedup Extraction

Try to extract speedup metrics from the output:
1. Look for explicit speedup with `x` suffix: `(\d+\.?\d*)x`
2. Look for explicit `speedup` labels: `speedup.*?(\d+\.?\d*)`
3. Look for paired timing values (triton vs torch): extract both and compute
   `speedup = torch_latency / kernel_latency`
4. Look for timing with units: `(\d+\.?\d*)\s*(us|ms|s)`

**Fallback**: If latency values cannot be reliably parsed (e.g., output format is non-standard,
or only raw numbers are printed), **report the raw benchmark output** to the user instead of
attempting to compute speedup. Let the user interpret the results:

```
Benchmark completed, but speedup could not be automatically parsed. Raw output below:
<raw output>
```

---

## Step 9: Summary

Provide a clear summary to the user with **exact numbers** extracted from test and benchmark
output:

```
=== KernelGen Operator Generation Report ===

Operator Name: <kernel_name>
Target Repository: <project_name>
Generation Mode: New / Replace Existing / Custom Variant (v2)
Operator Type: <func_type>

File Changes:
  - [New/Modified] <kernel_file_path>           (Triton kernel)
  - [Modified] <init_file_path>                   (Registration, if applicable)
  - [New/Modified] <test_file_path>             (Accuracy Tests)
  - [New/Modified] <benchmark_file_path>        (Performance Benchmark)

Accuracy Tests: <N> passed, <M> failed (total <N+M> test cases)
  Pass Rate: <N/(N+M)*100>%
  Failed Cases: <list failed test names if any, or "None">

Performance Benchmark:
  Avg Speedup: <X.XX>x vs PyTorch reference
  Best Speedup: <X.XX>x (config details)
  Worst Speedup: <X.XX>x (config details)
  (If benchmark was not run, failed, or could not be parsed, write "Incomplete" and explain the reason)

MCP Iterations: <1, 2, or 3> (write 1 if first generation passed)

Performance Analysis:
  <Assess based on average speedup:
   - Speedup > 5.0x → "Extreme fusion success, typical for fused multi-op kernels"
   - Speedup > 3.0x → "Excellent compute-bound optimization, significant kernel fusion benefit"
   - Speedup > 2.0x → "Compute-bound optimization effective, significant speedup achieved"
   - Speedup 1.2x ~ 2.0x → "Moderate speedup, kernel likely balanced between compute and memory"
   - Speedup 0.5x ~ 1.2x → "Kernel likely memory-bound, limited optimization headroom"
   - Speedup < 0.5x → "Kernel slower than PyTorch reference — likely suboptimal memory access pattern"
   - Unable to parse → Skip this section>

Issues and Fixes: (if any)
```

**How to extract the numbers:**
- **Pass/fail counts**: Parse pytest output for the line matching `X passed` or `X passed, Y failed`.
  Use regex pattern `(\d+) passed` and `(\d+) failed` to extract.
- **Speedup**: Parse benchmark output for kernel vs PyTorch time ratios. Calculate
  `speedup = torch_time / kernel_time` for each config. Report min, max, and average.
  - Use these regex patterns to extract timing values:
    - `(\d+(?:\.\d+)?)\s*(us|ms|s)` — number with unit
    - `(?i)mean:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — mean timing (pytest-benchmark format, e.g., `mean: 0.123 ms`)
    - `(?i)avg:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — average timing value
    - `(?i)median:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — median timing value
    - `(?i)(triton|kernel).*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — kernel time
    - `(?i)(torch|pytorch|reference).*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — PyTorch reference time
  - **Do not just copy the raw table** — always compute and report the actual speedup ratios.
  - If parsing fails, report "Unable to parse" and include the raw output.

**After presenting the summary, check the speedup:**

- **If average speedup < 0.5x** (kernel slower than PyTorch), proactively warn the user:
  ```
  ⚠️ The currently generated Triton kernel performs worse than the PyTorch reference implementation.
  This is typically caused by suboptimal memory access patterns or improper block size configuration.
  Would you like to use /kernelgen_optimizer to optimize this kernel's performance?
  ```

- **If average speedup is 0.5x ~ 1.2x**, ask the user:
  ```
  Current speedup is low (<X.XX>x) and there may be room for optimization.
  Would you like to use /kernelgen_optimizer to try improving performance?
  ```

- **If average speedup > 1.2x**, no action needed — report the result normally.

---

## Step 10: Post-Completion Actions

After the summary is presented (whether based on MCP reports or local test results), perform
the following two checks.

### 10a. Chat Tool Notification

If the user's task was received via a chat tool (e.g., Feishu/飞书, Discord, Slack, Teams,
DingTalk/钉钉, WeChat Work/企业微信, Telegram, or any similar messaging platform), or if
the user mentions sending results to a client or team, ask:

> The operator generation is complete. Would you like me to prepare a message to send the
> generated files or summary to your client/team via **<chat_tool_name>**?

If the user confirms:
1. Prepare a concise summary message including: operator name, file paths, accuracy pass
   rate, speedup metrics, and any issues.
2. If the chat tool has a CLI or API integration available in the environment (e.g., `lark`
   CLI for Feishu, `discord` webhook, `slack` CLI), use it to send the message directly.
3. If no CLI/API is available, format the message as copy-paste ready text for the user.
4. If files need to be sent, list the exact file paths the user should attach.

### 10b. Git Repository PR Submission

Check whether the current project is managed by a git-based hosting service:

```bash
git remote -v 2>/dev/null | head -5
```

Detect the hosting platform from the remote URL:
- `github.com` → GitHub
- `gitlab.com` (or self-hosted GitLab) → GitLab
- `gitee.com` → Gitee
- `bitbucket.org` → Bitbucket
- Other git hosting → Generic git platform

**If a git hosting platform is detected**, ask the user:

> This project is hosted on **<platform>**. Would you like me to automatically create a
> Pull Request with the generated operator code?

**If the user confirms**, follow the standard PR creation workflow:

1. **Create a new branch**:
   ```bash
   git checkout -b kernelgen/<kernel_name>
   ```
   (Use `kernelgen/<kernel_name>-v2` for custom variants.)

2. **Stage all changed/new files** related to this operator generation:
   ```bash
   git add <kernel_file> <test_file> <benchmark_file> <registration_files...>
   ```
   Only stage files that were created or modified by this skill. Never use `git add -A`.

3. **Create a commit** with a descriptive message:
   ```bash
   git commit -m "$(cat <<'EOF'
   Add <kernel_name> operator via KernelGen

   Generated <kernel_name> (<func_type>) operator using KernelGen MCP service.
   Includes Triton kernel implementation, accuracy tests, and performance benchmarks.

   EOF
   )"
   ```

4. **Push the branch** to the remote:
   ```bash
   git push -u origin kernelgen/<kernel_name>
   ```

5. **Create the PR** using the platform's CLI tool:

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
   - **MCP iterations**: <count>
   - **Target device**: <device>
   - **Operator type**: <func_type>

   ---
   Generated by KernelGen MCP skill

   EOF
   )"
   ```

   **For Gitee** (no standard CLI — provide URL):
   ```
   Please create a PR manually at:
   https://gitee.com/<owner>/<repo>/pull/new/<branch_name>

   I have prepared the PR description above — you can copy-paste it.
   ```

   **For GitLab** (`glab` if available, otherwise provide URL):
   ```bash
   glab mr create --title "Add <kernel_name> operator via KernelGen" --description "..."
   ```

6. **Return the PR URL** to the user:
   ```
   ✅ Pull Request created: <PR_URL>
   ```

**If PR creation fails** (no CLI tool, no permissions, authentication issues, etc.):
- Report the error to the user
- Provide the branch name: `kernelgen/<kernel_name>`
- Suggest they create the PR manually with the prepared description

**If the user declines the PR**, do nothing — the changes remain as local uncommitted files
(or on the current branch if already committed).

---

## Important Notes

- **Dynamically discover everything** — never hardcode file paths or conventions. Always read
  existing code first and follow the local patterns.
- **Never overwrite existing operators** without explicit user permission (Step 3 choice).
- **Always ask the user before installing packages** — never run `pip install` without confirmation.
- **Match existing code style exactly** — imports, decorators, naming, logging, license headers.
  The generated code must look like it was written by the same team.
- **Use proper tools for all file operations**: Read to read files, Write to create new files,
  Edit to modify existing files. Never use Write to overwrite an existing file — use Edit.
- **Always run both accuracy AND performance tests** — don't stop at one.
- **Use per-dtype tolerances** in tests: fp32 → 1e-5, fp16 → 1e-2, bf16 → 2e-2.
- **Always use `ref_impl` for benchmark baselines**, not `torch.<op>` — many custom/fused
  kernels have no single torch equivalent.
- **Only rewrite to wrapper style when the repo explicitly uses wrappers**. Otherwise keep the
  MCP-generated raw Triton code — this is the safer default.
- **For raw Triton repos**: only add `@triton.autotune` if existing kernels use it. If the repo
  does not use autotune, do NOT introduce it — use a fixed BLOCK_SIZE matching the repo pattern.
- **Follow alphabetical ordering** when adding entries to `__init__.py`, `__all__`, or config lists.
- If accuracy tests fail, try to fix. If benchmark is slow, consider calling
  `mcp__kernelgen-mcp__optimize_kernel` to optimize.
- **Speedup formula**: `speedup = torch_latency / kernel_latency` (>1.0 means kernel is faster).
  If parsing fails, report raw output instead.
- **Never modify unrelated files**. Keep the diff minimal and focused.
- **Use Chinese for user-facing messages** if the user communicates in Chinese.
