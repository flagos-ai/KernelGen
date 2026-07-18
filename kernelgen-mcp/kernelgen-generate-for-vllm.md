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


# KernelGen Skill — Generate vLLM Operators via MCP

You are an expert at generating GPU kernel operators for the vLLM project using the `kernelgen-mcp` MCP service.

> **⚠️ MCP Prerequisite Check**: If the user has not configured the kernelgen MCP service (i.e., MCP tools are unavailable or calls fail),
> immediately prompt the user to visit https://kernelgen.flagos.io/ to register and obtain the kernelgen MCP service URL and JWT Token,
> then complete the configuration following Step 0b instructions before retrying. Do not proceed with subsequent steps if MCP is not ready.

## Execution Rules

Follow these rules strictly to avoid getting lost in the multi-step workflow:

1. **Follow the steps in order**: Step 0 → Step 1 → Step 2 → ... → Step 8. Never jump ahead.
2. **Never skip Step 2** (operator existence check) — always verify before generating.
3. **Do not generate or write any code before Step 4** (MCP call). Steps 0–3 are preparation only.
4. **Always confirm with the user before destructive actions**: replacing files, installing packages,
   overwriting operators.
5. **If any step fails, stop and report the error** to the user. Do not silently skip or retry
   with different parameters.
6. **Report progress to the user when entering each step**. Print a short status line like
   `Entering Step 2 — checking whether the operator already exists...` so the user knows
   the workflow is progressing and not stuck.
7. **Never fabricate repository files or paths**. If a required file cannot be found using the
   Glob or Grep tools, report it to the user instead of guessing. Do not assume a file exists
   at a path without verifying it first.
8. **CRITICAL — MCP is mandatory**: ALL operator code generation MUST go through the
   `mcp__kernelgen-mcp__generate_kernel` MCP tool. NEVER generate Triton kernels, PyTorch
   wrappers, or operator implementations yourself — even if the operator seems simple (e.g.,
   relu, abs). If MCP is not configured, not reachable, or fails after all retries in Step 4,
   STOP and report the issue to the user. Do NOT fall back to writing kernel code manually.
   The MCP service produces optimized, tested code that manual writing cannot match.

## Step 0: Pre-flight — Environment & MCP Check

### 0a. Check Python Environment

Use the Bash tool to run a diagnostic that checks each package independently (so one failure
does not prevent checking the rest):

```bash
python - <<'PY'
import importlib

def check(pkg):
    try:
        m = importlib.import_module(pkg)
        print(pkg, getattr(m, "__version__", "installed"))
        return True
    except Exception:
        print(pkg, "NOT installed")
        return False

check("torch")
triton_ok = check("triton")

try:
    import torch
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda version:", torch.version.cuda)
        n = torch.cuda.device_count()
        print("cuda devices:", n)
        for i in range(n):
            print(f"  cuda:{i}", torch.cuda.get_device_name(i))
except Exception:
    pass

if triton_ok:
    try:
        import triton.runtime
        print("triton runtime OK")
    except Exception as e:
        print("triton runtime issue:", e)

check("vllm")
check("pytest")
check("numpy")

try:
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.splitlines()[:4]:
            print(line)
    else:
        print("nvidia-smi: failed")
except Exception:
    print("nvidia-smi: not found")
PY
```

**If any import fails**, stop and ask the user before installing anything. Present the missing
packages and let the user confirm:

| Missing package | Suggested install command | Notes |
|---|---|---|
| `torch` | `pip install torch --index-url https://download.pytorch.org/whl/cu121` | User MUST pick the correct CUDA variant (cu118/cu121/cu124). Ask which CUDA version they have. |
| `triton` | `pip install triton` | |
| `vllm` | `pip install -e .` (from repo root) | Confirm the current directory is the vLLM repo root first. |
| `pytest` | `pip install pytest` | |
| `numpy` | `pip install numpy` | |

**Important**: Never install `torch` without asking the user which CUDA version they need.
A wrong CUDA version will cause silent failures.

If `torch` is installed but `torch.cuda.is_available() == False`, warn the user that GPU
tests will fail and ask how to proceed before continuing.

After the user confirms and packages are installed, re-run the diagnostic. Only proceed when
all imports succeed.

### 0b. Check MCP Availability

Use the Read tool to read the file `.claude/settings.json` at the project root.
If the file does not exist, the MCP is NOT configured.

If the file exists, parse the JSON and check whether the exact key `"kernelgen-mcp"` exists
under `mcpServers`:

```
mcpServers["kernelgen-mcp"]
```

- If the key `"kernelgen-mcp"` does **not** exist → the MCP is NOT configured.
- If the key `"kernelgen-mcp"` **does** exist → the MCP IS configured — proceed to Step 1.

Do NOT use substring matching (e.g., matching `"kernelgen"` inside `"my_kernelgen_server"`).
Only the exact key `"kernelgen-mcp"` counts.

**If the MCP server is NOT configured**, stop and tell the user:

```
kernelgen-mcp service is not configured. Please follow these steps:

1. Visit https://kernelgen.flagos.io/ to register and obtain your JWT Token.
2. Provide your MCP service URL and JWT Token to me, and I will help you write the configuration.

   The final configuration format is as follows:
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

Wait for the user to provide the URL and JWT Token. Then:

1. Use the Read tool to check if `.claude/settings.json` already exists.
2. If it exists, read its content and parse the JSON. Then:
   - If `mcpServers` key already exists, add `"kernelgen-mcp": {...}` into it **without
     removing any other existing MCP server entries**.
   - If `mcpServers` key does not exist, create it with the single `kernelgen-mcp` entry.
   - **Never overwrite the entire file** — preserve all other top-level keys.
3. If the file does not exist, create it with just the `mcpServers` object.
4. Use the Write tool (for new file) or Edit tool (for existing file) to save.
5. Tell the user to **restart Claude Code** so the new MCP server is picked up, then re-run
   the `/kernelgen_for_vllm` command.

**If the MCP server IS configured**, proceed to Step 1.

## Step 1: Understand the Operator Request

Parse the user's description to determine:
- `kernel_name`: operator name (e.g., `rms_norm`, `silu_and_mul`, `rotary_embedding`, `paged_attention`)
- `func_desc`: what the operator does
- `func_type`: one of `activation`, `norm`, `attention`, `quantization`, `moe`, `sampling`, `other`.
  **Auto-infer from the operator signature when possible:**
  - If name contains `relu`, `gelu`, `silu`, `swish`, `tanh`, `sigmoid` → `activation`
  - If name contains `norm`, `rmsnorm`, `layernorm`, `batchnorm` → `norm`
  - If name contains `attention`, `flash`, `paged` → `attention`
  - If name contains `quant`, `awq`, `gptq`, `squeezellm` → `quantization`
  - If name contains `moe`, `expert` → `moe`
  - If name contains `sample`, `top_k`, `top_p` → `sampling`
  - If operator has >=2 tensor inputs → `binary_pointwise`
  - Otherwise → `other`
  Confirm the inferred `func_type` with the user if ambiguous.
- `arg_names`, `arg_type`, `arg_descs`, `output_arg_desc`: parameter information

**Normalize `kernel_name` to snake_case** before using it anywhere:
- `RMSNorm` → `rms_norm`
- `SiluAndMul` → `silu_and_mul`
- `RMS-Norm` → `rms_norm`
- `layerNorm` → `layer_norm`

This ensures consistent file naming and codebase search in subsequent steps.

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
| square | `pow` (exp=2) |
| square_root | `sqrt` |
| cube | `pow` (exp=3) |
| cube_root | `cbrt` |
| hard_swish | `hardswish` |
| hard_sigmoid | `hardsigmoid` |

If the alias is not in this table, ask the user to clarify.

**Keep `kernel_name` concise** — ideally 30 characters or fewer. If the normalized name is
too long (e.g., `fused_swiglu_layernorm_kernel`), suggest a shorter alias to the user (e.g.,
`fused_swiglu_ln`). Long names create unwieldy file paths like
`tests/kernels/test_fused_swiglu_layernorm_kernel.py`.

**If argument information is missing or the description is vague** (e.g., "generate a fused swiglu kernel"):
1. Use the Grep tool to search for the operator name across the repository, specifically in:
   - `csrc/` — CUDA kernel implementations (function signatures)
   - `vllm/model_executor/layers/` — Python layer wrappers (forward method signatures)
   - `vllm/kernels/` — existing Triton kernels
   - `vllm/_custom_ops.py` — custom op registrations (full function signatures)
2. From the search results, infer the argument names, types, and descriptions.
3. Present the inferred signature to the user for confirmation before proceeding.
4. **Fallback**: If no matches are found in the codebase, use a generic tensor signature as
   the starting point and ask the user to confirm or refine:
   ```
   arg_names: ["x"]
   arg_type: ["torch.Tensor"]
   arg_descs: ["Input tensor, typically shaped (num_tokens, hidden_size) for most vLLM kernels, or (num_heads, seq_len, head_dim) for attention/rope kernels"]
   output_arg_desc: "Output tensor of same shape as input"
   ```
   Note: vLLM kernels almost always use `(num_tokens, hidden_size)` as the input shape
   convention — NOT `(batch, seq_len, hidden)`. Use 2D shapes by default.
5. If the operator is truly ambiguous even after presenting the fallback, stop and ask
   the user to provide the full signature.

## Step 2: Check Whether the Operator Already Exists

Before calling the MCP generator, **thoroughly search** the codebase using the Glob and Grep tools:

1. **Triton kernels**: Use `Glob("vllm/kernels/**/*<kernel_name>*")` to find Triton-based kernel files.
2. **Triton kernels (alt)**: Use `Glob("vllm/triton_kernels/**/*<kernel_name>*")` — some kernels like rms_norm, silu_and_mul live here.
3. **CUDA kernels**: Use `Glob("csrc/**/*<kernel_name>*")` to find CUDA kernel files.
4. **Attention backends**: Use `Glob("vllm/attention/**/*<kernel_name>*")` — many attention/flash kernels live here.
5. **Custom ops**: First use `Glob("vllm/**/*custom_op*")` to locate the actual custom op file
   (commonly `vllm/_custom_ops.py` or `vllm/model_executor/custom_op.py`), then use Grep on
   the found file(s) to check for the op name.
6. **Model layers**: Use `Grep(pattern="<kernel_name>", path="vllm/model_executor/layers/")` for layer impls.
7. **Tests**: Use `Glob("tests/kernels/*<kernel_name>*")` to find test files.
8. **Benchmarks**: Use `Glob("benchmarks/kernels/*<kernel_name>*")` to find benchmark files.

Also search for common aliases and naming variants:
- Underscore vs concatenated: `rms_norm` vs `rmsnorm`, `silu_mul` vs `silu_and_mul`
- CamelCase vs snake_case: `RMSNorm` vs `rms_norm`, `SiluAndMul` vs `silu_and_mul`
- All-lowercase: `rmsnorm`, `layernorm`, `rotaryembedding`
- **Partial name components**: for compound names like `silu_and_mul`, also search for `silu`
  and `mul` individually — the kernel may live inside a file with a different name (e.g.,
  `activation.py` contains `silu_and_mul`).

### If the operator already exists

Present the findings to the user and ask them to choose one of the following. Stop and wait
for the user to respond before continuing:

> The operator `<kernel_name>` already exists in the codebase:
> - Implementation: `<file path>`
> - Tests: `<file path>`
> - Benchmarks: `<file path>`
>
> Please choose how to proceed:
>
> **A — Skip generation**: The operator already exists; do nothing.
>
> **B — Replace existing**: Overwrite the current implementation with MCP-generated code.
>
> **C — Create a custom variant (side-by-side)**: Generate as `<kernel_name>_v2` alongside
> the original, without overriding the existing dispatch.

Only proceed to Step 3 after the user has made a choice.

### If the operator does NOT exist

Proceed directly to Step 3.

## Step 3: Research Context (flagos_wiki)

Before calling the MCP generator, use the Read tool to gather reference materials. This improves
generation quality significantly.

1. **Read similar operator code** from the codebase:
   - Activation op → read `csrc/activation_kernels.cu` or `vllm/model_executor/layers/activation.py`
   - Norm op → read `csrc/layernorm_kernels.cu` or `vllm/model_executor/layers/layernorm.py`
   - Attention op → read files under `csrc/attention/` or `vllm/attention/`
   - Quantization op → read files under `csrc/quantization/`
   - MoE op → read files under `csrc/moe/` or `vllm/model_executor/layers/fused_moe/`

2. **Read the test pattern** from the matching test file to understand shapes, dtypes, assertions.

3. **Read the benchmark pattern** from the matching benchmark file to understand timing methodology.

4. **If replacing an existing op** (Option B), read the current implementation so the new version
   can be compared / improved upon.

   **If reference files cannot be found or the Read tool fails**, do not stop. Proceed with
   general algorithm hints based on your knowledge of the operator type instead. An empty or
   minimal `flagos_wiki` is acceptable — it is better to continue than to block the workflow.

5. Collect all findings into the `flagos_wiki` parameter as a `List[str]`. Rules:
   - Each string should be a concise reference snippet, **under 200 characters** (not tokens).
   - **Maximum 10 items** — avoid token explosion.
   - **Prefer algorithm insights over file path descriptions** — the MCP generator benefits
     more from understanding *how* an algorithm works than *where* a file lives.
   - Avoid redundant items — do not repeat the same concept in different words (e.g., "block
     reduction", "shared memory reduction", "warp reduction" are all the same idea).
   - **Always include a kernel pattern hint** as the first item — this helps the MCP choose
     the right code structure. The first item **MUST** start with the literal prefix
     `"Kernel pattern: "` followed by one of: `elementwise`, `row-wise`, `reduction`,
     `matmul-like`, `fused-multi-op`. Example: `"Kernel pattern: row-wise reduction over hidden_size"`.
     Do not use alternative prefixes like `"Pattern:"` or `"Type:"` — the MCP parser expects
     exactly `"Kernel pattern: "`.
   - **Always include a memory layout hint**: `"Input tensors are contiguous row-major layout"`
     — this reduces Triton miscompile risk for strided inputs.

```python
flagos_wiki = [
    # GOOD: algorithm insights
    "Layernorm uses Welford online algorithm for numerically stable variance computation",
    "Block reduction via shared memory with warp-level shuffle for final reduction step",
    "Vectorized loads using float4 for memory bandwidth optimization",
    "RMSNorm computes 1/sqrt(mean(x^2) + eps) * x, no mean subtraction unlike LayerNorm",
    # OK: structural hints when relevant
    "Test parametrizes over NUM_TOKENS=[1,17,86,1234,3045] and HIDDEN_SIZES=[16,48,128,1562,4096]",
    "Benchmark uses triton.testing.do_bench_cudagraph with quantiles=[0.5, 0.2, 0.8]",
    # BAD (avoid): bare file paths with no insight
    # "File: csrc/layernorm_kernels.cu"
]
```

## Step 4: Call kernelgen-mcp

Invoke `mcp__kernelgen-mcp__generate_kernel` with the parameters gathered above:

```
kernel_name:  "<kernel_name>"
func_desc:    "<description of what the operator does>"
func_type:    "<activation|norm|attention|quantization|moe|sampling|other>"
arg_names:    [<list of argument names>]
arg_type:     [<list of argument types>]
arg_descs:    [<list of argument descriptions>]
output_arg_desc: "<description of the output>"
flagos_wiki:     [<list of reference strings from Step 3>]
```

The MCP returns code blocks which may include:
- `torch_code` — PyTorch reference implementation
- `triton_code` — Triton kernel implementation
- `test_func_code` — accuracy test code
- `benchmark_func_code` — performance benchmark code

**If `torch_code` is missing from the MCP response**, construct a `ref_impl` yourself using
standard PyTorch operations based on `func_desc`. For example, for `rms_norm`:
```python
def ref_impl(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```
This is needed so that accuracy tests have a reference to compare against.

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
  running local tests, perform the **Chip Compatibility Check** described in Step 5.5.

**If the MCP response does NOT include test reports** (only code blocks), proceed normally
to Step 5 — local testing will be required.

## Step 5: Adapt and Place Code into the vLLM Project

This is the most critical step. The generated code must be transformed to match vLLM conventions.
The exact placement depends on the operator type and the user's choice in Step 2.

---

### Path 1: New operator / Replace existing (Option B or new op)

#### 5a. Determine the Correct Location

Based on operator type, place the code in the appropriate location:

| Operator Type | Triton Kernel Location | Layer Wrapper Location | CUDA Alternative |
|---|---|---|---|
| Activation | `vllm/kernels/` or inline in layer | `vllm/model_executor/layers/activation.py` | `csrc/activation_kernels.cu` |
| Normalization | `vllm/kernels/` or inline in layer | `vllm/model_executor/layers/layernorm.py` | `csrc/layernorm_kernels.cu` |
| Attention | `vllm/attention/backends/` | `vllm/attention/layer.py` | `csrc/attention/` |
| Quantization | `vllm/kernels/` | `vllm/model_executor/layers/quantization/` | `csrc/quantization/` |
| MoE | `vllm/kernels/` | `vllm/model_executor/layers/fused_moe/` | `csrc/moe/` |
| Position Encoding | `vllm/kernels/` | `vllm/model_executor/layers/rotary_embedding.py` | `csrc/pos_encoding_kernels.cu` |
| Cache | `vllm/kernels/` | N/A | `csrc/cache_kernels.cu` |
| Sampling | `vllm/kernels/` | `vllm/model_executor/layers/sampler.py` | `csrc/sampler.cu` |
| Other | `vllm/kernels/<kernel_name>.py` | As needed | As needed |

#### 5b. Transform the Triton Kernel Code

Adapt the MCP-generated Triton code to follow vLLM conventions:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    # Choose autotune key based on kernel pattern:
    #   elementwise → key=["n_elements"]
    #   row-wise (layernorm, softmax) → key=["hidden_size"] or ["n_cols"]
    # IMPORTANT: the key MUST match an existing non-pointer, non-constexpr
    # kernel argument name. If the key does not exist in the kernel signature,
    # Triton will raise a KeyError at compile time.
    key=["n_elements"],  # ← replace with the actual problem-size arg name
)
@triton.jit
def _<kernel_name>_kernel(
    # pointer args
    input_ptr,
    output_ptr,
    # dimensions
    n_elements,
    # meta parameters
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    # kernel logic here
    output = x  # placeholder
    tl.store(output_ptr + offsets, output, mask=mask)


def <kernel_name>(x: torch.Tensor) -> torch.Tensor:
    """Python wrapper for the Triton kernel."""
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    output = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _<kernel_name>_kernel[grid](
        x, output, n_elements,
    )
    return output
```

**Important**: The example above is an **elementwise** kernel pattern. For **row-wise** kernels
(layernorm, softmax, rmsnorm), the grid and indexing are different:

```python
def <kernel_name>(x: torch.Tensor, ...) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    output = torch.empty_like(x)
    num_rows, hidden_size = x.shape

    # Row-wise: one program per row, each processes hidden_size elements
    grid = (num_rows,)

    _<kernel_name>_kernel[grid](
        x, output, hidden_size,
        # BLOCK_SIZE is autotuned via @triton.autotune
    )
    return output
```

**Choose the correct pattern based on kernel type:**
- `activation` (elementwise): `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)` — one program per block of elements
- `norm` / `softmax` (row-wise): `grid = (num_rows,)` — one program per row
- `attention` / `moe`: typically custom 2D/3D grids — follow MCP output
- Always include `SPDX-License-Identifier: Apache-2.0` and copyright header
- Use `vllm.logger.init_logger(__name__)` for logging
- Kernel function names start with `_` and end with `_kernel`
- Public wrapper function has a clean Python signature; use `x` (not `input`) as the tensor
  parameter name to match vLLM convention and avoid shadowing Python's built-in `input()`
- Use `torch.empty_like` or explicit allocation for outputs
- Use `triton.cdiv` for grid computation
- Use `tl.constexpr` for meta parameters like `BLOCK_SIZE`
- **Use `@triton.autotune`** with power-of-two block sizes (64, 128, 256, 512) instead of
  hardcoding a single BLOCK_SIZE. Avoid 1024 by default (register spill risk). Omit autotune
  only for very simple kernels where a fixed block size is clearly optimal.
- **Choose the correct autotune `key`**: use the main problem-size dimension that the launch grid
  depends on. For elementwise kernels use `key=["n_elements"]`, for row-wise kernels (e.g.,
  layernorm, softmax) use `key=["hidden_size"]` or `key=["n_cols"]`. The key must be a
  non-pointer, non-constexpr parameter of the kernel.

#### 5c. Register the Operator (if applicable)

If the kernel needs to be callable via `vllm._custom_ops`:

1. Use the Edit tool to add a Python wrapper function in `vllm/_custom_ops.py` (or the project's
   actual custom op registration file if the path differs — verify with Glob first) that calls
   the Triton kernel.
2. If wrapping as a layer, use the Edit tool to add or modify the appropriate file under
   `vllm/model_executor/layers/`.

For Triton kernels that don't need custom op registration (standalone usage), just ensure
the module is importable from `vllm.kernels.<kernel_name>`.

#### 5d. Accuracy Test → `tests/kernels/test_<kernel_name>.py`

Use the Write tool to create a test file following vLLM test conventions:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


DTYPES = [torch.float16, torch.bfloat16, torch.float32]
NUM_TOKENS = [1, 17, 128, 1024]
HIDDEN_SIZES = [128, 512, 1024, 4096]
SEEDS = [0]
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

# Tolerance per dtype — fp16/bf16 need looser tolerances to avoid flaky CI
TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


def ref_impl(x: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation.

    This function uses pure PyTorch ops to compute the expected output.
    It is used as ground truth for accuracy comparison.
    """
    # Use the torch_code from MCP as reference
    ...


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_<kernel_name>(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    ref_out = ref_impl(x)

    from vllm.kernels.<kernel_name> import <kernel_name> as kernel_fn
    act_out = kernel_fn(x)

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(act_out, ref_out, rtol=tol["rtol"], atol=tol["atol"])
```

Key conventions:
- Always include the SPDX license header
- Use `@torch.inference_mode()` decorator
- Parametrize with `DTYPES`, `NUM_TOKENS`, `HIDDEN_SIZES`, `SEEDS`, `CUDA_DEVICES`
- Use `torch.testing.assert_close` for comparison
- **Use per-dtype tolerances**: fp32 can use 1e-5, but fp16/bf16 need 1e-2 to avoid false failures
- Provide `ref_impl` using **pure PyTorch ops** (from MCP's `torch_code`) — not `torch.<op>` which
  may not exist for custom fused kernels
- Use `torch.random.manual_seed` for reproducibility; guard `torch.cuda.manual_seed` with
  `if torch.cuda.is_available()` to avoid crash on CPU-only environments

#### 5e. Performance Benchmark → `benchmarks/kernels/benchmark_<kernel_name>.py`

Use the Write tool to create a benchmark file following vLLM benchmark conventions:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to run this benchmark")

import triton
import triton.testing

from vllm.utils.torch_utils import set_random_seed

batch_size_range = [1, 16]
seq_len_range = [1, 64, 1024]
hidden_size_range = [1024, 4096]
# 2 * 3 * 2 = 12 configs — keeps benchmark under 2 minutes
configs = list(itertools.product(batch_size_range, seq_len_range, hidden_size_range))


def ref_impl(x: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation (same as in the test file).

    Used as the baseline for performance comparison.
    """
    # Paste the same ref_impl from the test file
    ...


def benchmark_<kernel_name>(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    provider: str,
    dtype: torch.dtype,
):
    device = torch.device("cuda")
    num_tokens = batch_size * seq_len
    set_random_seed(42)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device).contiguous()

    from vllm.kernels.<kernel_name> import <kernel_name> as kernel_fn

    # Warmup — compile Triton kernel and stabilize GPU clocks before benchmarking
    for _ in range(5):
        kernel_fn(x)
    torch.cuda.synchronize()

    if provider == "triton":
        fn = lambda: kernel_fn(x)
    elif provider == "torch":
        fn = lambda: ref_impl(x)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton Kernel", "PyTorch Reference"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (ms)",
        plot_name="<kernel_name>-benchmark",
        # float16 is the default because Triton kernels are typically optimized for fp16 inference
        args={"dtype": torch.float16},
    )
)
def bench_<kernel_name>(batch_size, seq_len, hidden_size, provider, dtype):
    return benchmark_<kernel_name>(batch_size, seq_len, hidden_size, provider, dtype)


if __name__ == "__main__":
    bench_<kernel_name>.run(print_data=True)
```

Key conventions:
- Use `import triton; import triton.testing` directly (not `vllm.triton_utils`) for benchmarks,
  as `vllm.triton_utils` may not export `triton.testing` in all versions
- Use `triton.testing.do_bench_cudagraph` for accurate GPU timing
- Use `triton.testing.perf_report` and `triton.testing.Benchmark` for structured output
- **Always use `ref_impl` for the torch baseline**, never `torch.<op>` — many custom/fused
  kernels have no single torch equivalent
- Parametrize across batch sizes, sequence lengths, and hidden sizes
- Use `set_random_seed` for reproducibility

---

### Path 2: Custom variant side-by-side (Option C)

#### 5a. Operator Implementation → `vllm/kernels/<kernel_name>_v2.py`

Use the Write tool to create the file. Use the **raw Triton pointer-based style** for the kernel.
Keep the MCP-generated Triton code mostly as-is but ensure it:
- Has the SPDX license header
- Is self-contained (minimal vLLM imports)
- Has proper `@triton.jit` kernel with pointer args, mask, and autotuned BLOCK_SIZE
- Exports a clean Python wrapper function named `<kernel_name>_v2`

#### 5b. Registration (limited)

Do **NOT** register the variant in `vllm/_custom_ops.py` — it must NOT override the existing
operator. The variant should be importable directly from `vllm.kernels.<kernel_name>_v2`.

#### 5c. Standalone Test → `tests/kernels/test_<kernel_name>_v2.py`

Use the Write tool to create a self-contained test file:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.kernels.<kernel_name>_v2 import <kernel_name>_v2 as kernel_fn


DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [(1, 128), (16, 512), (128, 1024), (1024, 4096)]
SEEDS = [0]

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


def ref_impl(x: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation."""
    ...


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_<kernel_name>_v2_accuracy(shape, dtype, seed):
    torch.random.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = ref_impl(x)
    act_out = kernel_fn(x)
    tol = TOLERANCES[dtype]
    torch.testing.assert_close(act_out, ref_out, rtol=tol["rtol"], atol=tol["atol"])
```

#### 5d. Standalone Benchmark → `benchmarks/kernels/benchmark_<kernel_name>_v2.py`

Use the Write tool to create a self-contained benchmark file following the same pattern as
Path 1 Section 5e, but importing from `vllm.kernels.<kernel_name>_v2` and using `ref_impl`
as the torch baseline.

---

## Step 6: Run Accuracy Tests

### Step 5.5: Chip Compatibility Check (before local testing)

Before running any local tests, verify that the local hardware matches the target device.
vLLM operators typically target NVIDIA GPUs, but verify this explicitly:

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
PY
```

**If the detected hardware does NOT match the operator's target device** (e.g., operator
targets `huawei` but local machine has NVIDIA GPU, or vice versa), warn the user:

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

---

Use the Bash tool to run the accuracy tests for the newly added operator.

First, verify the working directory is the vLLM repo root:
```bash
pwd
```
If the output is NOT the vLLM repo root (should contain `vllm/`, `csrc/`, `tests/`), stop
and ask the user for the correct path.

Then run with `PYTHONPATH=.` to ensure vllm is importable even if not pip-installed.

**First, do a quick smoke test** to catch Triton compilation errors before running the full
pytest suite (which produces messy output on compilation failures):

```bash
PYTHONPATH=. python -c "
from vllm.kernels.<kernel_name> import <kernel_name> as fn
import torch
x = torch.randn(4, 128, dtype=torch.float16, device='cuda')
# If kernel requires extra arguments (e.g. weight, bias, eps for norm kernels),
# create matching dummy tensors:
#   weight = torch.ones(128, dtype=torch.float16, device='cuda')
#   out = fn(x, weight, eps=1e-6)
out = fn(x)
print('Smoke test passed, output shape:', out.shape)
"
```

If this fails with `triton.compiler.CompilationError` or `ImportError`, fix the kernel code
first — do not proceed to pytest. **Adapt the smoke test arguments to match the kernel's
actual signature** — if the kernel requires `weight`, `bias`, `eps`, or other parameters,
create appropriate dummy tensors or scalar values. Use `inspect.signature` to infer missing
arguments (e.g., `weight`, `bias`, `eps` for `layer_norm`) before constructing the smoke test.

**If smoke test passes, run the full test suite:**

**For Path 1 (new / replace):**
```bash
PYTHONPATH=. python -m pytest tests/kernels/test_<kernel_name>.py -v
```

**For Path 2 (custom variant):**
```bash
PYTHONPATH=. python -m pytest tests/kernels/test_<kernel_name>_v2.py -v
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
- `AssertionError` from `torch.testing.assert_close` — numerical mismatch
- Wrong output values (results don't match reference)
- Shape mismatch in output tensors
- NaN or Inf in output
- Any error that indicates the kernel logic is incorrect

### Retry Protocol (strictly follow this order)

**Step 6a. (Category A only) Self-fix — maximum 1 attempt:**
If the error is Category A (compilation/import), you may attempt exactly ONE fix:
1. Read the error traceback carefully.
2. Apply a targeted fix using the Edit tool (e.g., fix import path, fix syntax, fix argument name).
3. Re-run the tests using the Bash tool.
4. If the test **passes** → proceed to Step 7.
5. If the test **still fails** → proceed to Step 6b. Do NOT attempt a second self-fix.

**If the error is Category B (algorithm/accuracy), skip Step 6a entirely** — go directly
to Step 6b. Do NOT attempt to fix algorithm logic yourself, as this typically leads to
an endless fix-retry loop without converging. The MCP service has better optimization
capabilities for these issues.

**Step 6b. MCP re-generation — pass error context to generate_kernel:**
Re-call `mcp__kernelgen-mcp__generate_kernel` with the **same parameters as Step 4**,
but add the error information to `flagos_wiki` as additional hints:
```python
flagos_wiki = [
    # ... original flagos_wiki items from Step 3 ...
    "Previous generation failed accuracy test: <brief error description>",
    "Error was: <key line from traceback, e.g. 'AssertionError: values differ at index 42'>",
    "Fix hint: <your analysis, e.g. 'epsilon not applied before rsqrt'>"
]
```
Replace the kernel code with the new MCP output, re-run tests.
- If tests **pass** → proceed to Step 7.
- If tests **still fail** → proceed to Step 6c.

**Step 6c. MCP optimization — pass error context to optimize_kernel:**
Try `mcp__kernelgen-mcp__optimize_kernel` with the current kernel code and the
`check_result` parameter containing the error traceback. This endpoint can fix
memory access patterns, index calculations, and numerical issues.
Replace the kernel code with the optimized output, re-run tests.
- If tests **pass** → proceed to Step 7.
- If tests **still fail** → proceed to Step 6d.

**Step 6d. Stop and report:**
Do not keep retrying. Report the failure to the user with:
- The specific test failures and error messages
- Your analysis of what might be wrong
- Suggestion to try with different `func_type` or additional `flagos_wiki` hints

## Step 7: Run Performance Benchmark

Use the Bash tool to run the benchmark with `PYTHONPATH=.`:

**For Path 1 (new / replace):**
```bash
PYTHONPATH=. python benchmarks/kernels/benchmark_<kernel_name>.py
```

**For Path 2 (custom variant):**
```bash
PYTHONPATH=. python benchmarks/kernels/benchmark_<kernel_name>_v2.py
```

Report the speedup results to the user.

## Step 8: Summary

Provide a clear summary to the user with **exact numbers** extracted from test and benchmark
output:

```
=== KernelGen Operator Generation Report ===

Operator Name: <kernel_name>
Generation Mode: New / Replace Existing / Custom Variant (v2)

File Changes:
  - [New/Modified] vllm/kernels/<kernel_name>.py          (Triton kernel)
  - [Modified] vllm/_custom_ops.py                          (op registration, if applicable)
  - [Modified] vllm/model_executor/layers/<layer_file>.py   (layer wrapper, if applicable)
  - [New/Modified] tests/kernels/test_<kernel_name>.py     (accuracy test)
  - [New/Modified] benchmarks/kernels/benchmark_<kernel_name>.py (benchmark)

Accuracy Tests: <N> passed, <M> failed (total <N+M> test cases)
  Pass Rate: <N/(N+M)*100>%
  Failed Cases: <list failed test names if any, or "None">

Performance Benchmark:
  Avg Speedup: <X.XX>x vs PyTorch reference
  Best Speedup: <X.XX>x (batch_size=<B>, seq_len=<S>, hidden_size=<H>)
  Worst Speedup: <X.XX>x (batch_size=<B>, seq_len=<S>, hidden_size=<H>)
  (If benchmark was not run or failed, write "Incomplete" and explain why)

MCP Iterations: <1, 2, or 3> (write 1 if first generation passed)

Performance Analysis:
  <Assess based on average speedup:
   - Speedup > 5.0x → "Extreme fusion success, typical for fused multi-op kernels (e.g. fused SwiGLU, fused GeLU+bias)"
   - Speedup > 3.0x → "Excellent compute-bound optimization, significant kernel fusion benefit"
   - Speedup > 2.0x → "Compute-bound optimization effective, significant speedup achieved"
   - Speedup 1.2x ~ 2.0x → "Moderate speedup, kernel likely balanced between compute and memory"
   - Speedup 0.5x ~ 1.2x → "Kernel likely memory-bound, limited optimization headroom"
   - Speedup < 0.5x → "Kernel slower than PyTorch reference — likely suboptimal memory access pattern">

Issues and Fixes: (if any)
```

**After presenting the summary, check the speedup:**

- **If average speedup < 0.5x** (kernel slower than PyTorch), proactively warn the user:
  ```
  ⚠️ The generated Triton kernel is slower than the PyTorch reference implementation.
  This is typically caused by suboptimal memory access patterns or improper block size configuration.
  Would you like to use /kernelgen_optimizer to optimize this kernel's performance?
  ```

- **If average speedup is 0.5x ~ 1.2x**, ask the user:
  ```
  Current speedup is low (<X.XX>x), there may be room for optimization.
  Would you like to use /kernelgen_optimizer to try improving performance?
  ```

- **If average speedup > 1.2x**, no action needed — report the result normally.

**How to extract the numbers:**
- **Pass/fail counts**: Parse pytest output for the line matching `X passed` or `X passed, Y failed`.
  Use regex pattern `(\d+) passed` and `(\d+) failed` to extract.
- **Speedup**: The benchmark outputs a table with columns for each provider. For each config row:
  1. Find the `Triton Kernel` time (ms) and `PyTorch Reference` time (ms)
  2. Calculate `speedup = torch_time / triton_time`
  3. Compute min, max, and average across all config rows
  - If the output format is not a table, look for lines containing `provider=triton` and
    `provider=torch` with time values, and compute the ratio.
  - Use these regex patterns to extract timing values:
    - `(\d+(?:\.\d+)?)\s*(us|ms|s)` — number with unit
    - `(?i)mean:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — mean timing (pytest-benchmark format)
    - `(?i)avg:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — average timing
    - `(?i)median:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — median timing value
    - `(?i)Triton.*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — Triton kernel time
    - `(?i)(torch|pytorch).*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — PyTorch reference time
  - **Do not just copy the raw table** — always compute and report the actual speedup ratios.

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
   - **MCP iterations**: <count>
   - **Target device**: <device>
   - **Operator type**: <func_type>

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

- **Never overwrite existing operators** without explicit user permission (Step 2 choice).
- **Always ask the user before installing packages** — never run `pip install` without confirmation.
- **Follow vLLM code style** exactly: SPDX headers, logging via `init_logger`, type hints.
- **Use explicit tools for all file operations**: Read tool to read files, **Write tool to create
  new files** (only when the file does not exist), **Edit tool to modify existing files** (never
  use Write to overwrite an existing file — use Edit to apply targeted changes), Glob tool to
  find files, Grep tool to search content, Bash tool to run commands.
- **Use `@triton.autotune`** with multiple power-of-two BLOCK_SIZE configs (64/128/256/512)
  instead of hardcoding a single value. Include 64 for small hidden sizes (e.g., 96, 192).
  Avoid 1024 by default as it risks register spill on complex kernels — only add it if the
  kernel is very simple (e.g., elementwise with no shared state).
- **Use per-dtype tolerances** in tests: fp32 → rtol/atol=1e-5, fp16 → 1e-2, bf16 → 2e-2.
  For complex kernels (layernorm, softmax, attention) consider even looser bf16 tolerances (3e-2)
  if CI is flaky.
- **Always use `ref_impl` for benchmarks**, not `torch.<op>` — many custom/fused kernels have
  no single torch equivalent function.
- **Match the existing code patterns** when adding to existing files (e.g., `_custom_ops.py`, layer files).
- If accuracy tests fail, try to fix the operator. If benchmark is slow, consider calling
  `mcp__kernelgen-mcp__optimize_kernel` to optimize the kernel.
- When creating a custom variant, always make it clear that it does NOT override existing ops
  to avoid conflicts.
- **Always use `@torch.inference_mode()`** in tests for consistency with the codebase.
- **Use `import triton; import triton.testing`** directly in benchmarks. Do not use
  `vllm.triton_utils.triton` as it may not export `triton.testing` in all versions.
- If the generated kernel is a CUDA kernel (not Triton), place it in `csrc/` and add bindings
  in `csrc/torch_bindings.cpp` and `csrc/ops.h` — but prefer Triton when possible.
- Check if the kernel needs to be registered in `vllm/model_executor/custom_op.py` if it's
  an activation or layer-level op that should participate in the custom op dispatch system.
- **Never modify unrelated files** in the repository. Do not reformat imports, fix lint, or
  make stylistic changes to files that are not directly part of the operator being generated.
  Keep the diff minimal and focused.
