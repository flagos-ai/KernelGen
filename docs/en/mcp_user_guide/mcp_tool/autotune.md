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

# Autotune Kernel

KernelGen Operator Development MCP Toolkit integrates an MCP tool for autotuning Kernels. You only need to provide a functional description of the operator. Through multiple rounds of intelligent iteration, the KernelGen automatically generates, verifies, and continuously optimizes the operator implementation, ultimately delivering a high-performance Triton Kernel along with a complete test report.

The autotune Kernel function is supported on all testing devices. See [Supported Hardware Platforms](../../KernelGen_overview/KernelGen-overview.md). You can invoke this tool through AI agents, including VSCode (and Github Copilot), Cursor, Claude Code, and OpenClaw.

If you haven’t connected to the KernelGen Operator Development MCP Toolkit, see [Configure and connect to KernelGen Operator Development MCP Toolkit](../connect_mcp/connect-mcp.md).

To autotune kernels, a typical prompt should include the mandatory and optional elements. See such information in Auto-tune template.


## Auto-tune template

```{code-block} shell
Call the MCP tool to iteratively generate the **[operator name]** operator on **[target device]**.
Task description: [Detailed description of the operator's functionality, inputs/outputs, and constraints]
- Iterations: [N]
- Speedup target: [X]
```

### Mandatory elements

1. `Invoke the KernelGen MCP tool`
2. Operator name
3. Task description

### Optional elements (Recommended)

1. Target device (e.g., MetaX, GPU model, etc.)
2. Number of iterations (default: 3 rounds, ~10 minutes each)
3. Speedup target (default: 1.2×)


## Auto-tune examples

### Example 1 — Basic version

```{code-block} shell
Call the KernelGen MCP tool to iteratively generate the **rmsnorm** operator on MetaX.
Task description: Implement RMSNorm with an input tensor of shape `(batch, hidden_size)` and produce the normalized output.
Use defaults:
- Iterations = 3
- Speedup target = 1.2×
```


### Example 2 — Recommended version

```{code-block} shell
Call the KernelGen MCP tool to iteratively generate the **rmsnorm** operator on MetaX.
Task description: Implement an RMSNorm operator supporting both float16 and float32 inputs, with input shape `(batch, hidden_size)`. Numerical stability and parallel optimization should be considered.
Iterations: 5
Speedup target: 1.5×
```

### Example 3 — Production-grade version

```{code-block} shell
Call the KernelGen MCP tool to iteratively generate a **fused softmax** operator on an **NVIDIA GPU**.
Task description: Implement a fused softmax with masking. The input is an attention score tensor of shape `(batch, head, seq_len, seq_len)`. The goal is to minimize memory access and maximize throughput.
Iterations: 6
Speedup target: 2.0×
```

### Tips for better results

The more specific your task description, the better the auto-tuning outcome.

**Do specify:**

- Data type: `fp16` / `bf16` / `fp32`
- Tensor shape
- Whether kernel fusion is required
- Performance bottleneck: memory-bound vs. compute-bound

**Avoid:**

- Vague descriptions like *"implement an operator"*
- Omitting input/output specifications
