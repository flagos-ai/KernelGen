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

# 自动调优 Kernel

KernelGen 算子开发 MCP 工具集集成了用于自动调优 Kernel 的 MCP 工具。您只需提供算子的功能描述，KernelGen 即可通过多轮智能迭代自动生成、验证并持续优化算子实现，最终交付高性能 Triton Kernel 及完整的测试报告。

所有测试设备均支持自动调优 Kernel 功能。请参见[支持的硬件平台](../../KernelGen_overview/KernelGen-overview.md)。您可以通过 AI 智能体调用此工具，包括 VSCode（及 Github Copilot）、Cursor、Claude Code 和 OpenClaw。

如果尚未连接到 KernelGen 算子开发 MCP 工具集，请参见[配置并连接到 KernelGen 算子开发 MCP 工具集](../connect_mcp/connect-mcp.md)。

自动调优 Kernel 时，典型提示词应包含必填项和可选项。详细信息请参见自动调优模板。


## 自动调优模板

```{code-block} shell
调用 MCP 工具，在 [测试设备] 上迭代生成 [算子名称] 算子。
任务描述：[详细描述算子功能、输入输出、约束条件]
迭代轮数：[N]
加速比目标：[X]
```

### 必填项

1. `调用 KernelGen MCP 工具`
2. 算子名称
3. 任务描述

### 可选项（推荐填写）

1. 目标设备（例如：沐曦（MetaX）、GPU 型号等）
2. 迭代轮数（默认：3 轮，每轮约 10 分钟）
3. 加速比目标（默认：1.2×）


## 自动调优示例

### 示例一 — 基础版

```{code-block} shell
调用 kernelgen MCP 工具，在 MetaX 上迭代生成 rmsnorm 算子。
任务描述：实现 RMSNorm，输入为 (batch, hidden_size) 的张量，输出归一化结果。
使用默认：
-  迭代轮数 = 3 
-  加速比 = 1.2 
```


### 示例二 — 推荐版

```{code-block} shell
调用 kernelgen MCP 工具，在 MetaX 上迭代生成 rmsnorm 算子。
任务描述：实现 RMSNorm 算子，支持 float16 和 float32 输入，
输入 shape 为 (batch, hidden_size)，
需要考虑数值稳定性和并行优化。
迭代轮数：5
加速比目标：1.5
```

### 示例三 — 工程化版

```{code-block} shell
调用 kernelgen MCP 工具，在 英伟达 GPU 上迭代生成 fused softmax 算子。
任务描述：实现带 mask 的 fused softmax，
输入为 attention score tensor (batch, head, seq_len, seq_len)，
要求减少显存访问、提升吞吐。
迭代轮数：6
加速比目标：2.0
```

### 获得更佳结果的技巧

任务描述越具体，自动调优效果越好。

**建议指定：**

- 数据类型：`fp16` / `bf16` / `fp32`
- 张量形状
- 是否需要算子融合
- 性能瓶颈：访存密集型还是计算密集型

**避免：**

- 模糊描述，如"实现一个算子"
- 省略输入/输出规格
