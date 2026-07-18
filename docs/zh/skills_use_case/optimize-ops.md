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

# 优化 Kernel

您可以使用 VSCode（及 Copilot）、Claude Code 或 OpenClaw 在 **NVIDIA** 上优化 Kernel。

优化 Kernel 时，典型的提示词应包含以下必填和可选要素：

算子名称（必填）、任务描述（必填），以及优化迭代次数。

请确保您已阅[前置条件](../mcp_user_guide/prerequisites.md)。



## 步骤

如果您尚未连接到 KernelGen 算子开发 MCP 工具集并加载Skills，请参见 [KernelGen Skills 用户指南](../skills_user_guide/skills-user-guide.md)；否则，请使用以下任一方式调用 `kernelgen-flagos` Skills 并优化 Kernel：

- **方式一**：使用斜杠命令和提示词

   ```{code-block} shell
   /kernelgen-flagos 优化 index_put 算子，共进行 5 次迭代。
   ```

- **方式二**：完全使用提示词

   ```{code-block} shell
   使用 kernelgen-flagos 优化 index_put 算子，共进行 5 次迭代。
   ```
