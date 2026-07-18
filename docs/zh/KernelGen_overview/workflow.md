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

# 工作流程

下图简要概述了 KernelGen 如何协助完成 Kernel 生成。

但用户的具体配置可能因所选平台、AI 智能体、Skills 及具体用例的不同而有所差异。更多信息请参阅[KernelGen Web 平台用户指南](../web_user_guide/web-user-guide.md)、[KernelGen 算子开发 MCP 工具集用户指南](../mcp_user_guide/mcp-user-guide.md)和 [KernelGen Skills 用户指南](../skills_user_guide/skills-user-guide.md)。

![alt text](../assets/images/KernelGen-workflow-ch.png)

生成过程如下：

1. **收集 Kernel 信息**：用户向 KernelGen 输入语义的算子定义，例如参考 [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#ReLU) 算子定义。KernelGen 从定义中收集算子的基本参数。
2. **搜索代码片段**：KernelGen 搜索与用户定义相似的代码片段作为参考，并提取 Kernel 参数。在此步骤中，用户可以选择是否使用搜索到的代码片段。
3. **生成 Kernel 代码和 CUDA 版基准实现代码**：KernelGen 生成 Kernel 代码及 CUDA 版基准实现代码。CUDA 版基准实现代码将作为 PyTorch 基准参考。
4. **基于 CUDA 版基准实现代码测试 Kernel**：KernelGen 基于 PyTorch 基准测试 Kernel，并输出正确性和加速比的测试结果。
