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

# 为 FlagGems 或 vLLM 项目生成 Kernel

## 前提条件

生成 Kernel 之前，请确保您已阅[前置条件](../mcp_user_guide/prerequisites.md)，并完成本节预安装步骤。

预先从源码安装 FlagGems 或 vLLM。

- KernelGen Skills 支持 FlagGems 项目，请参见下方*预安装 FlagGems*章节。

- KernelGen Skills 支持 vLLM 项目，请参见 [vLLM 用户指南](https://docs.vllm.ai/en/latest/getting_started/installation/)。同时，请安装 [FlagTree](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/install.html)。

## 预安装 FlagGems

安装信息请参见 [FlagGems 文档](https://docs.flagos.io/projects/FlagGems/en/latest/getting_started/install.html#)。通过 requirements 文本文件安装 FlagTree。

## 生成 Kernel

无论是使用 VSCode（配合 Copilot）、Claude Code 还是 OpenClaw，为 FlagGems 或 vLLM 项目生成算子的总体流程都是相似的。这包括连接到 KernelGen 算子开发 MCP 工具包以及加载技能，具体步骤请参考 [KernelGen Skills 用户指南](../skills_user_guide/skills-user-guide.md) 。你只需要在“通用算子生成”章节所述的提示词基础上，额外补充一句 **“将内核集成到 FlagGems 中”**。KernelGen 会自动检测 FlagGems 是否已安装，并将生成的输出文件直接提交到项目的实验目录中。
