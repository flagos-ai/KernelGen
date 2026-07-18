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

# 概念

本节列出了 Kernel 生成的基本概念：

- **正确性**：生成的 Kernel 的输出在数值上与 PyTorch 基准测试的匹配程度。KernelGen 会在每个场景中比较 Kernel 和 PyTorch 基准测试，并输出总体正确性。只有正确性通过，该 Kernel 才能被使用。

- **加速比**：生成的 Kernel 与 PyTorch 基准测试相比运行快多少。Kernel 相对于 PyTorch 基准测试的加速比是 PyTorch 执行时间与 Kernel 执行时间的比值。

- **场景**：输入参数的特定组合。每个独特的组合对应一个不同的生成 Kernel。例如，如果输入参数包括两个张量形状和两种数据类型，那么就有四个场景。

- **KernelGen 算子开发 MCP 工具集**：这是一个符合 MCP 标准的工具集，它把 Kernel 的生成、优化和特化工具都统一整合在了一起。

- **Skills**：预编写的操作指南，用于向 AI 智能体传授完成特定任务的最佳实践。在开始生成算子等任务之前，AI 会读取相关Skills文件，以确保输出质量高且一致。
