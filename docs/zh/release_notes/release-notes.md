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

# 发版说明

本节包含 KernelGen 的版本发布信息。

## V2.1.0

- **新增功能**
  - 新增曦望（Sunrise）AI 加速器平台，支持的芯片从 6 款扩展至 7 款。

- **功能增强**
  - 实验性 TLE 支持扩展至 Web 平台。

## V2.0.0

- **新增功能**：
  - 新增支持通过 KernelGen 算子开发 MCP 工具集、AI 智能体以及第一方和第三方 Skills ，跨多个硬件平台创建、优化和特化 Kernel。
  - 新增支持在 NVIDIA 硬件平台上生成 FlagTree TLE 算子。TLE 算子生成功能为实验性功能，目前正在积极开发中。
  - 扩展测试设备支持，新增沐曦（MetaX）平台。
  - 新增历史记录面板，用于追踪 Kernel 代码的修改记录。

## V1.0.0

- **新增功能**：
  - 基础工作流：支持单一固定步骤的代码生成，流程为 GroundTruth → TritonKernel → 正确性测试 → 性能测试。
  - 用户注册：支持自助注册申请，试用需经平台审核。
  - Web 界面：可通过 [https://kernelgen.flagos.io/](https://kernelgen.flagos.io/) 在线访问。
  - 核心功能：
    - 全自动工作流：自动生成、测试并优化完整的 AI 算子集。
    - 多后端支持：无缝支持多种 AI 库和芯片，自动适配与调试。
    - 易于使用：基于浏览器的界面，无需任何配置或使用经验。
    - 标准化验证：自动生成测试用例，确保算子正确性。
    - 深度生态集成：与 FlagGems 和 FlagTree 协同，加速算子库开发。
