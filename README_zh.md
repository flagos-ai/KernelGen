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

# KernelGen

新一代多芯片系统AI辅助内核工程，Flag-OS家族的新成员。

[English](./README.md) | 中文


# ‍概述

KernelGen 是一个基于 FlagOS 生态系统的 AI 驱动自动 Triton 内核开发平台。它为算子（内核）的生成、优化、测试和跨硬件平台部署提供了完全自动化的工作流程。

随着 2.0 版本的发布，KernelGen 演进为一个**完整的 AI 原生内核工程系统**，引入了基于 MCP 的自动化、IDE 集成技能、增强的 Web 功能以及先进的 Triton 语言扩展。

体验地址：https://kernelgen.flagos.io
MCP 服务 (ModelScope)：https://www.modelscope.cn/mcp/servers/flagos-ai/FlagOS_KernelGen

---

# 核心特性（增强版）

- 全自动化工作流程
  基于 MCP + AI 代理的端到端内核生命周期自动化
- 多后端支持
  广泛兼容各类 AI 框架和硬件平台
- AI 原生开发体验
  与 IDE、代理和开发者工作流程深度融合
- 标准化验证
  自动正确性和性能验证
- 深度生态集成
  与以下组件无缝协作：
  - FlagGems
  - FlagTree
  - FlagOS 基础设施

---

# 核心能力对比

KernelGen 2.0 将 Triton 内核开发从固定流水线转变为完全 AI 原生、代理驱动的系统 —— 实现跨硬件和代码仓库的自动生成、优化和集成。

---

## 内核开发与优化

| 特性 | 版本 1.0 | 版本 2.0 |
|--------|------------|------------|
| **工作流程类型** | 固定步骤（线性流水线） | 代理式（迭代与自适应） |
| **错误处理** | 手动调试 | 自动错误修复（日志驱动） |
| **优化** | 基础性能测试 | 自动调优 + AI 驱动优化 |
| **测试** | 基础正确性与性能测试 | 全自动测试生成（正确性 + 基准测试） |
| **内核生命周期管理** | 部分 | 完整生命周期（生成 → 优化 → 测试 → 集成） |

---

## 硬件与性能能力

| 特性 | 版本 1.0 | 版本 2.0 |
|--------|------------|------------|
| **多硬件适配** | 支持 | 智能自动适配与特化 |

---

## 开发者体验

| 特性 | 版本 1.0 | 版本 2.0 |
|--------|------------|------------|
| **接口** | 仅 Web 浏览器 | Web + IDE + CLI (MCP) |
| **开发入口** | 仅 Web UI | 自然语言 + CLI + AI 代理 |
| **IDE / 代理集成** | 不支持 | Claude Code / VS Code / OpenClaw / MCP 代理 |
| **用户生产力** | 辅助开发 | 全自动开发 |

---

## 集成与生态

| 特性 | 版本 1.0 | 版本 2.0 |
|--------|------------|------------|
| **代码仓库集成** | 手动下载与集成 | 通过 Skills 自动生成 PR |
| **Web 平台功能** | 基础 UI | 算子历史追踪 + 增强 UX |
| **生态集成** | FlagOS 基础集成 | 与 FlagGems / FlagTree / Skills 深度集成 |
| **目标用户** | Triton 开发者 | Triton 开发者 + AI 原生开发者 |

---

如果您有任何建议或问题，欢迎在本仓库的 issues 中记录。
