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

# kernelgen-flagos：统一 GPU 算子生成技能

## 概述

`kernelgen-flagos` 是一个统一的 AI 编程技能，通过 `kernelgen-mcp` MCP 服务生成 GPU 算子。它自动检测目标仓库类型，并分发到相应的专用工作流。

### 解决的问题

编写高性能 GPU 算子复杂且容易出错。不同项目（FlagGems、vLLM、自定义 Triton 仓库）各自有独特的算子实现规范、测试模式和注册系统。此前，用户需要分别安装多个技能。

本统一技能将 **九个子技能** 打包为一个，用户只需安装一次：

| 子技能 | 用途 |
|---|---|
| **kernelgen-generate** | 为任意 Python/Triton 仓库生成 GPU 算子 |
| **kernelgen-generate-for-flaggems** | FlagGems 专用（`pointwise_dynamic`、`_FULL_CONFIG`、分类测试） |
| **kernelgen-generate-for-vllm** | vLLM 专用（SPDX 头、`init_logger`、`@triton.autotune`、自定义算子注册） |
| **kernelgen-optimize** | 通用 Triton 内核优化，通过 MCP 迭代循环进行多轮优化 |
| **kernelgen-optimize-for-flaggems** | FlagGems 专用优化（3 种模式：内置/外部/实验性算子） |
| **kernelgen-optimize-for-vllm** | vLLM 专用优化，含 CustomOp 注册和集成 |
| **kernelgen-specialize** | 平台特化，将 Triton 算子迁移到目标平台（如 GPU → 昇腾 NPU） |
| **kernelgen-specialize-for-flaggems** | 平台特化 + FlagGems 框架集成，支持 4 种集成模式 |
| **kernelgen-submit-feedback** | 通过 GitHub Issues 或邮件提交 bug 报告 |

### 使用方式

```bash
# 生成算子（自动检测仓库类型）
/kernelgen-flagos relu

# 指定函数类型
/kernelgen-flagos rms_norm --func-type normalization

# 优化已有算子
/kernelgen-flagos optimize <算子源目录> [目标加速比] [最大迭代次数]
```

技能会自动：
- 检测到 **FlagGems** 仓库 → 使用 FlagGems 专用工作流
- 检测到 **vLLM** 仓库 → 使用 vLLM 专用工作流
- 其他情况 → 使用通用工作流，动态发现仓库结构

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `operator_name` | 是 | — | snake_case 格式的算子名称（如 `relu`、`rms_norm`、`silu_and_mul`） |
| `--func-type` | 否 | 自动推断 | 函数类型分类（根据检测到的仓库类型有所不同） |

### 反馈

在生成过程中遇到任何问题，只需说"提交反馈"或"报告 bug"，技能会引导你通过 GitHub issue 或邮件提交反馈。

---

## 工作原理

```
┌──────────────────────────────────────────────────────────┐
│  阶段 1   检测仓库类型                                     │
│           ├── FlagGems? → kernelgen-generate-for-flaggems.md │
│           ├── vLLM?     → kernelgen-generate-for-vllm.md    │
│           └── 其他?     → kernelgen-generate.md              │
│                                                          │
│  阶段 2   执行选定的子技能工作流                            │
│           （环境检查 → MCP 生成 →                          │
│            代码适配 → 测试 → 性能基准测试）                  │
│                                                          │
│  阶段 2b  优化（按需）                                     │
│           ├── FlagGems? → kernelgen-optimize-for-flaggems.md │
│           ├── vLLM?     → kernelgen-optimize-for-vllm.md    │
│           └── 其他?     → kernelgen-optimize.md              │
│                                                          │
│  阶段 2c  平台特化（按需）                                   │
│           ├── FlagGems? → kernelgen-specialize-for-flaggems.md │
│           └── 其他?     → kernelgen-specialize.md              │
│                                                          │
│  阶段 3   反馈处理（按需）                                  │
│           → kernelgen-submit-feedback.md                  │
└──────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
skills/kernelgen-flagos/
├── SKILL.md                              # 统一入口（路由逻辑）
├── kernelgen-generate.md                 # 通用生成子技能
├── kernelgen-generate-for-flaggems.md    # FlagGems 专用生成子技能
├── kernelgen-generate-for-vllm.md        # vLLM 专用生成子技能
├── kernelgen-optimize.md                 # 通用优化子技能
├── kernelgen-optimize-for-flaggems.md    # FlagGems 专用优化子技能
├── kernelgen-optimize-for-vllm.md        # vLLM 专用优化子技能
├── kernelgen-specialize.md               # 平台特化子技能
├── kernelgen-specialize-for-flaggems.md  # FlagGems 专用特化子技能
├── kernelgen-submit-feedback.md          # 反馈提交子技能
├── LICENSE.txt                           # Apache 2.0 许可证
├── README.md                             # 英文文档
└── README_zh.md                          # 本文档（中文版）
```

---

## 安装方式

### 快速安装（通过 npx）

```bash
# 安装统一的 kernelgen 技能（包含所有子技能）
npx skills add flagos-ai/skills --skill kernelgen-flagos -a claude-code

# 或一次安装所有 Flagos 技能
npx skills add flagos-ai/skills -a claude-code
```

### 手动安装

```bash
# 在项目根目录下
mkdir -p .claude/skills
cp -r <path-to-this-repo>/skills/kernelgen-flagos .claude/skills/
```

---

## 许可证

本项目基于 Apache 2.0 许可证。详见 [LICENSE.txt](LICENSE.txt)。
