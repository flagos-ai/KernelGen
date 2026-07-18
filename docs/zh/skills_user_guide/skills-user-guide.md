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

# KernelGen Skills 用户指南

本节介绍如何使用 VSCode（及 Copilot）、Claude Code 和 OpenClaw 连接到 KernelGen 算子开发 MCP 工具集，并使用 KernelGen Skills 进行算子的通用生成。

有关为 FlagGems 或 vLLM 项目生成 Kernel、优化 Kernel以及跨硬件平台特化 Kernel 的使用案例，请参见 [KernelGen Skills 使用案例](../skills_use_case/skills-use-case.md)。

## 前提条件

- 智能体客户端版本：
  
  - Claude Code 2.1 及更高版本

  - OpenClaw 2026.3.2 及更高版本

  - 已安装并激活 GitHub Copilot 的 VSCode

- 环境准备：
  
  - 请预先安装 `torch`、`triton` 和 `pytest` 软件包，以支持在不同硬件平台上进行 Kernel 测试。对于华为昇腾（Huawei Ascend）平台，除标准 `torch` 软件包外，还需安装 `torch_npu`。

  - 预先安装 [FlagTree](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/install.html)。


```{include} ../mcp_user_guide/connect_mcp/prerequisites.md
:heading-offset: 1
:relative-docs: ..
:relative-images: ../../assets/images
```

## 将 Claude Code 连接到 KernelGen 算子开发 MCP 工具集并加载 Skills

```{include} ../mcp_user_guide/connect_mcp/claudecode-connect-mcp.md
:heading-offset: 2
:relative-docs: ..
:relative-images: ../../assets/images
```

### 配置 Skills 

1. 配置 `kernelgen-flagos` 统一 Skills ，包含所有子技能：
   - **方式一**（推荐）：发送提示词配置 `kernelgen-flagos` 统一 Skill ，例如：

      ```{code-block} shell
      请安装 kernelgen-flagos skills, 网址是 https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos。
      ```

   - **方式二**：使用命令配置

      ```{code-block} shell
      npx skills add flagos-ai/skills --skill kernelgen-flagos -a claude-code
      ```

      ```{note}
      npx 需要 npm 5.2.0 或更高版本。如果未安装 npm 或版本过旧，请运行 `npm -v` 检查当前版本并进行更新。
      ```

      **方式三**：手动配置

      在项目中创建名为 skills 的文件夹并添加Skills文件。

      您可以前往 https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos 克隆技能Skills。下载 Skills 文件并将其添加到 skills 文件夹中。

      ```{code-block} shell
      mkdir -p .claude/skills
      git clone https://github.com/flagos-ai/skills/
      cp -r skills/skills/kernelgen-flagos/ .claude/skills/
      ```

2. 配置 Skills 后，使用 **Control+C** 重启 Claude Code。

3. 验证 Skills 是否已成功配置：

   {style=lower-alpha}

   1. 方式一：使用提示词：`请验证 kernelgen-flagos skill 能否测通。`
   2. 方式二：使用斜杠命令"/"，若列表中显示 kernelgen-flagos，则说明 Skills 已安装成功。


## 将 OpenClaw 连接到 KernelGen 算子开发 MCP 工具集并加载 Skills

1. 发送提示词连接到 KernelGen 算子开发 MCP 工具集，例如：
   - `根据Claude Code 配置文档：https://code.claude.com/docs/en/mcp，连接 MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 claude.json 文件里。`
   - `根据VSCode 文档：https://code.visualstudio.com/docs/copilot/customization/mcp-servers，配置 kernelgen MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 mcp.json 文件里。`

2. 发送提示词重启 OpenClaw，因为上一步已将 KernelGen 算子开发 MCP 工具集作为 MCP Server 添加到 `openclaw.json` 配置文件中。

3. 发送提示词配置 kernelgen-flagos 统一 skill，包含所有子技能，例如：`请安装 kernelgen-flagos skills, 网址是 https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos。`

4. 向 OpenClaw 发送提示词：`将技能转换为 OpenClaw 兼容的技能后重新安装。`

5. 验证 Skills 是否已成功安装：`请验证 kernelgen-flagos skill 能否测通。`


## 将 VSCode 和 Github Copilot 连接到 KernelGen 算子开发 MCP 工具集并加载 Skills

1. 发送提示词连接到 KernelGen 算子开发 MCP 工具集，例如：
   - `根据Claude Code 配置文档：https://code.claude.com/docs/en/mcp，连接 MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 claude.json 文件里。`
   - `根据VSCode 文档：https://code.visualstudio.com/docs/copilot/customization/mcp-servers，配置 kernelgen MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 mcp.json 文件里。`

2. 启动 KernelGen 算子开发 MCP 工具集：

    {style=lower-alpha}

   1. 按 **Ctrl+Shift+P** 打开命令面板，输入并搜索"MCP: List Servers"，然后按 Enter 键，查看 VSCode 中当前所有已配置的 MCP Server 及其运行状态。
   2. 选择"kernelgen-mcp"，然后选择"Start Server"。

3. 验证 KernelGen 算子开发 MCP 工具集连接，发送提示词：`请验证 kernelgen mcp 能否测通。`

4. 发送提示词配置 kernelgen-flagos 统一 Skill，包含所有子技能，例如：`请安装 kernelgen-flagos skills, 网址是 https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos。`

5. 验证 Skills 是否已成功安装：`请验证 kernelgen-flagos skill 能否测通。`


## 通用算子生成

生成算子时，典型提示词应包含以下必填项和可选项：算子名称（必填）、任务描述（必填）、输入参数及数据类型、输出参数及数据类型、测试设备，以及算子优化的迭代轮数。

您可以使用以下任一方式调用 `kernelgen-flagos`  Skills 并生成算子：

- **方式一**：使用斜杠命令和提示词

   ```{code-block} shell
   /kernelgen-flagos 生成 ReLU 算子，该算子属于 pointwise 类别，包含一个输入参数 input：类型为 torch.Tensor 的输入张量，可具有任意形状和数据类型（通常为浮点类型），需对其应用 ReLU 激活函数；输出为一个 torch.Tensor 类型的张量，其形状与输入相同，计算逻辑为 max(0, input)，即所有负值置零而正值保持不变，请使用 MetaX 进行生成。
   ```

- **方式二**：完全使用提示词

   ```{code-block} shell
   使用 kernelgen-flagos 生成 ReLU 算子，该算子属于 pointwise 类别，包含一个输入参数 input：类型为 torch.Tensor 的输入张量，可具有任意形状和数据类型（通常为浮点类型），需对其应用 ReLU 激活函数；输出为一个 torch.Tensor 类型的张量，其形状与输入相同，计算逻辑为 max(0, input)，即所有负值置零而正值保持不变，请使用 MetaX 进行生成。
   ```

有关支持生成算子的硬件平台，请参见[支持的硬件平台](../KernelGen_overview/supported-hardware-platforms.md)。

有关为 FlagGems 或 vLLM 项目生成算子、优化算子以及跨硬件平台特化 Kernel 的使用案例，请参见 [KernelGen Skills 使用案例](../skills_use_case/skills-use-case.md)。
