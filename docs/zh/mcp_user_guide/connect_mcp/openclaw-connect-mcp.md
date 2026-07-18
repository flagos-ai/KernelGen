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

# 使用 OpenClaw 连接到 KernelGen 算子开发 MCP 工具集

## 前提条件

使用 OpenClaw 2026.3.2 及更高版本。

## 步骤

按照以下步骤将 OpenClaw 连接到 KernelGen 算子开发 MCP 工具集：

1. 发送提示词连接到 KernelGen 算子开发 MCP 工具集，例如：

   - `根据Claude Code 配置文档：https://code.claude.com/docs/en/mcp，连接 MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 claude.json 文件里。`

   - `根据VSCode 文档：https://code.visualstudio.com/docs/copilot/customization/mcp-servers，配置 kernelgen MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 mcp.json 文件里。`

    **注意**：如果当前 OpenClaw 版本不支持 MCP，可通过提示词或命令安装 `mcporter`。以下为命令示例：

    ```{code-block} shell
    "npx skills add steipete/clawdis@mcporter -g -y"
    ```

2. 验证 KernelGen 算子开发 MCP 工具集连接, 发送提示词：
  
  ```{code-block} shell
  请验证 kernelgen mcp 能否测通。
  ```
