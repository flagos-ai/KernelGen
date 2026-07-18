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

# 使用 VSCode 连接到 KernelGen 算子开发 MCP 工具集

## 前提条件

安装 GitHub Copilot 扩展并确保其已激活。

## 步骤

按照以下步骤将 VSCode 连接到 KernelGen 算子开发 MCP 工具集：

1. 连接到 KernelGen 算子开发 MCP 工具集：

   - **方式一**（推荐）：发送提示词连接到 KernelGen 算子开发 MCP 工具集，例如：

     - `根据Claude Code 配置文档：https://code.claude.com/docs/en/mcp，连接 MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <your KernelGen Token>。配置在 claude.json 文件里。`

     - `根据VSCode 文档：https://code.visualstudio.com/docs/copilot/customization/mcp-servers，配置 kernelgen MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <your KernelGen Token>。配置在 mcp.json 文件里。`

   - **方式二**：手动配置

    {style=lower-alpha}

     1. 选择 **File** > **Preferences**，然后选择 **Settings**。导航至 **Chat** > **MCP**。在 **Server Sampling** 区域，点击"Edit in settings.json"链接。

     2. 将以下代码添加到 `settings.json` 文件中：

        ```json
        {
          "servers": {
            "kernelgen-mcp": {
              "type": "sse",
              "url": "https://kernelgen.flagos.io/sse",
              "headers": {
                "Authorization": "Bearer <你的 KernelGen Token>"
              }
            }
          }
        }
        ```

2. 启动服务器。

  {style=lower-alpha}

   1. 按 **Ctrl**+**Shift**+**P** 打开命令面板，输入并搜索"MCP: List Servers"，然后按 Enter 键，即可查看 VSCode 中当前所有已配置的 MCP Server 及其运行状态。

   2. 从列表中选择"kernelgen-mcp"，然后选择"启动服务器"。

3. 验证 KernelGen 算子开发 MCP 工具集连接，发送提示词：
  
  ```{code-block} shell
  请验证 kernelgen mcp 能否测通。
  ```
