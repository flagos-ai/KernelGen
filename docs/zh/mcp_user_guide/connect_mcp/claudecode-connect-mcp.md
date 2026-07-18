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

# 使用 Claude Code 连接到 KernelGen 算子开发 MCP 工具集

## 前提条件

- 使用 Claude Code 2.1 及更高版本。
- 了解 Claude Code 设置：<https://code.claude.com/docs/en/settings#>。

## 步骤

按照以下步骤将 Claude Code 连接到 KernelGen 算子开发 MCP 工具集：

1. 使用 Server-Sent Events（SSE）协议和 Bearer 认证方式，将 KernelGen 算子开发 MCP 工具集注册到 Claude Code：

   - **方式一**（推荐）：发送提示词连接到 KernelGen 算子开发 MCP 工具集，例如：

     - `根据Claude Code 配置文档：https://code.claude.com/docs/en/mcp，连接 MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 claude.json 文件里。`

     - `根据VSCode 文档：https://code.visualstudio.com/docs/copilot/customization/mcp-servers，配置 kernelgen MCP，MCP 的 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。配置在 mcp.json 文件里。`
  
   - **方式二**：使用以下命令：

     ```bash
     claude mcp add --transport sse kernelgen-mcp https://kernelgen.flagos.io/sse/ --header "Authorization: Bearer <your KernelGen Token>"
     ```

   - **方式三**：手动修改配置文件。
  
      - **方式 A**：在 `.claude.json` 文件中添加 JSON 配置：

          ```{code-block} json
          {
            "projects": {
              "/root/projects/my-project": {
                "mcpServers": {
                  "kernelgen-mcp": {
                    "type": "sse",
                    "url": "https://kernelgen.flagos.io/sse",
                    "headers": {
                      "Authorization": "Bearer <你的 KernelGen Token>"
                    }
                  }
                }
              }
            }
          }
          ```

      - **方式 B**：创建 `mcp.json` 文件并添加 JSON 配置：

        ```{code-block} json
        {
          "mcpServers": {
            "kernelgen_mcp": {
              "url": "http://kernelgen.flagos.io/sse",
              "headers": {
                "Authorization": "Bearer <你的 KernelGen Token>"
              }
            }
          }
        }
        ```

2. 验证 KernelGen 算子开发 MCP 工具集连接：

   - 方式一：使用提示词

    ```{code-block} shell
    请验证 kernelgen mcp 能否测通。
    ```

   - 方式二：使用命令

    ```{code-block} shell
    /mcp
    ```
