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

# Use Claude Code to connect to KernelGen Operator Development MCP Toolkit

## Prerequisites

- Use Claude Code version 2.1 and later
- Learn Claude Code settings: <https://code.claude.com/docs/en/settings#>.

## Steps

To connect Claude Code to the KernelGen Operator Development MCP Toolkit, perform the following steps:

1. Use the Server-Sent Events (SSE) protocol and Bear authentication to register the KernelGen Operator Development MCP Toolkit with Claude Code:

   - **Option 1** (Recommended): Send a prompt to connect to the KernelGen Operator Development MCP Toolkit, for example:

     - `Based on the Claude Code configuration documentation: https://code.claude.com/docs/en/mcp, connect to the MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the claude.json file.`

     - `Based on the VSCode documentation: https://code.visualstudio.com/docs/copilot/customization/mcp-servers, configure the kernelgen MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the mcp.json file.`
  
   - **Option 2** : Use the following command:

     ```bash
     claude mcp add --transport sse kernelgen-mcp https://kernelgen.flagos.io/sse/ --header "Authorization: Bearer <your KernelGen Token>"
     ```

   - **Option 3**: Manually modify the configuration file.
  
      - **Option A**: Add JSON configuration to the `.claude.json` file

          ```{code-block} json
          {
            "projects": {
              "/root/projects/my-project": {
                "mcpServers": {
                  "kernelgen-mcp": {
                    "type": "sse",
                    "url": "https://kernelgen.flagos.io/sse",
                    "headers": {
                      "Authorization": "Bearer <your KernelGen Token>"
                    }
                  }
                }
              }
            }
          }
          ```

      - **Option B**：Create `mcp.json` file, and add JSON configuration.

        ```{code-block} python
        {
          "mcpServers": {
            "kernelgen_mcp": {
              "url": "http://kernelgen.flagos.io/sse",
              "headers": {
                "Authorization": "Bearer <your KernelGen Token>"
              }
            }
          }
        }
        ```

2. Verify KernelGen Operator Development MCP Toolkit connection：

   - Option 1: Use prompt

    ```{code-block} shell
    Please verify the kernelgen mcp connection is successful.
    ```

- Option 2: Use command

    ```{code-block} shell
    /mcp
    ```
