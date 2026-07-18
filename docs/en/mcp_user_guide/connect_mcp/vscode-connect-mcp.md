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

# Use VSCode to connect to KernelGen Operator Development MCP Toolkit

## Prerequisites

Install the GitHub Copilot extension and ensure it is activated.

## Steps

To connect VSCode to KernelGen Operator Development MCP Toolkit, perform the following steps:

1. Connect to KernelGen Operator Development MCP Toolkit：

   - **Option 1** (Recommend): Send a prompt to connect to the KernelGen Operator Development MCP Toolkit, for example:

     - `Based on the Claude Code configuration documentation: https://code.claude.com/docs/en/mcp, connect to the MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the claude.json file.`

     - `Based on the VSCode documentation: https://code.visualstudio.com/docs/copilot/customization/mcp-servers, configure the kernelgen MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the mcp.json file. `

   - **Option 2**: Manual configuration

    {style=lower-alpha}

     1. Select **File** > **Preferences**, then choose **Settings**. Navigate to **Chat** > **MCP**. In the **Server Sampling** section, click the "Edit in settings.json" link.

     2. Add the following code to the `setting.json` file.

        ```json
        {
          "servers": {
            "kernelgen-mcp": {
              "type": "sse",
              "url": "https://kernelgen.flagos.io/sse",
              "headers": {
                "Authorization": "Bearer <your KernelGen Token>"
              }
            }
          }
        }
        ```

2. Start server.

  {style=lower-alpha}

   1. Press **Ctrl**+**Shift**+**P** to open the command palette, type and search for "MCP: List Servers", then press Enter to display a list of all MCP servers currently configured in VSCode along with their running status.

   2. Select "kernelgen-mcp" from list and select "Start Server".

3. Verify KernelGen Operator Development MCP Toolkit connection, prompt：
  
  ```{code-block} shell
  Please verify the kernelgen mcp connection is successful.
  ```
