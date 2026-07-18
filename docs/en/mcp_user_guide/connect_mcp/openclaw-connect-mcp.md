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

# Use OpenClaw to connect to KernelGen Operator Development MCP Toolkit

## Prerequisites

Use OpenClaw version 2026.3.2 and later

## Steps

To connect OpenClaw to the KernelGen Operator Development MCP Toolkit, perform the following steps:

1. Send a prompt to connect to the KernelGen Operator Development MCP Toolkit, for example:

   - `Based on the Claude Code configuration documentation: https://code.claude.com/docs/en/mcp, connect to the MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the claude.json file.`

   - `Based on the VSCode documentation: https://code.visualstudio.com/docs/copilot/customization/mcp-servers, configure the kernelgen MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the mcp.json file. `

    **Note**: If the current OpenClaw version does not support MCP, you can setup `mcporter` via prompt or command．The following is the command example.

    ```{code-block} shell
    "npx skills add steipete/clawdis@mcporter -g -y"
    ```

2. Verify KernelGen Operator Development MCP Toolkit connection, prompt：
  
  ```{code-block} shell
  Please verify the kernelgen mcp connection is successful.
  ```
