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

# Generate a Kernel for FlagGems or vLLM project

## Prerequisites

Before generating a kernel, make sure you read the [prerequisites](../mcp_user_guide/prerequisites.md) and accomplish the pre-installation steps in this section.

  Preinstall FlagGems or vLLM from source:

- KernelGen Skills support FlagGems project, see the next *Preinstall FlagGems* section.

- KernelGen Skills support vLLM project, see [vLLM user guide](https://docs.vllm.ai/en/latest/getting_started/installation/).

## Preinstall FlagGems

For installation information, see [FlagGems Documentation](https://docs.flagos.io/projects/FlagGems/en/latest/getting_started/install.html#). The FlagTree is installed while installing the requirement text file.

## Generate a kernel

Using VSCode (and Copilot), Claude Code, or OpenClaw to generate an operator for the FlagGems or vLLM project follows a similar general process (including connecting to the KernelGen Operator Development MCP Toolkit and load skills) in [KernelGen Skills User Guide](../skills_user_guide/skills-user-guide.md). You only need to add **"Integrate the kernel into FlagGems"** additionally to the prompt documented in the "Generate an operator generally" section. KernelGen automatically detects if FlagGems is installed and submits the output files to the project's experimental directory.
