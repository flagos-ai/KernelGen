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

# Specialize a Kernel

You can use either VSCode (and Copilot), Claude Code, or OpenClaw to migrate CUDA-implemented operators to Huawei Ascend.

To specialize a Kernel, a typical prompt should include the following mandatory and optional elements: Operator name（mandatory）and task description (mandatory).

Make sure you read the [prerequisites](../mcp_user_guide/prerequisites.md).

## Steps

If you haven't connected to the KernelGen Operator Development MCP Toolkit and load skills, see [KernelGen Skills User Guide](../skills_user_guide/skills-user-guide.md), otherwise use one of the following methods to invoke the `kernelgen-flagos` skill and specialize a Kernel:

- **Option 1**: Use the slash command and prompt

   ```{code-block} python
   /kernelgen-flagos Migrate the CUDA-implemented operator fused/silu_and_mul.py to the Ascend chip, with the operator file stored in the FlagGems repository, and the directory is _ascend/fused/silu_and_mul.py, ensuring that the accuracy verification passes.
   ```

- **Option 2**: Completely use prompt

   ```{code-block} python
   Use kernelgen-flagos to migrate the CUDA-implemented operator fused/silu_and_mul.py to the Ascend chip, with the operator file stored in the FlagGems repository, and the directory is _ascend/fused/silu_and_mul.py, ensuring that the accuracy verification passes.
   ```
