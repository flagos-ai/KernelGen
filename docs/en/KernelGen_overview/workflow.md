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

# Workflow

The diagram below provides a brief overview of how KernelGen assists in Kernel generation.

However, user configurations may vary depending on the selected platform, AI agent, skill, and specific use case. For more information, see [KernelGen Web Platform User Guide](../web_user_guide/web-user-guide.md), [KernelGen Operator Development MCP Toolkit User Guide](../mcp_user_guide/mcp-user-guide.md), and [KernelGen Skills User Guide](../skills_user_guide/skills-user-guide.md).

![alt text](../assets/images/KernelGen-workflow-en.png)

The generation process is as follows:

1. **Collect Kernel information**: User enters semantic operator definitions into KernelGen, for example, by referring to the [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#ReLU) operator definitions. KernelGen collects operator basic parameters from the definitions.
2. **Search code snippets**: KernelGen searches code snippets similar to user's definitions as references and extracts Kernel parameters. During this step, the user can select to use the searched code snippets or not.
3. **Generate Kernel code and CUDA implementation code**: KernelGen generates codes of Kernel and CUDA implementation. CUDA implementation is used as a PyTorch benchmark reference.
4. **Test Kernel based on CUDA implementation code**：KernelGen tests Kernel based on the PyTorch benchmark, and outputs the test results of Correctness and Speedup Ratio.
