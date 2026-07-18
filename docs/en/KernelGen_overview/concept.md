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

# Concepts

This section lists the basic concepts for Kernel generation:

- **Correctness**: How closely the output of the generated Kernel matches the PyTorch benchmark numerically. KernelGen compares the Kernel and PyTorch benchmark in each scenario and outputs an overall correctness. Only correctness is passed, the Kernel can be used.

- **Speedup**: How much faster the generated Kernel runs compared to a PyTorch benchmark. The speedup of a Kernel over a PyTorch benchmark is the ratio of the PyTorch execution time to the Kernel execution time.

- **Scenario**: A specific combination of input parameters. Each unique combination maps to a differently generated Kernel. For example, if input parameters include two tensor shapes and two data types, there are four scenarios.

- **KernelGen Operator Development MCP Toolkit**: An MCP-compliant toolkit that unifies Kernel generation, optimization, and specialization tools.

- **Skills**: Pre-written instruction guides that teach AI agents the best practices for completing specific tasks. Before starting a task like generating operators, the AI reads the relevant Skill file to ensure high-quality, consistent output.
