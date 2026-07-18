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

# Release Notes

This section includes the KernelGen release information.

## V2.1.0

- **Added features**
  - Added support for the Sunrise AI accelerator platform, expanding the number of supported chips from six to seven.

- **Enhanced features**
  - Experimental TLE support has been extended to the Web platform.

## V2.0.0

- **Added features**：
  - Added support for creating, optimizing, and specializing Kernels across multiple hardware platforms via the KernelGen Operator Development MCP Toolkit, AI agents, and first- and third-party skills.
  - Added support for generating the FlagTree TLE operator on NVIDIA hardware platforms. TLE operator generation capability is an experimental feature currently under active development.
  - Extended testing device support to include MetaX platform.
  - Introduced a History panel to track modifications to the Kernel code.

## V1.0.0

- **Added features**：
  - Basic Workflow: Supports single, fixed-step code generation including GroundTruth → TritonKernel → Correctness Test → Performance Test.
  - User Registration: Supports self-service registration and application, with platform approval required for trial use.
  - Web Interface: Online access at [https://kernelgen.flagos.io/](https://kernelgen.flagos.io/).
  - Core Features:
    - Fully Automated Workflow: Automatically generates, tests, and optimizes complete AI operator sets.
    - Multi-Backend Support: Seamlessly supports multiple AI libraries and chips with automatic adaptation and debugging.
    - Easy-to-Use: Browser-based interface requiring no setup or prior experience.
    - Standardized Verification: Automatically generates test cases to ensure operator correctness.
    - Deep Ecosystem Integration: Collaborates with FlagGems and FlagTree to accelerate operator library development.
