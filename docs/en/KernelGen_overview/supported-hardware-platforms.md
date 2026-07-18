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

# Supported hardware platforms

KernelGen internally integrates support for the following testing devices: Huawei Ascend, Hygon, Iluvatar, MetaX, Mthreads, Sunrise, and NVIDIA.

- **Generating Kernels**:
  - If users do not select a testing device, NVIDIA is used by default.
  - For generating FlagTree TLE operators specifically, the testing device can only be NVIDIA.
- **Specializing Kernels**: Only supports Kernel specialization from NVIDIA to Huawei Ascend.
