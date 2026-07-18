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

# 支持的硬件平台

KernelGen Web 内置支持以下测试设备：华为昇腾（Huawei Ascend）、海光（Hygon）、天数智芯（Iluvatar）、沐曦（MetaX）、摩尔线程（Mthreads）、曦望（Sunrise）和 NVIDIA。

- **生成 Kernel**：
  - 若用户未选择测试设备，默认使用 NVIDIA。
  - 针对生成 FlagTree TLE 算子，测试设备只能为 NVIDIA。
- **特化 Kernel**：仅支持将 Kernel 从 NVIDIA 特化至华为昇腾。
