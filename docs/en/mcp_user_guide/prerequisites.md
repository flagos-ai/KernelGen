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

# Prerequisites

- Agent client versions:

  - Claude Code version 2.1 and later

  - OpenClaw version 2026.3.2 and later

  - VSCode with Github Copilot activated

- Environment preparations:

  - Preinstall the `torch`, `triton`, and `pytest` packages to enable Kernel testing across different hardware platforms. Import `torch` through the following python code to check whether CUDA is available:
  
    ```{code-block} python
    import torch

    # Check if CUDA is available
    print("CUDA available:", torch.cuda.is_available())

    # If available, perform a simple GPU computation
    if torch.cuda.is_available():
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y  # Triggers CUDA kernel
        print("Result:", z)
        print("Device:", z.device)
    else:
        print("CUDA is not available")
    ```

    For Huawei Ascend, install and import `torch_npu` in addition to installing and importing standard `torch` package.

  - Preinstall [FlagTree](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/install.html).
