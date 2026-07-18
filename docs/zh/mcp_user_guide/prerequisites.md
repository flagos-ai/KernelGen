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

# 前置条件

- 智能体客户端版本
  
  - Claude Code 版本 2.1 及以上
  
  - OpenClaw 版本 2026.3.2 及以上
  
  - VSCode 需启用 Github Copilot

- 环境准备
  
  - 预安装依赖包：请预先安装 `torch`、`triton` 和 `pytest` 软件包，以便在不同硬件平台上进行 Kernel 测试。通过以下 Python 代码导入 `torch` 已检测 CUDA 是否可用：

    ```{code-block} python
    import torch

    # 检查 CUDA 是否可用
    print("CUDA available:", torch.cuda.is_available())

    # 如果可用，执行一个简单的 GPU 计算
    if torch.cuda.is_available():
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y  # 触发 CUDA kernel
        print("Result:", z)
        print("Device:", z.device)
    else:
        print("CUDA is not available")
    ```

    针对华为昇腾（Huawei Ascend）平台：除了安装和导入标准的 `torch` 包外，还需安装并导入 `torch_npu`。

- 预安装 FlagTree：请预先安装 [FlagTree](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/install.html)。
