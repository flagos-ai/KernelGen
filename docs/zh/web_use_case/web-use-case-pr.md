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

# Web 平台使用案例

本用例介绍如何将 KernelGen 生成的 **Kernel**、**CUDA版基准实现**、**正确性测例** 和 **加速比测例** 代码贡献到 [FlagGems](https://github.com/flagos-ai/FlagGems) GitHub。

流程如下：

1. 生成 **Kernel**、**CUDA版基准实现**、**正确性测例** 和 **加速比测例** 代码。更多信息请参阅 [通过您的算子定义生成Kernel](../web_user_guide/generate-triton-kernels-through-your-operator-definitions.md)。
2. 将这些代码分别保存为文件。
3. 转换这些文件。
4. 将转换后的文件贡献到 FlagGems GitHub。

```{note}
以 KernelGen 中预定义的 **ReLU** 用例为例。在本用例中，我们假设 ReLU 算子是一个你刚刚通过 KernelGen 生成但尚未贡献到 FlagGems GitHub 的新算子。
```

## 转换从 KernelGen 生成的文件

转换从 KernelGen 生成的文件：

1. 将四个文件重命名如下：

   - `relu_triton.py`: 此文件包含 **Kernel** 代码。
   - `relu_baseline.py`: 此文件包含 **CUDA 版基准实现** 代码。
   - `test_relu_accuracy.py`: 此文件包含 **正确性测例** 代码。
   - `test_relu_performance.py`: 此文件包含 **加速比测例** 代码。

1. 克隆 FlagGems GitHub 仓库。

    ```bash
    git clone https://github.com/flagos-ai/FlagGems
    ```

2. 克隆 KernelGen GitHub 仓库。

    ```bash
    git clone https://github.com/flagos-ai/KernelGen
    ```

3. 导航到 KernelGen 项目根目录的`tools`目录下。

    ```bash
    cd /your/project/KernelGen/tools
    ```

4. 运行以下脚本，将文件转换为两个与 FlagGems 兼容的文件。

    ```bash
    python kernelgen_to_flaggems.py \
        ./tests \
        ./output \
        relu
    ```

    转换后的文件如下：
   - `./output/relu.py`：包含 Kernel 代码。此代码与 KernelGen 中的相同。
   - `./output/relu_test.py`: 包含转换后的正确性测试和加速比测试代码。

## 在 FlagGems GitHub 上创建拉取请求

您现在可以在 [FlagGems](https://github.com/flagos-ai/FlagGems) GitHub 上创建拉取请求。

确保两个转换后的文件被正确放置，如下所示：

- 将 `relu.py` 文件放置在：
    `src/flag_gems/experimental_ops`
- 将 `relu_test.py` 文件放置在：
    `src/flag_gems/experimental_ops/exp_tests`
