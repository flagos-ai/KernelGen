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

# KernelGen Web Platform Use Cases

This use case introduces how to contribute the KernelGen generated **Kernel**, **CUDA Implementation**, **Correctness Test**, and **Speedup Ratio Test** codes to [FlagGems](https://github.com/flagos-ai/FlagGems) GitHub.

The process is as follows:

1. Generate **Kernel**, **CUDA Implementation**, **Correctness Test**, and **Speedup Ratio Test** codes. For more information, see [](../web_user_guide/generate-triton-kernels-through-your-operator-definitions.md).
2. Save the codes as files, respectively.
3. Convert the files.
4. Contribute the converted file to FlagGems GitHub.

```{note}
The predefined use case **ReLU** in KernelGen is used as an example. In this use case, we assume that the ReLU operator is a new operator you just generated though KernelGen but you have not contributed it to FlagGems GitHub.
```

## Convert files generated from KernelGen

To convert files generate from KernelGen:

1. Rename the four files as follows:

   - `relu_triton.py`: This file includes the **Kernel** code.
   - `relu_baseline.py`: This file includes the **CUDA Implementation** code.
   - `test_relu_accuracy.py`: This file includes the **Correctness Test** code.
   - `test_relu_performance.py`: This file includes the **Speedup Ratio Test** code.

2. Clone the FlagGems GitHub repository.

    ```bash
    git clone https://github.com/flagos-ai/FlagGems
    ```

3. Clone the KernelGen GitHub repository。

    ```bash
    git clone https://github.com/flagos-ai/KernelGen
    ```

4. Navigate to the `tools` directory of the KernelGen project.

    ```bash
    cd /your/project/KernelGen/tools
    ```

5. Run the following script to convert these files into two FlagGems-compatible files.

    ```bash
    python kernelgen_to_flaggems.py \
        ./tests \
        ./output \
        relu
    ```

    The converted files are as follows:

    - `/tmp/output/relu.py`: Includes the Kernel code. This code is the same as it in KernelGen.
    - `/tmp/output/relu_test.py`: Includes the converted Correctness test and Speedup Ratio codes.

## Create a pull request in FlagGems github

You can now create a pull request in [FlagGems](https://github.com/flagos-ai/FlagGems) GitHub.

Ensure the two converted files are placed properly as follows:

- Place the `relu.py` file at:
  `src/flag_gems/experimental_ops`
- Place the `relu_test.py` file at:
  `src/flag_gems/experimental_ops/exp_tests`
