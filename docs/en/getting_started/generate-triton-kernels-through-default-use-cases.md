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

# Generate Kernels through predefined use cases

After signing in to KernelGen, you will see the welcome page.
![alt text](../assets/images/welcom-en.png)
On the welcome page, perform the following steps to generate Triton
Kernels:

1. Click a predefined use case in the **Use Case** section.

   This example uses the predefined use case named **ReLU**.

   The operator definitions of the **ReLU** use case are automatically populated in the text box.

2. Click ![alt text](../assets/icons/right-arrow.png){w=20px}.

   KernelGen searches for a list of GitHub repositories to find code snippets similar to the operator definitions. For more information about the list of GitHub repositories, see [Repository List](../references/search-repo-list.md).

3. Select one or more URLs of repositories for reference or select the direct generation checkbox, and then click **Next**.

4. In the confirmation dialogue box, click **Confirm**.

5. On the operator definition and configuration page, configure the operator definition parameters and the maximum number of iterations KernelGen attempts to pass the correctness test:

   {style=lower-alpha}

   1. In the **Operator Definition** section, select an operator testing device from the **Testing Device** dropdown list.  The default is **Nvidia**.
   2. Configure the other operator definition parameters as needed.
   3. In the **KernelGen Configuration** section, increase or decrease the maximum number of iterations KernelGen attempts to pass the correctness test. The default is **5**. 
   4. Click **Next**.
   5. In the confirmation dialogue box, click **Confirm**.
   ![alt text](../assets/images/operator-definition-en.png)

6. On the Kernel generation and test page, check the generation status on the **KernelGen** panel at the right.

7. After the statuses of the **Kernel**, **CUDA Implementation**, **Correctness Test**, and **Speedup Ratio Test** change to **Completed**, and the **Correctness Test** is passed (turns to green), click **View Details**. The **Speedup Ratio Details** table lists speedup of each scenario and overall speedup.

8. If the speedup information meets your performance criteria, close the **Speedup Ratio Details**, and click **Download Kernel**. If you want to use the correctness test and speedup ratio test results, click the **Correctness Test** and **Speedup Ratio Test** sections to copy and paste the corresponding codes.
