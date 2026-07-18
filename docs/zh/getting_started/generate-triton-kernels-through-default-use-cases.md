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

# 通过预定义用例生成Kernel

登录 KernelGen 后，您将看到欢迎页面。
![alt text](../assets/images/welcom-ch.png)

在欢迎页面上，执行以下步骤生成 Triton Kernel：

1. 在 **典型案例** 部分点击一个预定义用例。

   此示例使用名为 **ReLU** 的预定义用例。

   **ReLU** 用例的算子定义将自动填充到文本框中。

2. 点击 ![alt text](../assets/icons/right-arrow.png){w=20px}。

   KernelGen 将搜索 GitHub 仓库列表，以查找与算子定义相似的代码片段。有关 GitHub 仓库列表的更多信息，请参见 [仓库列表](../references/search-repo-list.md)。

3. 选择一个或多个仓库 URL 作为参考，或选中直接生成复选框，然后点击 **下一步**。

4. 在确认对话框中，点击 **确认**。

5. 在算子定义和配置页面上，配置算子定义参数以及 KernelGen 尝试通过正确性测试的最大迭代轮次：

   {style=lower-alpha}

   1. 在 **算子定义** 部分，从 **评测设备** 下拉列表中选择一个算子测试设备。默认值为 **Nvidia**。
   2. 根据需要配置其他算子定义参数。
   3. 在 **KernelGen配置** 部分，增加或减少 KernelGen 尝试通过正确性测试的最大迭代轮次。默认值为 **5**。
   4. 点击 **下一步**。
   5. 在确认对话框中，点击 **确认**。
   ![alt text](../assets/images/operator-definition-ch.png)

6. 在Kernel生成和测试页面上，查看右侧 **代码生成** 面板上的生成状态。

7. 当 **Kernel**、**CUDA版基准实现**、**正确性测例** 和 **加速比测例** 的状态变为 **已完成**，并且 **正确性测例** 通过（变为绿色）后，点击 **查看详情**。**加速比详情** 表格列出了每个场景的加速比和整体加速比。

8. 如果加速比信息满足您的性能标准，关闭 **加速比详情**，然后点击 **下载Kernel**。如果您想使用正确性测例子和加速比测例的结果，点击 **正确性测例** 和 **加速比测例** 部分以复制并粘贴相应的代码。
