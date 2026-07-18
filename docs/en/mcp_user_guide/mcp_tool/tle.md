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


# TLE Kernel

You can use MCP tool to generate kernels with TLE on NVIDIA. This section uses sparse\_mla Kernel as an example. For information about this Kernel, see [FlagTree Documentation](https://docs.flagos.io/projects/FlagTree/en/latest/user_guide/examples.html#).

Before generating the TLE operator, preinstall the FlagTree branch 3.6.x. See [Install FlagTree.](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/install.html).

To generate a TLE kernel, a typical prompt should include the following mandatory and optional elements: “Invoke MCP tools” (mandatory), operator name (mandatory), and task description (mandatory).

Prompt example:

```{code-block} python
Invoke MCP tools to generate the TLE ReLU operator.
```

**Note**: The TLE operator generation capability is an experimental feature currently under active development.
