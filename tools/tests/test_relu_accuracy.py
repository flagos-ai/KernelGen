import bench
from bench.sandbox.test.test_parametrize import parametrize, label
from bench.sandbox.config import DEVICE as device
from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from bench.sandbox.utils.accuracy_utils import to_reference
import torch


@label("relu")
@parametrize("shape", [(512,), (64, 128), (8, 32, 64), (2, 4, 8, 16), (1,)])
@parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_relu(shape, dtype):
    # initialize the input data based on the parameters
    input = torch.randn(shape, dtype=dtype, device=device)

    # cast to reference dtype if necessary
    ref_input = to_reference(input, True)

    ref_out = bench.relu(ref_input)
    res_out = bench.triton.relu(input)

    assert_close(res_out, ref_out, dtype, reduce_dim=1)
