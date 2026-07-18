# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


# Fast path: contiguous memory with no broadcasting
@triton.jit
def bitwise_not_kernel_contiguous(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for contiguous tensors - direct linear access."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = ~x
    tl.store(out_ptr + offsets, y, mask=mask)


# General kernel for strided/broadcasted tensors (up to MAX_DIMS)
@triton.jit
def bitwise_not_kernel(
    x_ptr,  # input tensor
    out_ptr,  # output tensor
    n_elements: tl.int64,  # total number of elements in the output
    # shapes (padded to MAX_DIMS)
    s0: tl.int64, s1: tl.int64, s2: tl.int64, s3: tl.int64, s4: tl.int64, s5: tl.int64, s6: tl.int64, s7: tl.int64,
    # input strides (broadcasted to output shape; padded to MAX_DIMS)
    xst0: tl.int64, xst1: tl.int64, xst2: tl.int64, xst3: tl.int64, xst4: tl.int64, xst5: tl.int64, xst6: tl.int64, xst7: tl.int64,
    # output strides (padded to MAX_DIMS)
    ost0: tl.int64, ost1: tl.int64, ost2: tl.int64, ost3: tl.int64, ost4: tl.int64, ost5: tl.int64, ost6: tl.int64, ost7: tl.int64,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements

    # Map linear index -> multi-dimensional index (row-major order)
    r = off.to(tl.int64)
    c0 = r % s0
    r = r // s0
    c1 = r % s1
    r = r // s1
    c2 = r % s2
    r = r // s2
    c3 = r % s3
    r = r // s3
    c4 = r % s4
    r = r // s4
    c5 = r % s5
    r = r // s5
    c6 = r % s6
    r = r // s6
    c7 = r % s7

    # Compute input and output offsets using (possibly broadcasted) strides
    x_offset = (
        c0 * xst0 +
        c1 * xst1 +
        c2 * xst2 +
        c3 * xst3 +
        c4 * xst4 +
        c5 * xst5 +
        c6 * xst6 +
        c7 * xst7
    )
    out_offset = (
        c0 * ost0 +
        c1 * ost1 +
        c2 * ost2 +
        c3 * ost3 +
        c4 * ost4 +
        c5 * ost5 +
        c6 * ost6 +
        c7 * ost7
    )

    x = tl.load(x_ptr + x_offset, mask=mask, other=0)
    y = ~x
    tl.store(out_ptr + out_offset, y, mask=mask)


# Helpers
MAX_DIMS = 8

def _pad_to_max_dims(lst, fill, max_dims=MAX_DIMS):
    lst = list(lst)
    if len(lst) > max_dims:
        raise ValueError(f"Rank {len(lst)} exceeds MAX_DIMS={max_dims}")
    return lst + [fill] * (max_dims - len(lst))

def _broadcast_strides(in_shape, in_strides, out_shape):
    # Returns input strides broadcasted to out_shape; stride 0 where dimension is broadcasted
    in_shape = list(in_shape)
    in_strides = list(in_strides)
    out_shape = list(out_shape)

    in_ndim = len(in_shape)
    out_ndim = len(out_shape)
    diff = out_ndim - in_ndim

    b_strides = [0] * out_ndim
    for i in range(out_ndim - 1, -1, -1):
        if i - diff >= 0:
            s = in_shape[i - diff]
            st = in_strides[i - diff]
            o = out_shape[i]
            if s == o:
                b_strides[i] = st
            elif s == 1:
                b_strides[i] = 0
            else:
                raise ValueError(f"Shapes not broadcastable: input {tuple(in_shape)} to {tuple(out_shape)}")
        else:
            # leading broadcasted dims
            b_strides[i] = 0
    return b_strides

def _shape_from_tensor(t):
    return list(t.shape)

def _strides_in_elements(t):
    # torch.stride() already returns element strides (not bytes)
    return list(t.stride())


def _launch_bitwise_not_kernel(x: torch.Tensor, out: torch.Tensor):
    # Validate device/dtype
    if x.device != out.device:
        raise ValueError("Input and output tensors must be on the same device")
    if x.dtype != out.dtype:
        raise ValueError("Input and output tensors must have the same dtype")

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Adaptive BLOCK_SIZE based on data size
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 1024 * 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Fast path: both tensors contiguous and same shape (no broadcasting)
    if (x.is_contiguous() and out.is_contiguous() and
        x.shape == out.shape):
        bitwise_not_kernel_contiguous[grid](
            x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return out

    # General path: handle broadcasting and strided tensors
    out_shape = _shape_from_tensor(out)
    x_broadcast_strides = _broadcast_strides(
        _shape_from_tensor(x), _strides_in_elements(x), out_shape
    )
    out_strides = _strides_in_elements(out)

    shape_padded = _pad_to_max_dims(out_shape, 1)
    x_strides_padded = _pad_to_max_dims(x_broadcast_strides, 0)
    out_strides_padded = _pad_to_max_dims(out_strides, 0)

    bitwise_not_kernel[grid](
        x, out, n_elements,
        shape_padded[0], shape_padded[1], shape_padded[2], shape_padded[3],
        shape_padded[4], shape_padded[5], shape_padded[6], shape_padded[7],
        x_strides_padded[0], x_strides_padded[1], x_strides_padded[2], x_strides_padded[3],
        x_strides_padded[4], x_strides_padded[5], x_strides_padded[6], x_strides_padded[7],
        out_strides_padded[0], out_strides_padded[1], out_strides_padded[2], out_strides_padded[3],
        out_strides_padded[4], out_strides_padded[5], out_strides_padded[6], out_strides_padded[7],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# Wrapper for aten::bitwise_not(Tensor self) -> Tensor
def bitwise_not(self: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(self)
    return _launch_bitwise_not_kernel(self, out)


# Wrapper for aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def bitwise_not_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    # Ensure out shape is broadcasted shape of self to out
    # For unary, just verify self is broadcastable to out
    _ = _broadcast_strides(_shape_from_tensor(self), _strides_in_elements(self), _shape_from_tensor(out))
    return _launch_bitwise_not_kernel(self, out)
