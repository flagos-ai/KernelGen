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

MAX_DIMS = 8

def _pad_to_max_dims(arr, fill=0):
    result = [fill] * MAX_DIMS
    for i, v in enumerate(arr):
        if i < MAX_DIMS:
            result[i] = v
    return result

def _cumprod_reversed_dims(dims_rev):
    L = len(dims_rev)
    cp = [1] * L
    for i in range(1, L):
        cp[i] = cp[i - 1] * dims_rev[i - 1]
    return cp

def _select_block_size(n_elements: int):
    if n_elements < 1024:
        return 256
    elif n_elements < 1024 * 1024:
        return 1024
    else:
        return 2048

@triton.jit
def fill_contiguous_kernel(
    value_ptr,  # pointer to 1-element tensor with correct dtype
    out_ptr,    # pointer to output (self)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    offs64 = offs.to(tl.int64)
    val = tl.load(value_ptr)  # scalar load, broadcasted
    tl.store(out_ptr + offs64, val, mask=mask)

@triton.jit
def fill_strided_kernel(
    value_ptr,  # pointer to 1-element tensor with correct dtype
    out_ptr,
    n_elements,
    # Reversed (row-major) padded shape
    d0, d1, d2, d3, d4, d5, d6, d7,
    # Reversed cumulative products for indexing
    s0, s1, s2, s3, s4, s5, s6, s7,
    # Output strides (reversed)
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    r = offs.to(tl.int64)

    i0 = (r // s0) % d0
    i1 = (r // s1) % d1
    i2 = (r // s2) % d2
    i3 = (r // s3) % d3
    i4 = (r // s4) % d4
    i5 = (r // s5) % d5
    i6 = (r // s6) % d6
    i7 = (r // s7) % d7

    o_off = (
        i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 +
        i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7
    ).to(tl.int64)

    val = tl.load(value_ptr)
    tl.store(out_ptr + o_off, val, mask=mask)

def _prepare_value_scalar_buffer(self: torch.Tensor, value):
    # Create a 0-d tensor on the same device/dtype as self with the scalar value
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError("fill_ only supports a single-element tensor as value")
        if value.device != self.device or value.dtype != self.dtype:
            value = value.to(device=self.device, dtype=self.dtype)
        value = value.contiguous()
        if value.dim() == 0:
            return value
        else:
            return value.view(())
    else:
        # Avoid using fill_ to prevent recursion; create scalar tensor directly
        return torch.tensor(value, device=self.device, dtype=self.dtype)

def _launch_fill(self: torch.Tensor, value_buf: torch.Tensor):
    # value_buf: 0-d, device=self.device, dtype=self.dtype
    n_elements = self.numel()
    if n_elements == 0:
        return self

    if self.is_contiguous():
        BLOCK_SIZE = _select_block_size(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fill_contiguous_kernel[grid](
            value_buf, self, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        out_shape_rev = list(self.shape)[::-1]
        out_strides_rev = list(self.stride())[::-1]
        cum_prod = _cumprod_reversed_dims(out_shape_rev)

        d = _pad_to_max_dims(out_shape_rev, fill=1)
        # pad cumprod with n_elements to keep division safe
        s = _pad_to_max_dims(cum_prod, fill=n_elements if n_elements > 0 else 1)
        ost = _pad_to_max_dims(out_strides_rev, fill=0)

        BLOCK_SIZE = _select_block_size(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fill_strided_kernel[grid](
            value_buf, self, n_elements,
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7],
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return self

def fill__Scalar(self: torch.Tensor, value):
    if self.numel() == 0:
        return self
    value_buf = _prepare_value_scalar_buffer(self, value)
    return _launch_fill(self, value_buf)

def fill__Tensor(self: torch.Tensor, value: torch.Tensor):
    if self.numel() == 0:
        return self
    if not isinstance(value, torch.Tensor):
        raise TypeError("fill__Tensor expects a Tensor as 'value'")
    if value.numel() != 1:
        raise RuntimeError("fill_ only supports a single-element tensor as value")
    value_buf = _prepare_value_scalar_buffer(self, value)
    return _launch_fill(self, value_buf)
