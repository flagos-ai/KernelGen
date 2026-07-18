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

def _broadcast_strides(in_shape, in_strides, out_shape):
    ndim_out = len(out_shape)
    ndim_in = len(in_shape)
    b_strides = [0] * ndim_out
    for i in range(ndim_out - 1, -1, -1):
        in_i = i - (ndim_out - ndim_in)
        if in_i >= 0:
            if in_shape[in_i] == out_shape[i]:
                b_strides[i] = in_strides[in_i]
            elif in_shape[in_i] == 1:
                b_strides[i] = 0
            else:
                b_strides[i] = 0
        else:
            b_strides[i] = 0
    return b_strides

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def resolve_conj_kernel_contiguous(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def resolve_conj_kernel_strided(
    x_ptr, out_ptr, n_elements,
    # reversed (row-major) output shape padded
    d0, d1, d2, d3, d4, d5, d6, d7,
    # reversed cumulative products for output indexing
    s0, s1, s2, s3, s4, s5, s6, s7,
    # reversed input strides (broadcast handled as stride=0)
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    # reversed output strides
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

    x_off = i0 * xst0 + i1 * xst1 + i2 * xst2 + i3 * xst3 + i4 * xst4 + i5 * xst5 + i6 * xst6 + i7 * xst7
    o_off = i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 + i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7

    val = tl.load(x_ptr + x_off, mask=mask)
    tl.store(out_ptr + o_off, val, mask=mask)

def resolve_conj(x: torch.Tensor) -> torch.Tensor:
    # Empty tensor early return
    if x.numel() == 0:
        return x.clone()

    # Triton does not support complex dtypes; handle all complex tensors via PyTorch
    if x.is_complex():
        out = torch.empty_like(x)
        out.copy_(x.conj() if x.is_conj() else x)
        return out

    # For non-CUDA tensors, use PyTorch copy as Triton requires CUDA
    if not x.is_cuda:
        out = torch.empty_like(x)
        out.copy_(x.conj() if x.is_conj() else x)
        return out

    # If conj bit is set (for non-complex types, this should be false, but keep for safety)
    if hasattr(x, "is_conj") and x.is_conj():
        out = torch.empty_like(x)
        out.copy_(x.conj())
        return out

    # Use Triton kernels for non-complex CUDA tensors: identity copy
    out = torch.empty_like(x)
    n_elements = out.numel()

    # Fast path: contiguous memory
    if x.is_contiguous() and out.is_contiguous() and x.shape == out.shape:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        resolve_conj_kernel_contiguous[grid](
            x, out, n_elements,
        )
        return out

    # General path: arbitrary strides
    out_shape = list(out.shape)[::-1]
    x_bstrides = _broadcast_strides(list(x.shape), list(x.stride()), list(out.shape))[::-1]
    out_strides = list(out.stride())[::-1]

    # cumulative products for row-major index mapping
    cum_prod = [1] * len(out_shape)
    for i in range(1, len(out_shape)):
        cum_prod[i] = cum_prod[i - 1] * out_shape[i - 1]

    d = _pad_to_max_dims(out_shape, fill=1)
    s = _pad_to_max_dims(cum_prod, fill=n_elements)
    xst = _pad_to_max_dims(x_bstrides, fill=0)
    ost = _pad_to_max_dims(out_strides, fill=0)

    # Adaptive block size selection
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 1024 * 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    resolve_conj_kernel_strided[grid](
        x, out, n_elements,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        xst[0], xst[1], xst[2], xst[3], xst[4], xst[5], xst[6], xst[7],
        ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Export with the exact required name for dispatcher compatibility
globals()['aten::resolve_conj'] = resolve_conj
