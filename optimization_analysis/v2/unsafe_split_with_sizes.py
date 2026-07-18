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
        result[i] = v
    return result

def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise IndexError(f"dim {dim} out of range for tensor of dimension {ndim}")
    return dim

def _adaptive_block_size(n_elements):
    if n_elements < 1024:
        return 256
    elif n_elements < 1024 * 1024:
        return 1024
    else:
        return 2048

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
def copy_contiguous_kernel(
    x_ptr, out_ptr,
    base_in_off,  # int64 offset (in elements) to the start of the slice in x
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    r = offs.to(tl.int64)
    base = tl.full((), base_in_off, tl.int64)
    x = tl.load(x_ptr + base + r, mask=mask)
    tl.store(out_ptr + r, x, mask=mask)

@triton.jit
def copy_nd_strided_kernel(
    x_ptr, out_ptr,
    base_in_off,  # int64 offset (in elements) to the start of the slice in x
    n_elements,
    # Padded shape (reversed order)
    d0, d1, d2, d3, d4, d5, d6, d7,
    # Cumulative products for index calculation (reversed)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # Input strides (reversed)
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    # Output strides (reversed)
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    r = offs.to(tl.int64)

    # Cast all scalar params to int64 to avoid overflow or dtype mismatch
    d0_i = tl.full((), d0, tl.int64); d1_i = tl.full((), d1, tl.int64)
    d2_i = tl.full((), d2, tl.int64); d3_i = tl.full((), d3, tl.int64)
    d4_i = tl.full((), d4, tl.int64); d5_i = tl.full((), d5, tl.int64)
    d6_i = tl.full((), d6, tl.int64); d7_i = tl.full((), d7, tl.int64)

    s0_i = tl.full((), s0, tl.int64); s1_i = tl.full((), s1, tl.int64)
    s2_i = tl.full((), s2, tl.int64); s3_i = tl.full((), s3, tl.int64)
    s4_i = tl.full((), s4, tl.int64); s5_i = tl.full((), s5, tl.int64)
    s6_i = tl.full((), s6, tl.int64); s7_i = tl.full((), s7, tl.int64)

    xst0_i = tl.full((), xst0, tl.int64); xst1_i = tl.full((), xst1, tl.int64)
    xst2_i = tl.full((), xst2, tl.int64); xst3_i = tl.full((), xst3, tl.int64)
    xst4_i = tl.full((), xst4, tl.int64); xst5_i = tl.full((), xst5, tl.int64)
    xst6_i = tl.full((), xst6, tl.int64); xst7_i = tl.full((), xst7, tl.int64)

    ost0_i = tl.full((), ost0, tl.int64); ost1_i = tl.full((), ost1, tl.int64)
    ost2_i = tl.full((), ost2, tl.int64); ost3_i = tl.full((), ost3, tl.int64)
    ost4_i = tl.full((), ost4, tl.int64); ost5_i = tl.full((), ost5, tl.int64)
    ost6_i = tl.full((), ost6, tl.int64); ost7_i = tl.full((), ost7, tl.int64)

    i0 = (r // s0_i) % d0_i
    i1 = (r // s1_i) % d1_i
    i2 = (r // s2_i) % d2_i
    i3 = (r // s3_i) % d3_i
    i4 = (r // s4_i) % d4_i
    i5 = (r // s5_i) % d5_i
    i6 = (r // s6_i) % d6_i
    i7 = (r // s7_i) % d7_i

    x_off = i0 * xst0_i + i1 * xst1_i + i2 * xst2_i + i3 * xst3_i + i4 * xst4_i + i5 * xst5_i + i6 * xst6_i + i7 * xst7_i
    o_off = i0 * ost0_i + i1 * ost1_i + i2 * ost2_i + i3 * ost3_i + i4 * ost4_i + i5 * ost5_i + i6 * ost6_i + i7 * ost7_i

    base = tl.full((), base_in_off, tl.int64)
    x = tl.load(x_ptr + base + x_off, mask=mask)
    tl.store(out_ptr + o_off, x, mask=mask)

def _copy_slice_into_out(x: torch.Tensor, out: torch.Tensor, dim: int, start: int, size: int):
    if out.numel() == 0:
        return

    assert x.device.type == 'cuda' and out.device.type == 'cuda', "Triton kernels require CUDA tensors"
    assert x.dtype == out.dtype, "Input and output dtypes must match"

    dim = _normalize_dim(dim, x.ndim)
    base_in_off = start * x.stride(dim)
    n_elements = out.numel()

    # Use fast path only when slice maps to a single contiguous region in input and out is contiguous.
    use_fast_path = (
        x.is_contiguous()
        and out.is_contiguous()
        and dim == x.ndim - 1
        and size == x.size(dim)
        and start == 0
    )

    if use_fast_path:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        copy_contiguous_kernel[grid](
            x, out,
            base_in_off,
            n_elements,
        )
    else:
        out_shape = list(out.shape)
        d_rev = out_shape[::-1]
        x_strides_rev = list(x.stride())[::-1]
        out_strides_rev = list(out.stride())[::-1]

        # Compute cumulative products for reversed dims
        s_rev = [1] * len(d_rev)
        for i in range(1, len(d_rev)):
            s_rev[i] = s_rev[i - 1] * d_rev[i - 1]

        # Pad arrays
        d = _pad_to_max_dims(d_rev, fill=1)
        s = _pad_to_max_dims(s_rev, fill=n_elements)
        xst = _pad_to_max_dims(x_strides_rev, fill=0)
        ost = _pad_to_max_dims(out_strides_rev, fill=0)

        BLOCK_SIZE = _adaptive_block_size(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        copy_nd_strided_kernel[grid](
            x, out,
            base_in_off,
            n_elements,
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            xst[0], xst[1], xst[2], xst[3], xst[4], xst[5], xst[6], xst[7],
            ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7],
            BLOCK_SIZE=BLOCK_SIZE,
        )

def unsafe_split_with_sizes(self: torch.Tensor, split_sizes, dim: int = 0):
    if not isinstance(split_sizes, (list, tuple)):
        raise TypeError("split_sizes must be a list or tuple of sizes")
    split_sizes = [int(s) for s in split_sizes]
    dim = _normalize_dim(dim, self.ndim)
    total = sum(split_sizes)
    if total != self.size(dim):
        raise ValueError(f"Sum of split_sizes ({total}) must equal size of dimension {dim} ({self.size(dim)})")

    if self.numel() == 0:
        outs = []
        for size in split_sizes:
            shape = list(self.shape)
            shape[dim] = size
            outs.append(torch.empty(shape, dtype=self.dtype, device=self.device))
        return outs

    outs = []
    start = 0
    for size in split_sizes:
        shape = list(self.shape)
        shape[dim] = size
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
        _copy_slice_into_out(self, out, dim, start, size)
        start += size
        outs.append(out)
    return outs

def unsafe_split_with_sizes_out(self: torch.Tensor, split_sizes, dim: int = 0, *, out):
    if not isinstance(split_sizes, (list, tuple)):
        raise TypeError("split_sizes must be a list or tuple of sizes")
    split_sizes = [int(s) for s in split_sizes]
    dim = _normalize_dim(dim, self.ndim)
    total = sum(split_sizes)
    if total != self.size(dim):
        raise ValueError(f"Sum of split_sizes ({total}) must equal size of dimension {dim} ({self.size(dim)})")
    if not isinstance(out, (list, tuple)):
        raise TypeError("out must be a list/tuple of output tensors")
    if len(out) != len(split_sizes):
        raise ValueError("out length must match number of split sizes")
    for i, size in enumerate(split_sizes):
        expected_shape = list(self.shape)
        expected_shape[dim] = size
        if list(out[i].shape) != expected_shape:
            raise ValueError(f"out[{i}] shape mismatch: expected {expected_shape}, got {list(out[i].shape)}")
        if out[i].device != self.device:
            raise ValueError(f"out[{i}] device mismatch: expected {self.device}, got {out[i].device}")
        if out[i].dtype != self.dtype:
            raise ValueError(f"out[{i}] dtype mismatch: expected {self.dtype}, got {out[i].dtype}")

    if self.numel() == 0:
        return

    start = 0
    for i, size in enumerate(split_sizes):
        _copy_slice_into_out(self, out[i], dim, start, size)
        start += size
    return
