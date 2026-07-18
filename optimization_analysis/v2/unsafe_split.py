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

def _cumprod_rev(shape_rev):
    s = [1] * len(shape_rev)
    for i in range(1, len(shape_rev)):
        s[i] = s[i - 1] * shape_rev[i - 1]
    return s

def _choose_block_size(n_elements: int):
    if n_elements < 1024:
        return 256
    elif n_elements < 1024 * 1024:
        return 1024
    else:
        return 2048

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def copy_kernel_contiguous(
    x_ptr, out_ptr,
    base,  # base offset in input (int64 elements)
    n_elements,  # number of elements to copy
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    r = offs.to(tl.int64)
    x = tl.load(x_ptr + base + r, mask=mask)
    tl.store(out_ptr + r, x, mask=mask)

@triton.jit
def copy_kernel_strided(
    x_ptr, out_ptr,
    base,  # base offset in input (int64 elements)
    n_elements,
    # reversed output shape (row-major indexing)
    d0, d1, d2, d3, d4, d5, d6, d7,
    # cumulative products for indexing (reversed)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # input strides reversed
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    # output strides reversed
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

    x_off = base + (i0 * xst0 + i1 * xst1 + i2 * xst2 + i3 * xst3 + i4 * xst4 + i5 * xst5 + i6 * xst6 + i7 * xst7)
    o_off = (i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 + i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7)

    val = tl.load(x_ptr + x_off, mask=mask)
    tl.store(out_ptr + o_off, val, mask=mask)

def _copy_chunk_triton(src: torch.Tensor, dst: torch.Tensor, dim: int, start_idx: int):
    # Early return for empty tensors
    n_elements = dst.numel()
    if n_elements == 0:
        return

    # Compute base offset in input (elements)
    base = start_idx * src.stride()[dim]

    # Fast path: contiguous tensors and slicing along dim=0 yields contiguous block
    if src.is_contiguous() and dst.is_contiguous() and dim == 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        copy_kernel_contiguous[grid](
            src, dst,
            base, n_elements,
        )
        return

    # General path: strided copy
    out_shape = list(dst.shape)[::-1]
    out_strides = list(dst.stride())[::-1]
    in_strides = list(src.stride())[::-1]

    d = _pad_to_max_dims(out_shape, fill=1)
    s = _pad_to_max_dims(_cumprod_rev(out_shape), fill=n_elements)
    xst = _pad_to_max_dims(in_strides, fill=0)
    ost = _pad_to_max_dims(out_strides, fill=0)

    BLOCK_SIZE = _choose_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    copy_kernel_strided[grid](
        src, dst,
        base, n_elements,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        xst[0], xst[1], xst[2], xst[3], xst[4], xst[5], xst[6], xst[7],
        ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7],
        BLOCK_SIZE=BLOCK_SIZE,
    )

def _normalize_dim(dim: int, ndim: int):
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError("dim out of range")
    return dim

def unsafe_split_Tensor(self: torch.Tensor, split_size, dim: int = 0):
    if not self.is_cuda:
        raise RuntimeError("unsafe_split_Tensor requires CUDA tensors")
    if isinstance(split_size, torch.SymInt):
        split_size = int(split_size)
    if not isinstance(split_size, int) or split_size <= 0:
        raise ValueError("split_size must be a positive integer")

    dim = _normalize_dim(dim, self.ndim)
    total = self.size(dim)
    if total == 0:
        return []

    num_chunks = (total + split_size - 1) // split_size
    outs = []
    for i in range(num_chunks):
        start = i * split_size
        size_i = min(split_size, total - start)
        out_shape = list(self.shape)
        out_shape[dim] = size_i
        out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        _copy_chunk_triton(self, out, dim, start)
        outs.append(out)
    return outs

def unsafe_split_Tensor_out(self: torch.Tensor, split_size, dim: int = 0, *, out):
    if not self.is_cuda:
        raise RuntimeError("unsafe_split_Tensor_out requires CUDA tensors")
    if isinstance(split_size, torch.SymInt):
        split_size = int(split_size)
    if not isinstance(split_size, int) or split_size <= 0:
        raise ValueError("split_size must be a positive integer")

    if not isinstance(out, (list, tuple)):
        raise TypeError("out must be a list or tuple of tensors")

    dim = _normalize_dim(dim, self.ndim)
    total = self.size(dim)
    num_chunks = (total + split_size - 1) // split_size
    if len(out) != num_chunks:
        raise ValueError(f"out list length must be {num_chunks}, got {len(out)}")

    for i in range(num_chunks):
        start = i * split_size
        size_i = min(split_size, total - start)
        expected_shape = list(self.shape)
        expected_shape[dim] = size_i
        dst = out[i]
        if not isinstance(dst, torch.Tensor):
            raise TypeError("out elements must be tensors")
        if dst.device != self.device or dst.dtype != self.dtype:
            raise ValueError("out tensors must have same device and dtype as input")
        if list(dst.shape) != expected_shape:
            raise ValueError(f"out[{i}] has incorrect shape, expected {expected_shape}, got {list(dst.shape)}")
        _copy_chunk_triton(self, dst, dim, start)
    return None
