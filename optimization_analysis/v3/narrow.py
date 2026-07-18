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
    cp = [1] * len(shape_rev)
    for i in range(1, len(shape_rev)):
        cp[i] = cp[i - 1] * shape_rev[i - 1]
    return cp


def _prod(arr):
    p = 1
    for x in arr:
        p *= int(x)
    return p


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def narrow_kernel_linear_contiguous(
    in_ptr, out_ptr, n_elements,
    slice_extra,  # scalar int64
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fastest path for contiguous tensors when the slice is a single contiguous block (rows == 1).
    Copies n_elements sequentially from in_ptr + slice_extra to out_ptr.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # Stream loads/stores
    x = tl.load(in_ptr + slice_extra + offs, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + offs, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_COLS': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_COLS': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_COLS': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_COLS': 1024}, num_warps=8, num_stages=3),
    ],
    key=['rows', 'seg_len'],
)
@triton.jit
def narrow_kernel_2d_contiguous(
    in_ptr, out_ptr,
    rows, seg_len,  # rows = prod(shape[:dim]), seg_len = length * inner
    in_row_stride,  # in stride between rows = original_dim_len * inner
    out_row_stride,  # out stride between rows = length * inner (== seg_len)
    slice_inner_offset,  # start * inner
    BLOCK_COLS: tl.constexpr,
):
    """
    Fast path for contiguous tensors: treat the operation as copying 'rows' blocks,
    each of length 'seg_len', from input to output. Each block is contiguous.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    col_offs = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_mask = pid_row < rows
    col_mask = col_offs < seg_len
    mask = row_mask & col_mask

    # Compute base offsets for this row
    in_row_base = pid_row * in_row_stride + slice_inner_offset
    out_row_base = pid_row * out_row_stride

    x = tl.load(in_ptr + in_row_base + col_offs, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + out_row_base + col_offs, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def narrow_kernel_strided(
    in_ptr, out_ptr, n_elements,
    # Padded output shape (reversed)
    d0, d1, d2, d3, d4, d5, d6, d7,
    # Padded cumulative products for index calculation (reversed)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # Input strides (reversed)
    inst0, inst1, inst2, inst3, inst4, inst5, inst6, inst7,
    # Output strides (reversed)
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    # Constant extra input offset = start * in_stride[dim]
    slice_extra,
    BLOCK_SIZE: tl.constexpr,
):
    """
    General strided kernel. Uses row-major reversed indexing with int64 arithmetic.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    r = offs.to(tl.int64)

    # Compute multi-dimensional indices in row-major (reversed) order
    i0 = (r // s0) % d0
    i1 = (r // s1) % d1
    i2 = (r // s2) % d2
    i3 = (r // s3) % d3
    i4 = (r // s4) % d4
    i5 = (r // s5) % d5
    i6 = (r // s6) % d6
    i7 = (r // s7) % d7

    x_off = (
        i0 * inst0 + i1 * inst1 + i2 * inst2 + i3 * inst3 +
        i4 * inst4 + i5 * inst5 + i6 * inst6 + i7 * inst7
    ) + slice_extra

    o_off = (
        i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 +
        i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7
    )

    x = tl.load(in_ptr + x_off, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + o_off, x, mask=mask)


def _narrow_impl(self: torch.Tensor, dim: int, start: int, length: int) -> torch.Tensor:
    assert self.is_cuda, "Input tensor must be on CUDA device"
    ndim = self.dim()
    if ndim == 0:
        assert dim in (0, -1), "Invalid dim for 0-D tensor"
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim, "dim out of range"
    size_dim = self.size(dim)
    assert length >= 0, "length must be non-negative"
    assert 0 <= start <= size_dim, "start out of range"
    assert start + length <= size_dim, "start + length out of range"

    out_shape = list(self.shape)
    out_shape[dim] = length
    out = torch.empty(out_shape, device=self.device, dtype=self.dtype)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Prepare reversed shapes and strides for general path
    out_shape_rev = out_shape[::-1]
    in_strides = list(self.stride())
    out_strides = list(out.stride())

    in_strides_rev = in_strides[::-1]
    out_strides_rev = out_strides[::-1]

    cum_prod_rev = _cumprod_rev(out_shape_rev)

    # Pad arrays
    d = _pad_to_max_dims(out_shape_rev, fill=1)
    s = _pad_to_max_dims(cum_prod_rev, fill=n_elements)
    inst = _pad_to_max_dims(in_strides_rev, fill=0)
    ost = _pad_to_max_dims(out_strides_rev, fill=0)

    # Constant extra offset into input
    slice_extra = int(start) * int(in_strides[dim])

    # Fast hierarchical strategy
    if self.is_contiguous() and out.is_contiguous():
        # Collapse to 2D copy: rows x seg_len
        # rows: product of dims before 'dim'
        # inner: product of dims after 'dim'
        rows = _prod(self.shape[:dim]) if dim > 0 else 1
        inner = _prod(self.shape[dim + 1:]) if dim + 1 < ndim else 1
        orig_len_dim = int(self.shape[dim])
        seg_len = int(length) * int(inner)

        # If rows == 1, the slice is a single contiguous block -> 1D linear kernel
        if rows == 1:
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            narrow_kernel_linear_contiguous[grid](
                self, out, n_elements,
                slice_extra,
            )
        else:
            # 2D contiguous copy across rows
            in_row_stride = orig_len_dim * inner
            out_row_stride = seg_len  # contiguous out
            # Grid over (rows, seg_len)
            def grid(meta):
                BLOCK_COLS = meta['BLOCK_COLS']
                return (rows, triton.cdiv(seg_len, BLOCK_COLS))

            narrow_kernel_2d_contiguous[grid](
                self, out,
                rows, seg_len,
                in_row_stride,
                out_row_stride,
                start * inner,
            )
    else:
        # General strided path with autotune
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        narrow_kernel_strided[grid](
            self, out, n_elements,
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            inst[0], inst[1], inst[2], inst[3], inst[4], inst[5], inst[6], inst[7],
            ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7],
            slice_extra,
        )

    return out


def narrow(self: torch.Tensor, dim: int, start: int, length: int) -> torch.Tensor:
    return _narrow_impl(self, dim, start, length)


def narrow_Tensor(self: torch.Tensor, dim: int, start: torch.Tensor, length: int) -> torch.Tensor:
    assert start.ndim == 0 or (start.ndim == 1 and start.numel() == 1), "start tensor must be 0-dim or 1-element"
    start_val = int(start.item())
    return _narrow_impl(self, dim, start_val, length)
