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
    """Compute input strides aligned to out_shape for broadcasting semantics."""
    out_ndim = len(out_shape)
    in_ndim = len(in_shape)
    b_strides = [0] * out_ndim
    for i in range(out_ndim):
        in_idx = i - (out_ndim - in_ndim)
        if in_idx < 0:
            # Extra leading dims: broadcast
            b_strides[i] = 0
        else:
            s = in_shape[in_idx]
            o = out_shape[i]
            st = in_strides[in_idx]
            if s == o:
                b_strides[i] = st
            elif s == 1:
                b_strides[i] = 0
            else:
                # Incompatible shapes; unsqueeze should never hit this path
                b_strides[i] = 0
    return b_strides


def _find_contiguous_tail(shape, strides):
    """
    Find largest contiguous tail (in physical memory) of input tensor.
    Returns (tail_start_index, N), where N = product of tail shapes.
    Contiguity criterion: strides[i] == expected, with expected starting at 1 and
    multiplying by shape as we move left.
    Size-1 dims do not enlarge contiguous region unless their stride matches expected
    (but they don't change N anyway).
    """
    expected = 1
    N = 1
    tail_start = len(shape)  # default: empty tail
    for i in range(len(shape) - 1, -1, -1):
        # Skip size-1 dims that don't break contiguity but don't enlarge N
        if shape[i] == 1:
            # If stride matches expected, we can treat as part of contiguous chain,
            # but N doesn't grow (shape==1). Either way, safe to continue.
            if strides[i] == expected:
                tail_start = i
            # Even if it doesn't match, size-1 dims don't contribute; we break if it mismatches.
            else:
                break
            continue
        if strides[i] == expected:
            tail_start = i
            expected *= shape[i]
            N *= shape[i]
        else:
            break
    return tail_start, N


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def unsqueeze_kernel_contiguous(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    r = offs.to(tl.int64)
    x = tl.load(x_ptr + r, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + r, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def unsqueeze_kernel_2d_tail_contig(
    x_ptr, out_ptr,
    M, N,
    # Outer (non-contiguous) dims only
    md0, md1, md2, md3, md4, md5, md6, md7,
    ms0, ms1, ms2, ms3, ms4, ms5, ms6, ms7,  # cumulative products for outer dims
    mxst0, mxst1, mxst2, mxst3, mxst4, mxst5, mxst6, mxst7,  # x strides for outer dims
    OUTER_DIMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D launch: rows (outer dims collapsed), cols (contiguous tail)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (pid_m < M) & (offs_n < N)

    # Decode pid_m into outer multi-index once per block
    rm = pid_m.to(tl.int64)

    # Initialize row base offset
    row_base = tl.zeros((), dtype=tl.int64)

    # Compute outer indices and base offset using cumulative products
    # Note: we always pass 8 params; OUTER_DIMS tells how many are active.
    if OUTER_DIMS > 0:
        im0 = (rm // ms0) % md0
        row_base += im0 * mxst0
    if OUTER_DIMS > 1:
        im1 = (rm // ms1) % md1
        row_base += im1 * mxst1
    if OUTER_DIMS > 2:
        im2 = (rm // ms2) % md2
        row_base += im2 * mxst2
    if OUTER_DIMS > 3:
        im3 = (rm // ms3) % md3
        row_base += im3 * mxst3
    if OUTER_DIMS > 4:
        im4 = (rm // ms4) % md4
        row_base += im4 * mxst4
    if OUTER_DIMS > 5:
        im5 = (rm // ms5) % md5
        row_base += im5 * mxst5
    if OUTER_DIMS > 6:
        im6 = (rm // ms6) % md6
        row_base += im6 * mxst6
    if OUTER_DIMS > 7:
        im7 = (rm // ms7) % md7
        row_base += im7 * mxst7

    # x tail is contiguous: x_off = row_base + offs_n
    x_offs = row_base + offs_n.to(tl.int64)
    # out is fully contiguous: out_off = pid_m * N + offs_n
    out_row_base = pid_m.to(tl.int64) * N
    out_offs = out_row_base + offs_n.to(tl.int64)

    x = tl.load(x_ptr + x_offs, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + out_offs, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def unsqueeze_kernel_strided(
    x_ptr, out_ptr, n_elements,
    # Reversed (row-major) output dims
    d0, d1, d2, d3, d4, d5, d6, d7,
    # Cumulative products for linear index -> multi-index (reversed)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # Input strides aligned to out (reversed)
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    # Output strides (reversed)
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    r = offs.to(tl.int64)
    mask = r < n_elements

    # Multi-dimensional indices
    i0 = (r // s0) % d0
    i1 = (r // s1) % d1
    i2 = (r // s2) % d2
    i3 = (r // s3) % d3
    i4 = (r // s4) % d4
    i5 = (r // s5) % d5
    i6 = (r // s6) % d6
    i7 = (r // s7) % d7

    x_off = (
        i0 * xst0 + i1 * xst1 + i2 * xst2 + i3 * xst3 +
        i4 * xst4 + i5 * xst5 + i6 * xst6 + i7 * xst7
    )
    o_off = (
        i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 +
        i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7
    )

    x = tl.load(x_ptr + x_off, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + o_off, x, mask=mask)


def unsqueeze(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Normalize dim to be within [0, x.ndim]
    ndim = x.ndim
    if dim < 0:
        dim = dim + ndim + 1
    if not (0 <= dim <= ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim-1}, {ndim}], but got {dim - (ndim + 1)})"
        )

    # Compute output shape
    out_shape = list(x.shape)
    out_shape.insert(dim, 1)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Fast path: both input and output contiguous and same numel
    if x.is_contiguous() and out.is_contiguous() and x.numel() == n_elements:
        # Launch contiguous kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        unsqueeze_kernel_contiguous[grid](
            x, out, n_elements,
        )
        return out

    # Attempt 2D optimization when input has a contiguous tail region
    x_shape_list = list(x.shape)
    x_strides_list = list(x.stride())
    tail_start, N = _find_contiguous_tail(x_shape_list, x_strides_list)
    M = x.numel() // max(N, 1)

    use_2d = (N > 1) and (M > 1) and out.is_contiguous()
    if use_2d:
        # Prepare outer dims (0..tail_start-1)
        outer_shapes = x_shape_list[:tail_start]
        outer_strides = x_strides_list[:tail_start]
        # Compute cumulative products for outer dims mapping from pid_m -> indices
        # Reverse for row-major decoding
        outer_shapes_rev = outer_shapes[::-1]
        ms = [1] * len(outer_shapes_rev)
        for i in range(1, len(outer_shapes_rev)):
            ms[i] = ms[i - 1] * outer_shapes_rev[i - 1]

        # Pad to MAX_DIMS
        md = _pad_to_max_dims(outer_shapes_rev, fill=1)
        msc = _pad_to_max_dims(ms, fill=M)  # fill doesn't matter
        mxst = _pad_to_max_dims(list(outer_strides[::-1]), fill=0)

        OUTER_DIMS = len(outer_shapes_rev)

        grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_N']))
        unsqueeze_kernel_2d_tail_contig[grid](
            x, out,
            M, N,
            md[0], md[1], md[2], md[3], md[4], md[5], md[6], md[7],
            msc[0], msc[1], msc[2], msc[3], msc[4], msc[5], msc[6], msc[7],
            mxst[0], mxst[1], mxst[2], mxst[3], mxst[4], mxst[5], mxst[6], mxst[7],
            OUTER_DIMS=OUTER_DIMS,
        )
        return out

    # General path: handle non-contiguous input/output via strides and broadcasting
    out_shape_list = list(out.shape)
    x_shape_list = list(x.shape)
    x_strides_list = list(x.stride())

    # Compute input strides aligned to output shape (broadcast dims get stride=0)
    x_b_strides = _broadcast_strides(x_shape_list, x_strides_list, out_shape_list)

    # Reverse for row-major indexing
    out_shape_rev = out_shape_list[::-1]
    out_strides_rev = list(out.stride())[::-1]
    x_b_strides_rev = x_b_strides[::-1]

    # Compute cumulative products for reversed dims
    cum_prod = [1] * len(out_shape_rev)
    for i in range(1, len(out_shape_rev)):
        cum_prod[i] = cum_prod[i - 1] * out_shape_rev[i - 1]

    # Pad arrays to MAX_DIMS
    d = _pad_to_max_dims(out_shape_rev, fill=1)
    s = _pad_to_max_dims(cum_prod, fill=n_elements)
    xst = _pad_to_max_dims(x_b_strides_rev, fill=0)
    ost = _pad_to_max_dims(out_strides_rev, fill=0)

    # Launch autotuned strided kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    unsqueeze_kernel_strided[grid](
        x, out, n_elements,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        xst[0], xst[1], xst[2], xst[3], xst[4], xst[5], xst[6], xst[7],
        ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7],
    )
    return out
