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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def _transpose_2d_kernel(
    x_ptr, out_ptr,
    M, N,
    x_stride0, x_stride1,
    o_stride0, o_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm64 = rm.to(tl.int64)
    rn64 = rn.to(tl.int64)

    x_stride0 = tl.full((), x_stride0, tl.int64)
    x_stride1 = tl.full((), x_stride1, tl.int64)
    o_stride0 = tl.full((), o_stride0, tl.int64)
    o_stride1 = tl.full((), o_stride1, tl.int64)

    # Load a [BLOCK_M, BLOCK_N] tile from x
    x_offsets = rm64[:, None] * x_stride0 + rn64[None, :] * x_stride1
    mask_load = (rm[:, None] < M) & (rn[None, :] < N)
    x_tile = tl.load(x_ptr + x_offsets, mask=mask_load, other=0, eviction_policy='evict_last')

    # Store the transposed tile to out at [rn, rm] to ensure coalesced stores
    o_offsets_T = rn64[:, None] * o_stride0 + rm64[None, :] * o_stride1
    mask_store = (rn[:, None] < N) & (rm[None, :] < M)
    tl.store(out_ptr + o_offsets_T, tl.trans(x_tile), mask=mask_store, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def _copy_contiguous_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0, eviction_policy='evict_last')
    tl.store(out_ptr + offs, x, mask=mask, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def _copy_strided_kernel(
    x_ptr, out_ptr,
    n_elements,
    # dimensions (reversed)
    d0, d1, d2, d3, d4, d5, d6, d7,
    # cumulative products for linear to multi-dim index (reversed)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # input strides (reversed)
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    # output strides (reversed)
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

    x_off = (
        i0 * xst0 + i1 * xst1 + i2 * xst2 + i3 * xst3 +
        i4 * xst4 + i5 * xst5 + i6 * xst6 + i7 * xst7
    )
    o_off = (
        i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 +
        i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7
    )

    x = tl.load(x_ptr + x_off, mask=mask, other=0, eviction_policy='evict_last')
    tl.store(out_ptr + o_off, x, mask=mask, eviction_policy='evict_last')


def t(x: torch.Tensor) -> torch.Tensor:
    # Handle empty tensors early
    if x.numel() == 0:
        if x.ndim == 2:
            return torch.empty((x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device)
        else:
            return torch.empty_like(x)

    if x.ndim == 2:
        M, N = x.shape
        out = torch.empty((N, M), dtype=x.dtype, device=x.device)

        x_stride0, x_stride1 = x.stride()
        o_stride0, o_stride1 = out.stride()

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        _transpose_2d_kernel[grid](
            x, out,
            M, N,
            x_stride0, x_stride1,
            o_stride0, o_stride1,
        )
        return out
    else:
        # For non-2D tensors, aten::t is identity; we return a copy with identical layout
        out = torch.empty_like(x)
        n_elements = out.numel()

        # Fast path: contiguous linear copy
        if x.is_contiguous() and out.is_contiguous():
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _copy_contiguous_kernel[grid](
                x, out,
                n_elements,
            )
            return out

        # General strided copy for non-contiguous tensors
        out_shape = list(out.shape)[::-1]
        x_strides = list(x.stride())[::-1]
        out_strides = list(out.stride())[::-1]

        # cumulative products for reversed shape
        cum_prod = [1] * MAX_DIMS
        for i in range(1, min(MAX_DIMS, len(out_shape))):
            cum_prod[i] = cum_prod[i - 1] * out_shape[i - 1]

        d = _pad_to_max_dims(out_shape, fill=1)
        s = _pad_to_max_dims(cum_prod, fill=n_elements)
        xst = _pad_to_max_dims(x_strides, fill=0)
        ost = _pad_to_max_dims(out_strides, fill=0)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _copy_strided_kernel[grid](
            x, out,
            n_elements,
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            xst[0], xst[1], xst[2], xst[3], xst[4], xst[5], xst[6], xst[7],
            ost[0], ost[1], ost[2], ost[3], ost[4], ost[5], ost[6], ost[7],
        )
        return out

# Expose wrapper with exact required name
globals()['aten::t'] = t
