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

def _contiguous_strides(size):
    if len(size) == 0:
        return []
    strides = [0] * len(size)
    if len(size) > 0:
        strides[-1] = 1
        for i in range(len(size) - 2, -1, -1):
            strides[i] = strides[i + 1] * size[i + 1]
    return strides

def _prod(size):
    p = 1
    for s in size:
        p *= int(s)
    return p

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def as_strided_kernel_contiguous(
    x_ptr, out_ptr,
    base_offset,  # int64
    n_elements,   # int32/int64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    r = offs.to(tl.int64)
    src_off = r + tl.full([1], base_offset, tl.int64)
    x = tl.load(x_ptr + src_off, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + r, x, mask=mask, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def as_strided_kernel_1d_step(
    x_ptr, out_ptr,
    base_offset,  # int64
    step,         # int64 (vstr0)
    n_elements,   # int32/int64
    BLOCK_SIZE: tl.constexpr,
):
    # 1D gather/scatter along inner dimension only, outer_size == 1
    pid = tl.program_id(0)
    k = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = k < n_elements
    k64 = k.to(tl.int64)
    base_i64 = tl.full([1], base_offset, tl.int64)
    step_i64 = tl.full([1], step, tl.int64)
    src_off = base_i64 + k64 * step_i64
    x = tl.load(x_ptr + src_off, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + k64, x, mask=mask, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 1024}, num_warps=8, num_stages=4),
    ],
    key=['d0', 'outer_size'],
)
@triton.jit
def as_strided_kernel_outer1_tiled(
    x_ptr, out_ptr,
    base_offset,  # int64
    d0,           # inner (last) dimension size (reversed indexing)
    outer_size,   # product of remaining dimensions (= d1)
    # Active single outer dim size and stride
    d1,           # int64
    vstr0, vstr1, # int64
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_row_block = tl.program_id(0)
    pid_col = tl.program_id(1)

    k = pid_col * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < d0
    k64 = k.to(tl.int64)

    base_i64 = tl.full([1], base_offset, tl.int64)
    d0_i64 = tl.full([1], d0, tl.int64)
    v0_i64 = tl.full([1], vstr0, tl.int64)
    v1_i64 = tl.full([1], vstr1, tl.int64)

    for m in range(BLOCK_M):
        row = pid_row_block * BLOCK_M + m
        row_mask = row < outer_size
        mask = row_mask & k_mask

        i1 = row.to(tl.int64)  # only one active outer dim
        outer_src = i1 * v1_i64

        src_off = base_i64 + outer_src + k64 * v0_i64
        dst_off = row.to(tl.int64) * d0_i64 + k64

        x = tl.load(x_ptr + src_off, mask=mask, eviction_policy='evict_last')
        tl.store(out_ptr + dst_off, x, mask=mask, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 1024}, num_warps=8, num_stages=4),
    ],
    key=['d0', 'outer_size'],
)
@triton.jit
def as_strided_kernel_outer2_tiled(
    x_ptr, out_ptr,
    base_offset,  # int64
    d0,           # inner (last) dimension size (reversed indexing)
    outer_size,   # product of remaining dimensions (= d1*d2)
    # Active outer dim sizes and strides
    d1, d2,       # int64
    vstr0, vstr1, vstr2, # int64
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_row_block = tl.program_id(0)
    pid_col = tl.program_id(1)

    k = pid_col * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < d0
    k64 = k.to(tl.int64)

    base_i64 = tl.full([1], base_offset, tl.int64)
    d0_i64 = tl.full([1], d0, tl.int64)
    d1_i64 = tl.full([1], d1, tl.int64)
    v0_i64 = tl.full([1], vstr0, tl.int64)
    v1_i64 = tl.full([1], vstr1, tl.int64)
    v2_i64 = tl.full([1], vstr2, tl.int64)

    for m in range(BLOCK_M):
        row = pid_row_block * BLOCK_M + m
        row_mask = row < outer_size
        mask = row_mask & k_mask

        tmp = row.to(tl.int64)
        i1 = tmp % d1_i64
        i2 = tmp // d1_i64

        outer_src = i1 * v1_i64 + i2 * v2_i64

        src_off = base_i64 + outer_src + k64 * v0_i64
        dst_off = row.to(tl.int64) * d0_i64 + k64

        x = tl.load(x_ptr + src_off, mask=mask, eviction_policy='evict_last')
        tl.store(out_ptr + dst_off, x, mask=mask, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 1024}, num_warps=8, num_stages=4),
    ],
    key=['d0', 'outer_size'],
)
@triton.jit
def as_strided_kernel_tiled(
    x_ptr, out_ptr,
    base_offset,  # int64
    d0,           # inner (last) dimension size (reversed indexing)
    outer_size,   # product of remaining dimensions

    # Remaining dimensions (reversed, row-major), padded to MAX_DIMS
    d1, d2, d3, d4, d5, d6, d7,

    # View strides (reversed), can be negative
    vstr0, vstr1, vstr2, vstr3, vstr4, vstr5, vstr6, vstr7,

    # Number of active outer dims (excluding inner), in [0..7].
    nd_outer,

    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D tiled grid: pid_row_block iterates over tiles of rows, pid_col iterates over tiles along d0
    pid_row_block = tl.program_id(0)
    pid_col = tl.program_id(1)

    # k vector across inner dimension
    k = pid_col * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < d0
    k64 = k.to(tl.int64)

    # Scalars as int64
    base_i64 = tl.full([1], base_offset, tl.int64)
    d0_i64 = tl.full([1], d0, tl.int64)
    v0_i64 = tl.full([1], vstr0, tl.int64)

    # Cache outer dims/strides as int64 scalars
    d1_i64 = tl.full([1], d1, tl.int64)
    d2_i64 = tl.full([1], d2, tl.int64)
    d3_i64 = tl.full([1], d3, tl.int64)
    d4_i64 = tl.full([1], d4, tl.int64)
    d5_i64 = tl.full([1], d5, tl.int64)
    d6_i64 = tl.full([1], d6, tl.int64)
    d7_i64 = tl.full([1], d7, tl.int64)

    v1_i64 = tl.full([1], vstr1, tl.int64)
    v2_i64 = tl.full([1], vstr2, tl.int64)
    v3_i64 = tl.full([1], vstr3, tl.int64)
    v4_i64 = tl.full([1], vstr4, tl.int64)
    v5_i64 = tl.full([1], vstr5, tl.int64)
    v6_i64 = tl.full([1], vstr6, tl.int64)
    v7_i64 = tl.full([1], vstr7, tl.int64)

    # Process BLOCK_M rows per program to reduce grid size and improve locality
    for m in range(BLOCK_M):
        row = pid_row_block * BLOCK_M + m
        row_mask = row < outer_size
        mask = row_mask & k_mask

        # Decode multi-dim indices for outer coordinates
        tmp = row.to(tl.int64)

        i1 = tl.where(nd_outer >= 1, tmp % d1_i64, tl.zeros([1], dtype=tl.int64))
        tmp = tl.where(nd_outer >= 1, tmp // d1_i64, tmp)

        i2 = tl.where(nd_outer >= 2, tmp % d2_i64, tl.zeros([1], dtype=tl.int64))
        tmp = tl.where(nd_outer >= 2, tmp // d2_i64, tmp)

        i3 = tl.where(nd_outer >= 3, tmp % d3_i64, tl.zeros([1], dtype=tl.int64))
        tmp = tl.where(nd_outer >= 3, tmp // d3_i64, tmp)

        i4 = tl.where(nd_outer >= 4, tmp % d4_i64, tl.zeros([1], dtype=tl.int64))
        tmp = tl.where(nd_outer >= 4, tmp // d4_i64, tmp)

        i5 = tl.where(nd_outer >= 5, tmp % d5_i64, tl.zeros([1], dtype=tl.int64))
        tmp = tl.where(nd_outer >= 5, tmp // d5_i64, tmp)

        i6 = tl.where(nd_outer >= 6, tmp % d6_i64, tl.zeros([1], dtype=tl.int64))
        tmp = tl.where(nd_outer >= 6, tmp // d6_i64, tmp)

        i7 = tl.where(nd_outer >= 7, tmp % d7_i64, tl.zeros([1], dtype=tl.int64))
        # tmp // d7 not needed further

        # Compute source base offset for this outer row
        outer_src = (
            i1 * v1_i64 +
            i2 * v2_i64 +
            i3 * v3_i64 +
            i4 * v4_i64 +
            i5 * v5_i64 +
            i6 * v6_i64 +
            i7 * v7_i64
        )

        # Inner offsets
        src_off = base_i64 + outer_src + k64 * v0_i64
        dst_off = row.to(tl.int64) * d0_i64 + k64

        x = tl.load(x_ptr + src_off, mask=mask, eviction_policy='evict_last')
        tl.store(out_ptr + dst_off, x, mask=mask, eviction_policy='evict_last')


def as_strided(self: torch.Tensor, size, stride, storage_offset=None):
    # Normalize inputs
    out_shape = [int(s) for s in size]
    view_stride = [int(s) for s in stride]
    n_elements = _prod(out_shape)

    # Early return for empty outputs
    if n_elements == 0:
        return torch.empty(out_shape, dtype=self.dtype, device=self.device)

    # Output tensor is materialized (contiguous)
    out = torch.empty(out_shape, dtype=self.dtype, device=self.device)

    # Base offset relative to self tensor pointer (which already includes self.storage_offset())
    base_storage_offset = self.storage_offset()
    if storage_offset is None:
        target_storage_offset = base_storage_offset
    else:
        target_storage_offset = int(storage_offset)
    base_offset_kernel = target_storage_offset - base_storage_offset

    # Fast path: contiguous strides mapping (read is contiguous, write is contiguous)
    cont_strides = _contiguous_strides(out_shape)
    is_contiguous_mapping = (len(view_stride) == len(cont_strides) and all(v == c for v, c in zip(view_stride, cont_strides)))

    if is_contiguous_mapping:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        as_strided_kernel_contiguous[grid](
            self, out,
            base_offset_kernel, n_elements
        )
        return out

    # General path optimized: vectorize along innermost (row-major) dimension
    # Prepare reversed shapes and strides
    d_rev = out_shape[::-1]
    vstr_rev = view_stride[::-1]

    # Pad arrays to MAX_DIMS
    d_pad = _pad_to_max_dims(d_rev, fill=1)
    vstr_pad = _pad_to_max_dims(vstr_rev, fill=0)

    d0 = int(d_pad[0])
    outer_size = n_elements // d0

    # Count active outer dims (exclude inner d0). We treat dims with size>1 as active.
    # Build compressed active dims for specialized kernels
    active_sizes = []
    active_strides = []
    for j in range(1, MAX_DIMS):
        sz = int(d_pad[j])
        if sz > 1:
            active_sizes.append(sz)
            active_strides.append(int(vstr_pad[j]))

    nd_outer = len(active_sizes)

    # Special path: single outer row -> 1D step kernel (minimal index math)
    if outer_size == 1:
        grid = lambda meta: (triton.cdiv(d0, meta['BLOCK_SIZE']),)
        as_strided_kernel_1d_step[grid](
            self, out,
            base_offset_kernel,
            int(vstr_pad[0]),
            d0
        )
        return out

    # Specialized path: exactly 1 active outer dimension (2D materialization)
    if nd_outer == 1:
        d1 = int(active_sizes[0])
        v1 = int(active_strides[0])
        grid = lambda meta: (triton.cdiv(outer_size, meta['BLOCK_M']), triton.cdiv(d0, meta['BLOCK_K']))
        as_strided_kernel_outer1_tiled[grid](
            self, out,
            base_offset_kernel,
            d0, outer_size,
            d1,
            int(vstr_pad[0]), v1,
        )
        return out

    # Specialized path: exactly 2 active outer dimensions (3D materialization)
    if nd_outer == 2:
        d1 = int(active_sizes[0])
        d2 = int(active_sizes[1])
        v1 = int(active_strides[0])
        v2 = int(active_strides[1])
        grid = lambda meta: (triton.cdiv(outer_size, meta['BLOCK_M']), triton.cdiv(d0, meta['BLOCK_K']))
        as_strided_kernel_outer2_tiled[grid](
            self, out,
            base_offset_kernel,
            d0, outer_size,
            d1, d2,
            int(vstr_pad[0]), v1, v2,
        )
        return out

    # Fallback: 2D tiled kernel over outer rows and tiles along inner dimension (general N-D)
    grid = lambda meta: (triton.cdiv(outer_size, meta['BLOCK_M']), triton.cdiv(d0, meta['BLOCK_K']))
    as_strided_kernel_tiled[grid](
        self, out,
        base_offset_kernel,
        d0, outer_size,
        int(d_pad[1]), int(d_pad[2]), int(d_pad[3]), int(d_pad[4]), int(d_pad[5]), int(d_pad[6]), int(d_pad[7]),
        int(vstr_pad[0]), int(vstr_pad[1]), int(vstr_pad[2]), int(vstr_pad[3]), int(vstr_pad[4]), int(vstr_pad[5]), int(vstr_pad[6]), int(vstr_pad[7]),
        nd_outer
    )

    return out
