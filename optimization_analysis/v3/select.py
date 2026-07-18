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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def select_kernel_contiguous(
    x_ptr,
    out_ptr,
    base_offset,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    r = offs.to(tl.int64)
    # Stream through input and output linearly; use cache hints for streaming
    x = tl.load(x_ptr + (base_offset + r), mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + r, x, mask=mask, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def select_kernel_strided(
    x_ptr,
    out_ptr,
    n_elements,
    # Output shape dims (reversed, padded to MAX_DIMS)
    d0, d1, d2, d3, d4, d5, d6, d7,
    # Input strides mapped to output dims (reversed)
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    # Base offset for the fixed selected index, includes storage_offset
    base_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    r = offs.to(tl.int64)

    # Progressive quotient / remainder decomposition to reduce divisions:
    # q_k = r // prod(d[:k]), i_k = q_k % d_k
    q = r
    # d* and xst* are scalars; math auto-upcasts to int64 with r
    q_div = q // d0
    i0 = q - q_div * d0
    q = q_div

    q_div = q // d1
    i1 = q - q_div * d1
    q = q_div

    q_div = q // d2
    i2 = q - q_div * d2
    q = q_div

    q_div = q // d3
    i3 = q - q_div * d3
    q = q_div

    q_div = q // d4
    i4 = q - q_div * d4
    q = q_div

    q_div = q // d5
    i5 = q - q_div * d5
    q = q_div

    q_div = q // d6
    i6 = q - q_div * d6
    q = q_div

    # Final dim
    # Note: d7 can be 1 (padded); this still works
    # No need to compute next q
    q_div = q // d7
    i7 = q - q_div * d7

    # Compute input offset
    x_off = (
        i0.to(tl.int64) * xst0
        + i1.to(tl.int64) * xst1
        + i2.to(tl.int64) * xst2
        + i3.to(tl.int64) * xst3
        + i4.to(tl.int64) * xst4
        + i5.to(tl.int64) * xst5
        + i6.to(tl.int64) * xst6
        + i7.to(tl.int64) * xst7
        + base_offset
    )

    x = tl.load(x_ptr + x_off, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + r, x, mask=mask, eviction_policy='evict_last')


def _normalize_dim(dim, ndim):
    dim = int(dim)
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"Dimension out of range: dim={dim}, ndim={ndim}")
    return dim


def _normalize_index(index, size):
    idx = int(index)
    if idx < 0:
        idx += size
    if idx < 0 or idx >= size:
        raise IndexError(f"Index out of range: index={index}, size={size}")
    return idx


def _prod(seq):
    p = 1
    for v in seq:
        p *= int(v)
    return p


def _select_prepare_strided(t: torch.Tensor, sel_dim: int, sel_index: int):
    in_shape = list(t.shape)
    in_strides = list(t.stride())
    ndim = len(in_shape)

    sel_dim = _normalize_dim(sel_dim, ndim)
    sel_index = _normalize_index(sel_index, in_shape[sel_dim])

    # Output shape: remove selected dimension
    out_shape = in_shape[:sel_dim] + in_shape[sel_dim + 1:]
    n_elements = _prod(out_shape)

    # Build mapping from output dims to original dims (preserved order)
    out_to_in = [d for d in range(ndim) if d != sel_dim]
    # Reverse for row-major linearization
    out_shape_rev = list(reversed(out_shape))
    out_to_in_rev = list(reversed(out_to_in))

    # Pad to MAX_DIMS
    pad_len = MAX_DIMS - len(out_shape_rev)
    d_rev_padded = out_shape_rev + [1] * pad_len
    # Input strides mapped to output dims; for padded dims, 0 is safe
    xst_rev_padded = [in_strides[d] for d in out_to_in_rev] + [0] * pad_len

    # Base offset includes storage_offset and fixed index along sel_dim
    base_offset = t.storage_offset() + sel_index * in_strides[sel_dim]

    return n_elements, d_rev_padded, xst_rev_padded, base_offset


def _select_launch(t: torch.Tensor, sel_dim: int, sel_index: int):
    device = t.device
    dtype = t.dtype

    in_shape = list(t.shape)
    sel_dim = _normalize_dim(sel_dim, len(in_shape))
    sel_index = _normalize_index(sel_index, in_shape[sel_dim])

    out_shape = in_shape[:sel_dim] + in_shape[sel_dim + 1:]
    out = torch.empty(out_shape, dtype=dtype, device=device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Fast path: contiguous slice becomes a single linear block when
    # - tensor is contiguous AND
    # - all dimensions before sel_dim have size 1 (degenerate), including the common sel_dim==0 case
    if t.is_contiguous() and _prod(in_shape[:sel_dim]) == 1:
        base_offset = t.storage_offset() + sel_index * t.stride(sel_dim)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        select_kernel_contiguous[grid](t, out, base_offset, n_elements)
    else:
        n_elements2, d_rev, xst_rev, base_offset = _select_prepare_strided(t, sel_dim, sel_index)
        grid = lambda meta: (triton.cdiv(n_elements2, meta['BLOCK_SIZE']),)
        select_kernel_strided[grid](
            t, out, n_elements2,
            d_rev[0], d_rev[1], d_rev[2], d_rev[3], d_rev[4], d_rev[5], d_rev[6], d_rev[7],
            xst_rev[0], xst_rev[1], xst_rev[2], xst_rev[3], xst_rev[4], xst_rev[5], xst_rev[6], xst_rev[7],
            base_offset,
        )
    return out


def select(self: torch.Tensor, dim, index) -> torch.Tensor:
    # Dispatch between int and named dimension
    if isinstance(dim, str):
        if hasattr(self, 'names') and self.names is not None:
            try:
                dim_idx = self.names.index(dim)
            except ValueError:
                raise ValueError(f"Dimension name '{dim}' not found in tensor names {self.names}")
        else:
            raise ValueError("Tensor does not have named dimensions; cannot use select.Dimname")
        return _select_launch(self, dim_idx, index)
    else:
        return _select_launch(self, int(dim), index)


# Optional helpers for parity with earlier API variants
def select_int(self: torch.Tensor, dim: int, index) -> torch.Tensor:
    dim = int(dim)
    return _select_launch(self, dim, index)


def select_Dimname(self: torch.Tensor, dim: str, index) -> torch.Tensor:
    if hasattr(self, 'names') and self.names is not None:
        try:
            dim_idx = self.names.index(dim)
        except ValueError:
            raise ValueError(f"Dimension name '{dim}' not found in tensor names {self.names}")
    else:
        raise ValueError("Tensor does not have named dimensions; cannot use select.Dimname")
    return _select_launch(self, dim_idx, index)


def select_t(lst, idx):
    return lst[int(idx)]
