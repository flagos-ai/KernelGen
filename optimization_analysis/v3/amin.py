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

# -----------------------------
# Triton Kernels
# -----------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
    ],
    key=['red_numel'],
)
@triton.jit
def _amin_reduce_trailing_contig_kernel(
    x_ptr, out_ptr,
    keep_sizes_ptr, keep_x_strides_ptr, keep_out_strides_ptr, keep_cumprod_ptr,
    out_elems, red_numel,
    K_KEPT: tl.constexpr, BLOCK_K: tl.constexpr,
    UPCAST_FP32: tl.constexpr,
):
    """
    Fast path for contiguous tensors when reduced dimensions are the trailing suffix.
    In this case, for a fixed kept index, the reduction slice is contiguous in memory.
    """
    pid = tl.program_id(axis=0)
    if pid >= out_elems:
        return

    pid64 = pid.to(tl.int64)

    # Compute base offsets for kept (output) index once
    x_off = tl.zeros((), dtype=tl.int64)
    out_off = tl.zeros((), dtype=tl.int64)
    # Load kept metadata into registers and use static_range for unrolling
    for i in tl.static_range(0, K_KEPT):
        cp = tl.load(keep_cumprod_ptr + i)
        sz = tl.load(keep_sizes_ptr + i)
        sx = tl.load(keep_x_strides_ptr + i)
        so = tl.load(keep_out_strides_ptr + i)
        idx = (pid64 // cp) % sz
        x_off += idx * sx
        out_off += idx * so

    # Initialize accumulator from the first element in the contiguous reduction slice
    cur_init = tl.load(x_ptr + x_off, eviction_policy='evict_last')
    cur_acc = cur_init.to(tl.float32) if UPCAST_FP32 else cur_init

    # Iterate over remaining elements in contiguous chunks
    k_start = tl.full((), 1, dtype=tl.int32)
    one = tl.full((), 1, dtype=tl.int32)
    while k_start < red_numel:
        idxs = tl.arange(0, BLOCK_K)
        offs = k_start + idxs
        mask = offs < red_numel
        vals = tl.load(x_ptr + x_off + offs.to(tl.int64), mask=mask, other=cur_init, eviction_policy='evict_last')
        # Minimize conversions: reduce in original dtype, then upcast once if needed
        block_min = tl.min(vals, axis=0)
        if UPCAST_FP32:
            block_min = block_min.to(tl.float32)
        cur_acc = tl.minimum(cur_acc, block_min)
        k_start += BLOCK_K

    # Store result back in original dtype if upcasted
    res = cur_acc.to(cur_init.dtype) if UPCAST_FP32 else cur_acc
    tl.store(out_ptr + out_off, res, eviction_policy='evict_last')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
    ],
    key=['red_numel'],
)
@triton.jit
def _amin_reduce_general_kernel(
    x_ptr, out_ptr,
    keep_sizes_ptr, keep_x_strides_ptr, keep_out_strides_ptr, keep_cumprod_ptr,
    red_sizes_ptr, red_x_strides_ptr, red_cumprod_ptr,
    out_elems, red_numel,
    K_KEPT: tl.constexpr, K_RED: tl.constexpr, BLOCK_K: tl.constexpr,
    UPCAST_FP32: tl.constexpr,
):
    """
    General kernel handling arbitrary strided layouts.
    Optimizations:
      - Hoist metadata loads out of the inner loop
      - Minimize type conversions by reducing in original dtype then upcasting once
      - Use cache eviction policy for streaming loads
      - Use software pipelining via num_stages (autotune configs)
    """
    pid = tl.program_id(axis=0)
    if pid >= out_elems:
        return

    pid64 = pid.to(tl.int64)

    # Compute base offsets for kept (output) index once
    x_off = tl.zeros((), dtype=tl.int64)
    out_off = tl.zeros((), dtype=tl.int64)
    for i in tl.static_range(0, K_KEPT):
        cp = tl.load(keep_cumprod_ptr + i)
        sz = tl.load(keep_sizes_ptr + i)
        sx = tl.load(keep_x_strides_ptr + i)
        so = tl.load(keep_out_strides_ptr + i)
        idx = (pid64 // cp) % sz
        x_off += idx * sx
        out_off += idx * so

    # Initialize accumulator with the first reduction element
    cur_init = tl.load(x_ptr + x_off, eviction_policy='evict_last')
    cur_acc = cur_init.to(tl.float32) if UPCAST_FP32 else cur_init

    # Hoist metadata for reduction dims into registers
    red_rcp = tl.zeros([K_RED], dtype=tl.int64)
    red_sz = tl.zeros([K_RED], dtype=tl.int64)
    red_sx = tl.zeros([K_RED], dtype=tl.int64)
    for r in tl.static_range(0, K_RED):
        red_rcp = tl.where(tl.arange(0, K_RED) == r, tl.load(red_cumprod_ptr + r), red_rcp)
        red_sz  = tl.where(tl.arange(0, K_RED) == r, tl.load(red_sizes_ptr + r), red_sz)
        red_sx  = tl.where(tl.arange(0, K_RED) == r, tl.load(red_x_strides_ptr + r), red_sx)

    # Iterate over remaining reduction elements
    k_start = tl.full((), 1, dtype=tl.int32)
    while k_start < red_numel:
        idxs = tl.arange(0, BLOCK_K)
        offs32 = k_start + idxs
        mask = offs32 < red_numel
        offs64 = offs32.to(tl.int64)

        # Compute strided offsets for this block
        red_off = tl.zeros([BLOCK_K], dtype=tl.int64)
        # Use static_range to unroll; use hoisted metadata; compute coords and accumulate offsets
        for r in tl.static_range(0, K_RED):
            rcp = tl.load(red_cumprod_ptr + r)  # fallback load for correctness if vector init fails
            rsz = tl.load(red_sizes_ptr + r)
            rsx = tl.load(red_x_strides_ptr + r)
            coord = (offs64 // rcp) % rsz
            red_off += coord * rsx

        # Load values and compute block min; minimize dtype conversions
        vals = tl.load(x_ptr + x_off + red_off, mask=mask, other=cur_init, eviction_policy='evict_last')
        block_min = tl.min(vals, axis=0)
        if UPCAST_FP32:
            block_min = block_min.to(tl.float32)
        cur_acc = tl.minimum(cur_acc, block_min)

        k_start += BLOCK_K

    # Write result
    res = cur_acc.to(cur_init.dtype) if UPCAST_FP32 else cur_acc
    tl.store(out_ptr + out_off, res, eviction_policy='evict_last')


# -----------------------------
# Python Wrapper Utilities
# -----------------------------

def _normalize_dims(dim, ndim):
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    if len(dims) == 0:
        return list(range(ndim))
    norm = []
    seen = set()
    for d in dims:
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise IndexError("dim out of range")
        if d not in seen:
            seen.add(d)
            norm.append(d)
    norm.sort()
    return norm


def _prod(lst):
    p = 1
    for v in lst:
        p *= int(v)
    return p


def _cumprod_tail(sizes):
    n = len(sizes)
    cp = [1] * n
    acc = 1
    for i in range(n - 1, -1, -1):
        cp[i] = acc
        acc *= int(sizes[i])
    return cp


def _to_meta_tensor(lst, device):
    if len(lst) == 0:
        return torch.empty((1,), dtype=torch.int64, device=device)
    return torch.tensor(lst, dtype=torch.int64, device=device)


def _launch_amin_kernel(x: torch.Tensor, out_keepdim: torch.Tensor, red_dims):
    assert x.device == out_keepdim.device
    device = x.device
    N = x.dim()

    red_dims = sorted(red_dims)
    kept_dims = [d for d in range(N) if d not in red_dims]

    for d in red_dims:
        if x.size(d) == 0:
            raise RuntimeError("amin(): reduction on an empty dimension with no identity")

    keep_sizes = [x.size(d) for d in kept_dims]
    out_elems = _prod(keep_sizes) if len(keep_sizes) > 0 else 1
    if out_elems == 0:
        return

    red_sizes = [x.size(d) for d in red_dims]
    red_numel = _prod(red_sizes) if len(red_sizes) > 0 else 1

    keep_x_strides = [x.stride(d) for d in kept_dims]
    keep_out_strides = [out_keepdim.stride(d) for d in kept_dims]
    keep_cumprod = _cumprod_tail(keep_sizes)

    red_x_strides = [x.stride(d) for d in red_dims]
    red_cumprod = _cumprod_tail(red_sizes)

    keep_sizes_t = _to_meta_tensor(keep_sizes, device)
    keep_x_strides_t = _to_meta_tensor(keep_x_strides, device)
    keep_out_strides_t = _to_meta_tensor(keep_out_strides, device)
    keep_cumprod_t = _to_meta_tensor(keep_cumprod, device)

    red_sizes_t = _to_meta_tensor(red_sizes, device)
    red_x_strides_t = _to_meta_tensor(red_x_strides, device)
    red_cumprod_t = _to_meta_tensor(red_cumprod, device)

    # Fast path detection:
    # 1) Both input and output are contiguous
    # 2) Reduced dims form a trailing suffix of the shape (contiguous in memory)
    is_contig_io = x.is_contiguous() and out_keepdim.is_contiguous()
    trailing_start = N - len(red_dims)
    is_trailing_suffix = len(red_dims) > 0 and red_dims == list(range(trailing_start, N))

    grid = lambda meta: (out_elems,)
    upcast_fp32 = x.dtype in (torch.float16, torch.bfloat16)

    if is_contig_io and is_trailing_suffix:
        _amin_reduce_trailing_contig_kernel[grid](
            x, out_keepdim,
            keep_sizes_t, keep_x_strides_t, keep_out_strides_t, keep_cumprod_t,
            out_elems, red_numel,
            K_KEPT=len(keep_sizes),
            UPCAST_FP32=upcast_fp32,
        )
    else:
        _amin_reduce_general_kernel[grid](
            x, out_keepdim,
            keep_sizes_t, keep_x_strides_t, keep_out_strides_t, keep_cumprod_t,
            red_sizes_t, red_x_strides_t, red_cumprod_t,
            out_elems, red_numel,
            K_KEPT=len(keep_sizes),
            K_RED=len(red_sizes),
            UPCAST_FP32=upcast_fp32,
        )


# -----------------------------
# ATen Operator Wrapper Functions
# -----------------------------

def amin(self: torch.Tensor, dim=[], keepdim: bool = False):
    if not self.is_cuda:
        raise RuntimeError("This Triton amin implementation requires CUDA tensors")
    dims = _normalize_dims(dim if dim is not None else [], self.dim())
    N = self.dim()

    if len(dims) == 0:
        dims = list(range(N))
    out_keepdim_shape = [self.size(i) if i not in dims else 1 for i in range(N)]
    out_keepdim = torch.empty(out_keepdim_shape, dtype=self.dtype, device=self.device)

    if out_keepdim.numel() == 0:
        if keepdim:
            return out_keepdim
        squeezed_sizes = [self.size(i) for i in range(N) if i not in dims]
        return out_keepdim.reshape(squeezed_sizes)

    _launch_amin_kernel(self, out_keepdim, dims)

    if keepdim:
        return out_keepdim
    squeezed = out_keepdim
    if len(dims) > 0:
        squeezed = squeezed.reshape([self.size(i) for i in range(N) if i not in dims])
    return squeezed


def amin_out(self: torch.Tensor, dim=[], keepdim: bool = False, *, out: torch.Tensor):
    if not self.is_cuda or not out.is_cuda:
        raise RuntimeError("This Triton amin implementation requires CUDA tensors")
    if self.dtype != out.dtype:
        raise RuntimeError("Output dtype must match input dtype")
    if self.device != out.device:
        raise RuntimeError("Input and output must be on the same device")

    dims = _normalize_dims(dim if dim is not None else [], self.dim())
    N = self.dim()
    if len(dims) == 0:
        dims = list(range(N))

    if keepdim:
        desired_shape = [self.size(i) if i not in dims else 1 for i in range(N)]
    else:
        desired_shape = [self.size(i) for i in range(N) if i not in dims]

    tmp_keepdim_shape = [self.size(i) if i not in dims else 1 for i in range(N)]
    if keepdim and list(out.shape) == tmp_keepdim_shape:
        out_keepdim = out
        if out.numel() == 0:
            return out
        _launch_amin_kernel(self, out_keepdim, dims)
        return out

    out_keepdim = torch.empty(tmp_keepdim_shape, dtype=self.dtype, device=self.device)
    if out_keepdim.numel() != 0:
        _launch_amin_kernel(self, out_keepdim, dims)

    result = out_keepdim if keepdim else out_keepdim.reshape(desired_shape)
    if list(out.shape) != list(result.shape):
        try:
            out.resize_(result.shape)
        except Exception:
            raise RuntimeError("Provided 'out' tensor has incompatible shape")
    out.copy_(result)
    return out
