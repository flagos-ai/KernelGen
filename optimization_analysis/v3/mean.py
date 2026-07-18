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

# --------------------------
# Triton Kernels
# --------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OUT': 128, 'BLOCK_R': 256}, num_warps=4),
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_R': 128}, num_warps=8),
        triton.Config({'BLOCK_OUT': 64,  'BLOCK_R': 512}, num_warps=4),
    ],
    key=['R'],
)
@triton.jit
def mean_reduce_lastdim_contig_kernel(
    in_ptr, out_ptr,
    OUTER: tl.int64,   # number of output elements (product of sizes excluding reduced dim)
    R: tl.int64,       # reduction length
    USE_FP64: tl.constexpr,      # whether to accumulate in fp64
    OUT_DTYPE_CODE: tl.constexpr, # 0=f16,1=bf16,2=f32,3=f64
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)).to(tl.int64)
    mask_out = offs < OUTER

    base_in = (offs * R).to(tl.int64)
    # Ensure safe pointers for masked lanes
    base_in_safe = tl.where(mask_out, base_in, tl.zeros_like(base_in))

    comp_dtype = tl.float64 if USE_FP64 else tl.float32
    acc = tl.zeros([BLOCK_OUT], dtype=comp_dtype)

    r = tl.zeros((), dtype=tl.int64)
    one64 = tl.full((), 1, dtype=tl.int64)
    while r < R:
        r_arange = tl.arange(0, BLOCK_R).to(tl.int64)
        r_offsets = r + r_arange
        mask_r = r_offsets < R
        # Safe r offsets to avoid OOB pointer creation
        r_offsets_safe = tl.where(mask_r, r_offsets, tl.zeros_like(r_offsets))
        ptrs = in_ptr + base_in_safe[:, None] + r_offsets_safe[None, :]
        vals = tl.load(ptrs, mask=(mask_out[:, None] & mask_r[None, :]), other=0)
        vals = vals.to(comp_dtype)
        acc += tl.sum(vals, axis=1)
        r += BLOCK_R * one64

    R_f = tl.full((), R, dtype=comp_dtype)
    res = acc / R_f

    if OUT_DTYPE_CODE == 0:
        out_vals = res.to(tl.float16)
    elif OUT_DTYPE_CODE == 1:
        out_vals = res.to(tl.bfloat16)
    elif OUT_DTYPE_CODE == 2:
        out_vals = res.to(tl.float32)
    else:
        out_vals = res.to(tl.float64)

    offs_safe = tl.where(mask_out, offs, tl.zeros_like(offs))
    tl.store(out_ptr + offs_safe, out_vals, mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OUT': 128, 'BLOCK_R': 128}, num_warps=4),
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_R': 64}, num_warps=8),
        triton.Config({'BLOCK_OUT': 64,  'BLOCK_R': 256}, num_warps=4),
    ],
    key=['R'],
)
@triton.jit
def mean_reduce_axis_strided_kernel(
    in_ptr, out_ptr,
    OUTER: tl.int64,    # number of logical output elements (product of sizes excluding reduced dim)
    R: tl.int64,        # reduction size
    red_stride: tl.int64,  # input stride along the reduced dimension (in elements)
    out_sizes_ptr,          # int64[MAX_DIMS] - sizes of output logical dims (excluding reduced dim), padded with 1
    in_strides_ex_ptr,      # int64[MAX_DIMS] - input strides for non-reduced dims, aligned to out dims order, padded with 0
    out_strides_ex_ptr,     # int64[MAX_DIMS] - output strides for non-reduced dims, aligned to out dims order, padded with 0
    MAX_DIMS: tl.constexpr, # constexpr number of padded dims
    USE_FP64: tl.constexpr,      # use fp64 accumulation
    OUT_DTYPE_CODE: tl.constexpr, # 0=f16,1=bf16,2=f32,3=f64
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)).to(tl.int64)
    mask_out = offs < OUTER

    lin = offs
    base_in = tl.zeros([BLOCK_OUT], dtype=tl.int64)
    base_out = tl.zeros([BLOCK_OUT], dtype=tl.int64)
    # Decompose linear index into coordinates over the non-reduced dims
    for i in range(MAX_DIMS - 1, -1, -1):
        size_i = tl.load(out_sizes_ptr + i).to(tl.int64)
        stride_in_i = tl.load(in_strides_ex_ptr + i).to(tl.int64)
        stride_out_i = tl.load(out_strides_ex_ptr + i).to(tl.int64)
        coord_i = lin % size_i
        lin = lin // size_i
        base_in += coord_i * stride_in_i
        base_out += coord_i * stride_out_i

    # Safe bases for masked lanes
    base_in_safe = tl.where(mask_out, base_in, tl.zeros_like(base_in))
    base_out_safe = tl.where(mask_out, base_out, tl.zeros_like(base_out))

    comp_dtype = tl.float64 if USE_FP64 else tl.float32
    acc = tl.zeros([BLOCK_OUT], dtype=comp_dtype)

    r = tl.zeros((), dtype=tl.int64)
    one64 = tl.full((), 1, dtype=tl.int64)
    while r < R:
        r_arange = tl.arange(0, BLOCK_R).to(tl.int64)
        r_offsets = r + r_arange
        mask_r = r_offsets < R
        r_offsets_safe = tl.where(mask_r, r_offsets, tl.zeros_like(r_offsets))
        ptrs = in_ptr + base_in_safe[:, None] + r_offsets_safe[None, :] * red_stride
        vals = tl.load(ptrs, mask=(mask_out[:, None] & mask_r[None, :]), other=0)
        vals = vals.to(comp_dtype)
        acc += tl.sum(vals, axis=1)
        r += BLOCK_R * one64

    R_f = tl.full((), R, dtype=comp_dtype)
    res = acc / R_f

    if OUT_DTYPE_CODE == 0:
        out_vals = res.to(tl.float16)
    elif OUT_DTYPE_CODE == 1:
        out_vals = res.to(tl.bfloat16)
    elif OUT_DTYPE_CODE == 2:
        out_vals = res.to(tl.float32)
    else:
        out_vals = res.to(tl.float64)

    tl.store(out_ptr + base_out_safe, out_vals, mask=mask_out)


# --------------------------
# Python Wrappers
# --------------------------

_MAX_DIMS = 8

def _dtype_to_code(dtype: torch.dtype):
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.float64:
        return 3
    raise RuntimeError(f"Unsupported dtype for mean: {dtype}. Only floating types are supported.")

def _select_out_dtype(x: torch.Tensor, dtype):
    if dtype is not None:
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise RuntimeError("mean: dtype must be a floating type (float16, bfloat16, float32, float64).")
        return dtype
    if x.is_floating_point():
        return x.dtype
    return torch.float32

def _normalize_dims(x: torch.Tensor, dims):
    if dims is None:
        return None
    if isinstance(dims, int):
        dims = [dims]
    if not isinstance(dims, (list, tuple)):
        raise RuntimeError("dim must be int or a sequence of ints")
    nd = x.ndim
    norm = []
    for d in dims:
        if not isinstance(d, int):
            raise RuntimeError("dim list must contain ints")
        if d < 0:
            d += nd
        if d < 0 or d >= nd:
            raise RuntimeError(f"Invalid dim {d} for tensor with {nd} dims")
        norm.append(d)
    if len(set(norm)) != len(norm):
        raise RuntimeError("dim list contains duplicate dimensions")
    return tuple(norm)

def _normalize_name_dim(x: torch.Tensor, dim):
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            raise RuntimeError("mean.names_dim supports only a single dimension.")
        dim = dim[0]
    if not isinstance(dim, str):
        raise RuntimeError("mean.names_dim expects str dimension.")
    if not hasattr(x, 'names') or x.names is None:
        raise RuntimeError("Tensor has no names; cannot resolve named dimension.")
    names = list(x.names)
    if dim not in names:
        raise RuntimeError(f"Dimension name {dim} not found in tensor names {names}")
    return names.index(dim)

def _prepare_ex_dims(x: torch.Tensor, out_tensor: torch.Tensor, red_dim: int, keepdim: bool):
    in_strides = list(x.stride())
    out_strides = list(out_tensor.stride())
    if keepdim:
        out_strides_ex = out_strides[:red_dim] + out_strides[red_dim+1:]
        out_sizes_ex = list(out_tensor.shape[:red_dim] + out_tensor.shape[red_dim+1:])
    else:
        out_strides_ex = out_strides[:]
        out_sizes_ex = list(out_tensor.shape)

    in_strides_ex = in_strides[:red_dim] + in_strides[red_dim+1:]

    def pad_list(lst, pad_value, length):
        lst = list(lst)
        if len(lst) > length:
            raise RuntimeError(f"Too many dims: {len(lst)} > {length}")
        return lst + [pad_value] * (length - len(lst))

    out_sizes_ex = pad_list(out_sizes_ex, 1, _MAX_DIMS)
    in_strides_ex = pad_list(in_strides_ex, 0, _MAX_DIMS)
    out_strides_ex = pad_list(out_strides_ex, 0, _MAX_DIMS)

    return (
        torch.tensor(out_sizes_ex, device=x.device, dtype=torch.int64),
        torch.tensor(in_strides_ex, device=x.device, dtype=torch.int64),
        torch.tensor(out_strides_ex, device=x.device, dtype=torch.int64),
    )

def _reduce_one_dim(x: torch.Tensor, red_dim: int, keepdim: bool, out_dtype: torch.dtype, out: torch.Tensor = None):
    if not x.is_cuda:
        raise RuntimeError("Triton mean kernels require CUDA tensors.")
    compute_fp64 = (out_dtype == torch.float64) or (x.dtype == torch.float64)
    out_dtype_code = _dtype_to_code(out_dtype)

    in_sizes = list(x.shape)
    R = in_sizes[red_dim]

    # Determine output shape for this step
    if keepdim:
        out_sizes = list(in_sizes)
        out_sizes[red_dim] = 1
    else:
        out_sizes = in_sizes[:red_dim] + in_sizes[red_dim+1:]

    if out is None:
        out = torch.empty(out_sizes, device=x.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != tuple(out_sizes):
            raise RuntimeError(f"Provided out shape {tuple(out.shape)} does not match expected {tuple(out_sizes)}")
        if out.dtype != out_dtype:
            raise RuntimeError(f"Provided out dtype {out.dtype} does not match requested dtype {out_dtype}")

    # Handle empty input (any zero size implies numel==0)
    if x.numel() == 0:
        out.fill_(float('nan'))
        return out

    # Fast path: contiguous input, reduce last dim, contiguous output
    if x.is_contiguous() and (red_dim == x.ndim - 1) and out.is_contiguous():
        n_outer = 1
        for s in in_sizes[:red_dim]:
            n_outer *= s
        OUTER = n_outer
        grid = lambda meta: (triton.cdiv(OUTER, meta['BLOCK_OUT']),)
        mean_reduce_lastdim_contig_kernel[grid](
            x, out,
            OUTER, R,
            USE_FP64=compute_fp64,
            OUT_DTYPE_CODE=out_dtype_code,
        )
        return out

    # General strided reduction along arbitrary dim
    out_sizes_ex, in_strides_ex, out_strides_ex = _prepare_ex_dims(x, out, red_dim, keepdim)
    red_stride = x.stride()[red_dim]
    OUTER = out.numel() if keepdim else out.numel()
    grid = lambda meta: (triton.cdiv(OUTER, meta['BLOCK_OUT']),)
    mean_reduce_axis_strided_kernel[grid](
        x, out,
        OUTER, R,
        red_stride,
        out_sizes_ex, in_strides_ex, out_strides_ex,
        MAX_DIMS=_MAX_DIMS,
        USE_FP64=compute_fp64,
        OUT_DTYPE_CODE=out_dtype_code,
    )
    return out

def _compute_out_shape(in_shape, dims, keepdim):
    in_shape = list(in_shape)
    if dims is None:
        return ()
    if len(dims) == 0:
        # No reduction, shape unchanged (or identical with keepdim)
        return tuple(in_shape)
    if keepdim:
        out_shape = list(in_shape)
        for d in dims:
            out_shape[d] = 1
        return tuple(out_shape)
    else:
        # remove dims in descending order
        out_shape = list(in_shape)
        for d in sorted(dims, reverse=True):
            del out_shape[d]
        return tuple(out_shape)

def _launch_mean(x: torch.Tensor, dim=None, keepdim: bool = False, dtype=None, out: torch.Tensor = None):
    if not x.is_cuda:
        raise RuntimeError("Triton mean kernels require CUDA tensors.")
    dims = _normalize_dims(x, dim) if dim is not None else None
    out_dtype = _select_out_dtype(x, dtype)

    # Handle global reduction (over all elements)
    if dims is None:
        out_shape = () if out is None else out.shape
        if out is None:
            out = torch.empty((), device=x.device, dtype=out_dtype)
        else:
            if out.dtype != out_dtype:
                raise RuntimeError(f"Provided out dtype {out.dtype} does not match requested dtype {out_dtype}")
            if out.numel() != 1:
                raise RuntimeError("mean over all elements expects scalar output (numel=1).")

        if x.numel() == 0:
            out.fill_(float('nan'))
            return out

        # Use contiguous reduction if needed
        if x.is_contiguous() and (out.is_contiguous() or out.numel() == 1):
            OUTER = 1
            R = x.numel()
            grid = lambda meta: (1,)
            mean_reduce_lastdim_contig_kernel[grid](
                x, out,
                OUTER, R,
                USE_FP64=(out_dtype == torch.float64 or x.dtype == torch.float64),
                OUT_DTYPE_CODE=_dtype_to_code(out_dtype),
            )
            return out
        else:
            x_contig = x.contiguous()
            OUTER = 1
            R = x_contig.numel()
            grid = lambda meta: (1,)
            mean_reduce_lastdim_contig_kernel[grid](
                x_contig, out,
                OUTER, R,
                USE_FP64=(out_dtype == torch.float64 or x_contig.dtype == torch.float64),
                OUT_DTYPE_CODE=_dtype_to_code(out_dtype),
            )
            return out

    # If dims is empty list => no reduction; just cast/copy
    if isinstance(dims, (list, tuple)) and len(dims) == 0:
        if out is None:
            return x.to(dtype=out_dtype)
        else:
            if out.dtype != out_dtype:
                raise RuntimeError(f"Provided out dtype {out.dtype} does not match requested dtype {out_dtype}")
            if tuple(out.shape) != tuple(x.shape if keepdim else x.shape):
                raise RuntimeError(f"Provided out shape {tuple(out.shape)} does not match expected {tuple(x.shape)}")
            out.copy_(x.to(out_dtype))
            return out

    # Prepare final output tensor
    final_shape = _compute_out_shape(x.shape, dims, keepdim)
    if out is None:
        final_out = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != tuple(final_shape):
            raise RuntimeError(f"Provided out shape {tuple(out.shape)} does not match expected {tuple(final_shape)}")
        if out.dtype != out_dtype:
            raise RuntimeError(f"Provided out dtype {out.dtype} does not match requested dtype {out_dtype}")
        final_out = out

    # Multi-dim reduction handled by chaining single-dim reductions
    curr = x
    if keepdim:
        # Reduce with keepdim=True for each dim; dims indices remain valid
        dims_order = list(dims)
        for i, d in enumerate(dims_order):
            is_last = (i == len(dims_order) - 1)
            step_out_shape = _compute_out_shape(curr.shape, (d,), True)
            step_out = final_out if is_last else torch.empty(step_out_shape, device=x.device, dtype=out_dtype)
            curr = _reduce_one_dim(curr, d, keepdim=True, out_dtype=out_dtype, out=step_out)
        return final_out
    else:
        # Reduce with keepdim=False in descending order, so indices stay valid
        dims_order = sorted(dims, reverse=True)
        for i, d in enumerate(dims_order):
            is_last = (i == len(dims_order) - 1)
            step_out_shape = _compute_out_shape(curr.shape, (d,), False)
            step_out = final_out if is_last else torch.empty(step_out_shape, device=x.device, dtype=out_dtype)
            curr = _reduce_one_dim(curr, d, keepdim=False, out_dtype=out_dtype, out=step_out)
        return final_out


# --------------------------
# ATen-style Wrapper Functions
# --------------------------

def mean(self: torch.Tensor, *, dtype: torch.dtype = None) -> torch.Tensor:
    return _launch_mean(self, dim=None, keepdim=False, dtype=dtype, out=None)

def mean_dim(self: torch.Tensor, dim=None, keepdim: bool = False, *, dtype: torch.dtype = None) -> torch.Tensor:
    # dim can be int or sequence of ints per ATen signature
    dims = _normalize_dims(self, dim) if dim is not None else None
    return _launch_mean(self, dim=dims, keepdim=keepdim, dtype=dtype, out=None)

def mean_names_dim(self: torch.Tensor, dim, keepdim: bool = False, *, dtype: torch.dtype = None) -> torch.Tensor:
    # ATen names_dim accepts a single named dimension
    dim_idx = _normalize_name_dim(self, dim)
    return _launch_mean(self, dim=(dim_idx,), keepdim=keepdim, dtype=dtype, out=None)

def mean_names_out(self: torch.Tensor, dim, keepdim: bool = False, *, dtype: torch.dtype = None, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        raise RuntimeError("mean.names_out: 'out' tensor must be provided")
    dim_idx = _normalize_name_dim(self, dim)
    return _launch_mean(self, dim=(dim_idx,), keepdim=keepdim, dtype=dtype, out=out)

def mean_out(self: torch.Tensor, dim=None, keepdim: bool = False, *, dtype: torch.dtype = None, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        raise RuntimeError("mean.out: 'out' tensor must be provided")
    dims = _normalize_dims(self, dim) if dim is not None else None
    return _launch_mean(self, dim=dims, keepdim=keepdim, dtype=dtype, out=out)

def mean_dtype_out(self: torch.Tensor, *, dtype: torch.dtype = None, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        raise RuntimeError("mean.dtype_out: 'out' tensor must be provided")
    return _launch_mean(self, dim=None, keepdim=False, dtype=dtype, out=out)
