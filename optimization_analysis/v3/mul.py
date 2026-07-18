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
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _mul_kernel_contiguous(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    r = offs.to(tl.int64)
    n64 = tl.full((), n_elements, tl.int64)
    mask = r < n64
    x = tl.load(x_ptr + r, mask=mask, other=0)
    y = tl.load(y_ptr + r, mask=mask, other=0)
    out = x * y
    tl.store(out_ptr + r, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _mul_kernel_strided(
    x_ptr, y_ptr, out_ptr, n_elements,
    d0, d1, d2, d3, d4, d5, d6, d7,
    s0, s1, s2, s3, s4, s5, s6, s7,
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    yst0, yst1, yst2, yst3, yst4, yst5, yst6, yst7,
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    r = offs.to(tl.int64)
    n64 = tl.full((), n_elements, tl.int64)
    mask = r < n64

    # Cast all scalar params to int64 to avoid overflow
    d0_ = tl.full((), d0, tl.int64)
    d1_ = tl.full((), d1, tl.int64)
    d2_ = tl.full((), d2, tl.int64)
    d3_ = tl.full((), d3, tl.int64)
    d4_ = tl.full((), d4, tl.int64)
    d5_ = tl.full((), d5, tl.int64)
    d6_ = tl.full((), d6, tl.int64)
    d7_ = tl.full((), d7, tl.int64)

    s0_ = tl.full((), s0, tl.int64)
    s1_ = tl.full((), s1, tl.int64)
    s2_ = tl.full((), s2, tl.int64)
    s3_ = tl.full((), s3, tl.int64)
    s4_ = tl.full((), s4, tl.int64)
    s5_ = tl.full((), s5, tl.int64)
    s6_ = tl.full((), s6, tl.int64)
    s7_ = tl.full((), s7, tl.int64)

    xst0_ = tl.full((), xst0, tl.int64)
    xst1_ = tl.full((), xst1, tl.int64)
    xst2_ = tl.full((), xst2, tl.int64)
    xst3_ = tl.full((), xst3, tl.int64)
    xst4_ = tl.full((), xst4, tl.int64)
    xst5_ = tl.full((), xst5, tl.int64)
    xst6_ = tl.full((), xst6, tl.int64)
    xst7_ = tl.full((), xst7, tl.int64)

    yst0_ = tl.full((), yst0, tl.int64)
    yst1_ = tl.full((), yst1, tl.int64)
    yst2_ = tl.full((), yst2, tl.int64)
    yst3_ = tl.full((), yst3, tl.int64)
    yst4_ = tl.full((), yst4, tl.int64)
    yst5_ = tl.full((), yst5, tl.int64)
    yst6_ = tl.full((), yst6, tl.int64)
    yst7_ = tl.full((), yst7, tl.int64)

    ost0_ = tl.full((), ost0, tl.int64)
    ost1_ = tl.full((), ost1, tl.int64)
    ost2_ = tl.full((), ost2, tl.int64)
    ost3_ = tl.full((), ost3, tl.int64)
    ost4_ = tl.full((), ost4, tl.int64)
    ost5_ = tl.full((), ost5, tl.int64)
    ost6_ = tl.full((), ost6, tl.int64)
    ost7_ = tl.full((), ost7, tl.int64)

    # Compute indices along each dimension
    i0 = (r // s0_) % d0_
    i1 = (r // s1_) % d1_
    i2 = (r // s2_) % d2_
    i3 = (r // s3_) % d3_
    i4 = (r // s4_) % d4_
    i5 = (r // s5_) % d5_
    i6 = (r // s6_) % d6_
    i7 = (r // s7_) % d7_

    # Compute offsets
    x_off = (i0 * xst0_ + i1 * xst1_ + i2 * xst2_ + i3 * xst3_ +
             i4 * xst4_ + i5 * xst5_ + i6 * xst6_ + i7 * xst7_)
    y_off = (i0 * yst0_ + i1 * yst1_ + i2 * yst2_ + i3 * yst3_ +
             i4 * yst4_ + i5 * yst5_ + i6 * yst6_ + i7 * yst7_)
    o_off = (i0 * ost0_ + i1 * ost1_ + i2 * ost2_ + i3 * ost3_ +
             i4 * ost4_ + i5 * ost5_ + i6 * ost6_ + i7 * ost7_)

    x = tl.load(x_ptr + x_off, mask=mask, other=0)
    y = tl.load(y_ptr + y_off, mask=mask, other=0)
    out = x * y
    tl.store(out_ptr + o_off, out, mask=mask)


def _broadcast_strides(in_shape, in_strides, out_shape):
    in_ndim = len(in_shape)
    out_ndim = len(out_shape)
    out_strides = [0] * out_ndim
    for i in range(1, out_ndim + 1):
        if i <= in_ndim:
            in_dim = in_shape[-i]
            out_dim = out_shape[-i]
            if in_dim == 1:
                out_strides[-i] = 0
            else:
                # If broadcast fails, PyTorch would have thrown earlier via broadcast_shapes
                # When in_dim == out_dim, use original stride
                out_strides[-i] = in_strides[-i]
        else:
            out_strides[-i] = 0
    return tuple(out_strides)


def _pad_to_max_dims(shape, strides):
    nd = len(shape)
    pad = MAX_DIMS - nd
    if pad < 0:
        return None, None
    shape_p = (1,) * pad + tuple(shape)
    strides_p = (0,) * pad + tuple(strides)
    return shape_p, strides_p


def _cumprod_strides(dims):
    s = [1] * len(dims)
    prod = 1
    for i in range(len(dims) - 1, -1, -1):
        s[i] = prod
        prod *= int(dims[i])
    return s


def _supports_triton_dtype(dtype: torch.dtype) -> bool:
    # Limit to dtypes Triton reliably supports for arithmetic
    return dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32)


def _ensure_out_shape(out: torch.Tensor, shape):
    if tuple(out.shape) != tuple(shape):
        out.resize_(shape)
    return out


def _to_result_dtype(x_t: torch.Tensor, y_t: torch.Tensor, out: torch.Tensor = None):
    return out.dtype if out is not None else torch.result_type(x_t, y_t)


def _binary_mul_tensor(x, y, out: torch.Tensor = None):
    # Determine device and base dtype from any tensor input
    if isinstance(x, torch.Tensor):
        device = x.device
        base_dtype = x.dtype
    elif isinstance(y, torch.Tensor):
        device = y.device
        base_dtype = y.dtype
    else:
        # No tensors: fallback to Python/scalar ops
        return torch.mul(torch.tensor(x), torch.tensor(y))

    # Canonicalize inputs to tensors on the same device
    if isinstance(x, torch.Tensor):
        x_t = x
    else:
        x_t = torch.tensor(x, device=device, dtype=base_dtype)
    if isinstance(y, torch.Tensor):
        y_t = y
    else:
        y_t = torch.tensor(y, device=device, dtype=base_dtype)

    # Device checks
    if x_t.device != device or y_t.device != device:
        raise ValueError("x and y must be on the same device")
    if out is not None and out.device != device:
        raise ValueError("out must be on the same device as inputs")

    # Result dtype
    res_dtype = _to_result_dtype(x_t, y_t, out)

    # Fallback conditions
    if device.type != 'cuda':
        return torch.mul(x_t.to(res_dtype), y_t.to(res_dtype), out=out)

    if x_t.ndim > MAX_DIMS or y_t.ndim > MAX_DIMS:
        return torch.mul(x_t.to(res_dtype), y_t.to(res_dtype), out=out)

    # Type promotions
    if out is not None:
        result_dtype = out.dtype
        x_cast = x_t.to(dtype=result_dtype)
        y_cast = y_t.to(dtype=result_dtype)
    else:
        result_dtype = res_dtype
        x_cast = x_t.to(dtype=result_dtype)
        y_cast = y_t.to(dtype=result_dtype)

    # Fallback for unsupported dtypes
    if not _supports_triton_dtype(result_dtype):
        return torch.mul(x_cast, y_cast, out=out)

    # Compute broadcasted shape
    out_shape = torch.broadcast_shapes(x_cast.shape, y_cast.shape)
    n_elements = 1
    for d in out_shape:
        n_elements *= int(d)

    if out is None:
        out_t = torch.empty(out_shape, dtype=result_dtype, device=device)
    else:
        out_t = _ensure_out_shape(out, out_shape)

    if n_elements == 0:
        return out_t

    # Fast path: contiguous, same shapes, out contiguous
    if (x_cast.is_contiguous() and y_cast.is_contiguous() and out_t.is_contiguous() and
        tuple(x_cast.shape) == tuple(out_shape) == tuple(y_cast.shape)):
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _mul_kernel_contiguous[grid](x_cast, y_cast, out_t, n_elements)
        return out_t

    # General strided path
    x_brd_strides = _broadcast_strides(x_cast.shape, x_cast.stride(), out_shape)
    y_brd_strides = _broadcast_strides(y_cast.shape, y_cast.stride(), out_shape)
    o_strides = out_t.stride()

    dims_p, _ = _pad_to_max_dims(out_shape, o_strides)
    if dims_p is None:
        return torch.mul(x_cast, y_cast, out=out)

    _, x_brd_strides_p = _pad_to_max_dims(out_shape, x_brd_strides)
    _, y_brd_strides_p = _pad_to_max_dims(out_shape, y_brd_strides)
    _, o_strides_p = _pad_to_max_dims(out_shape, o_strides)

    s = _cumprod_strides(dims_p)

    d0, d1, d2, d3, d4, d5, d6, d7 = [int(v) for v in dims_p]
    s0, s1, s2, s3, s4, s5, s6, s7 = [int(v) for v in s]
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7 = [int(v) for v in x_brd_strides_p]
    yst0, yst1, yst2, yst3, yst4, yst5, yst6, yst7 = [int(v) for v in y_brd_strides_p]
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7 = [int(v) for v in o_strides_p]

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _mul_kernel_strided[grid](
        x_cast, y_cast, out_t, n_elements,
        d0, d1, d2, d3, d4, d5, d6, d7,
        s0, s1, s2, s3, s4, s5, s6, s7,
        xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
        yst0, yst1, yst2, yst3, yst4, yst5, yst6, yst7,
        ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    )
    return out_t


# Wrapper functions for ATen interfaces

def mul_Tensor(self, other) -> torch.Tensor:
    return _binary_mul_tensor(self, other, out=None)

def mul_Scalar(self: torch.Tensor, other) -> torch.Tensor:
    return _binary_mul_tensor(self, other, out=None)

def mul_out(self: torch.Tensor, other, out: torch.Tensor) -> torch.Tensor:
    return _binary_mul_tensor(self, other, out=out)

def mul_Scalar_out(self: torch.Tensor, other, out: torch.Tensor) -> torch.Tensor:
    return _binary_mul_tensor(self, other, out=out)

def mul_left_t(l, n: int):
    return l * n

def mul_right_(n: int, l):
    return n * l

def mul_int(a: int, b: int) -> int:
    return int(a) * int(b)

def mul_complex(a: complex, b: complex) -> complex:
    return complex(a) * complex(b)

def mul_float(a: float, b: float) -> float:
    return float(a) * float(b)

def mul_int_complex(a: int, b: complex) -> complex:
    return complex(a) * complex(b)

def mul_complex_int(a: complex, b: int) -> complex:
    return complex(a) * int(b)

def mul_float_complex(a: float, b: complex) -> complex:
    return float(a) * complex(b)

def mul_complex_float(a: complex, b: float) -> complex:
    return complex(a) * float(b)

def mul_int_float(a: int, b: float) -> float:
    return float(a) * float(b)

def mul_float_int(a: float, b: int) -> float:
    return float(a) * int(b)

def mul(a, b):
    return a * b
