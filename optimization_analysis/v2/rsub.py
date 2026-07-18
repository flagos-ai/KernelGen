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

def _pad_to_max_dims(shape):
    shape = list(shape)
    return [1] * (MAX_DIMS - len(shape)) + shape

def _pad_strides_to_max_dims(strides, ndims):
    strides = list(strides)
    return [0] * (MAX_DIMS - ndims) + strides

def _broadcast_strides(in_shape, in_strides, out_shape):
    in_shape_p = _pad_to_max_dims(in_shape)
    out_shape_p = _pad_to_max_dims(out_shape)
    in_strides_p = _pad_strides_to_max_dims(in_strides, len(in_shape))
    bstrides = []
    for i in range(MAX_DIMS):
        if in_shape_p[i] == 1 and out_shape_p[i] != 1:
            bstrides.append(0)
        else:
            bstrides.append(in_strides_p[i])
    return bstrides

def _reverse(lst):
    return list(reversed(lst))

def _cumprod_shape(shape_rev):
    # shape_rev is reversed padded shape of length MAX_DIMS
    s = [1] * MAX_DIMS
    prod = 1
    for i in range(MAX_DIMS - 1, -1, -1):
        s[i] = prod
        prod *= int(shape_rev[i])
    return s

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def rsub_contiguous_kernel(
    x_ptr, y_ptr, out_ptr,
    alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    out = y - alpha * x
    tl.store(out_ptr + offs, out, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def rsub_contiguous_scalar_kernel(
    x_ptr, out_ptr,
    other, alpha, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    out = other - alpha * x
    tl.store(out_ptr + offs, out, mask=mask)

@triton.jit
def rsub_strided_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    d0, d1, d2, d3, d4, d5, d6, d7,
    s0, s1, s2, s3, s4, s5, s6, s7,
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    yst0, yst1, yst2, yst3, yst4, yst5, yst6, yst7,
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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

    x_off = i0 * xst0 + i1 * xst1 + i2 * xst2 + i3 * xst3 + i4 * xst4 + i5 * xst5 + i6 * xst6 + i7 * xst7
    y_off = i0 * yst0 + i1 * yst1 + i2 * yst2 + i3 * yst3 + i4 * yst4 + i5 * yst5 + i6 * yst6 + i7 * yst7
    o_off = i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 + i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7

    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)
    out = y - alpha * x
    tl.store(out_ptr + o_off, out, mask=mask)

@triton.jit
def rsub_strided_scalar_kernel(
    x_ptr, out_ptr, n_elements,
    d0, d1, d2, d3, d4, d5, d6, d7,
    s0, s1, s2, s3, s4, s5, s6, s7,
    xst0, xst1, xst2, xst3, xst4, xst5, xst6, xst7,
    ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
    other, alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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

    x_off = i0 * xst0 + i1 * xst1 + i2 * xst2 + i3 * xst3 + i4 * xst4 + i5 * xst5 + i6 * xst6 + i7 * xst7
    o_off = i0 * ost0 + i1 * ost1 + i2 * ost2 + i3 * ost3 + i4 * ost4 + i5 * ost5 + i6 * ost6 + i7 * ost7

    x = tl.load(x_ptr + x_off, mask=mask)
    out = other - alpha * x
    tl.store(out_ptr + o_off, out, mask=mask)

def _launch_contiguous_tensor(x, y, out, alpha):
    n_elements = out.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    rsub_contiguous_kernel[grid](x, y, out, alpha, n_elements)

def _launch_contiguous_scalar(x, out, other, alpha):
    n_elements = out.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    rsub_contiguous_scalar_kernel[grid](x, out, other, alpha, n_elements)

def _launch_strided_tensor(x, y, out, out_shape, alpha):
    n_elements = out.numel()
    if n_elements == 0:
        return
    # Prepare broadcast strides and reversed dims
    x_bstrides = _broadcast_strides(x.shape, x.stride(), out_shape)
    y_bstrides = _broadcast_strides(y.shape, y.stride(), out_shape)
    out_strides = list(out.stride())
    out_strides_p = _pad_strides_to_max_dims(out_strides, len(out_strides))
    dims_p = _pad_to_max_dims(out_shape)
    dims_rev = _reverse(dims_p)
    s_rev = _cumprod_shape(dims_rev)
    xst_rev = _reverse(x_bstrides)
    yst_rev = _reverse(y_bstrides)
    ost_rev = _reverse(out_strides_p)

    grid = (triton.cdiv(n_elements, 1024),)
    rsub_strided_kernel[grid](
        x, y, out, n_elements,
        dims_rev[0], dims_rev[1], dims_rev[2], dims_rev[3], dims_rev[4], dims_rev[5], dims_rev[6], dims_rev[7],
        s_rev[0], s_rev[1], s_rev[2], s_rev[3], s_rev[4], s_rev[5], s_rev[6], s_rev[7],
        xst_rev[0], xst_rev[1], xst_rev[2], xst_rev[3], xst_rev[4], xst_rev[5], xst_rev[6], xst_rev[7],
        yst_rev[0], yst_rev[1], yst_rev[2], yst_rev[3], yst_rev[4], yst_rev[5], yst_rev[6], yst_rev[7],
        ost_rev[0], ost_rev[1], ost_rev[2], ost_rev[3], ost_rev[4], ost_rev[5], ost_rev[6], ost_rev[7],
        alpha,
        BLOCK_SIZE=1024,
    )

def _launch_strided_scalar(x, out, alpha, other):
    n_elements = out.numel()
    if n_elements == 0:
        return
    out_shape = out.shape
    x_bstrides = _broadcast_strides(x.shape, x.stride(), out_shape)
    out_strides = list(out.stride())
    out_strides_p = _pad_strides_to_max_dims(out_strides, len(out_strides))
    dims_p = _pad_to_max_dims(out_shape)
    dims_rev = _reverse(dims_p)
    s_rev = _cumprod_shape(dims_rev)
    xst_rev = _reverse(x_bstrides)
    ost_rev = _reverse(out_strides_p)

    grid = (triton.cdiv(n_elements, 1024),)
    rsub_strided_scalar_kernel[grid](
        x, out, n_elements,
        dims_rev[0], dims_rev[1], dims_rev[2], dims_rev[3], dims_rev[4], dims_rev[5], dims_rev[6], dims_rev[7],
        s_rev[0], s_rev[1], s_rev[2], s_rev[3], s_rev[4], s_rev[5], s_rev[6], s_rev[7],
        xst_rev[0], xst_rev[1], xst_rev[2], xst_rev[3], xst_rev[4], xst_rev[5], xst_rev[6], xst_rev[7],
        ost_rev[0], ost_rev[1], ost_rev[2], ost_rev[3], ost_rev[4], ost_rev[5], ost_rev[6], ost_rev[7],
        other, alpha,
        BLOCK_SIZE=1024,
    )

def rsub_Tensor(self: torch.Tensor, other: torch.Tensor, *, alpha=1):
    device = self.device
    assert device.type == 'cuda' and other.device == device, "Tensors must be on the same CUDA device"
    out_shape = torch.broadcast_shapes(self.shape, other.shape)
    result_dtype = torch.result_type(self, other)
    x = self.to(result_dtype)
    y = other.to(result_dtype)
    out = torch.empty(out_shape, dtype=result_dtype, device=device)

    if x.is_contiguous() and y.is_contiguous() and out.is_contiguous() and x.shape == y.shape == out.shape:
        _launch_contiguous_tensor(x, y, out, float(alpha))
    else:
        _launch_strided_tensor(x, y, out, out_shape, float(alpha))
    return out

def rsub_Scalar(self: torch.Tensor, other, alpha=1):
    device = self.device
    assert device.type == 'cuda', "Tensor must be on CUDA device"
    # Determine result dtype with scalar
    other_tensor = torch.tensor(other, device=device)
    result_dtype = torch.result_type(self, other_tensor)
    x = self.to(result_dtype)
    out = torch.empty(x.shape, dtype=result_dtype, device=device)
    other_val = float(other_tensor.item())
    alpha_val = float(alpha)
    if x.is_contiguous() and out.is_contiguous():
        _launch_contiguous_scalar(x, out, other_val, alpha_val)
    else:
        _launch_strided_scalar(x, out, alpha_val, other_val)
    return out

def rsub_Tensor_out(self: torch.Tensor, other: torch.Tensor, *, alpha=1, out: torch.Tensor):
    device = self.device
    assert device.type == 'cuda' and other.device == device and out.device == device, "All tensors must be on the same CUDA device"
    out_shape = torch.broadcast_shapes(self.shape, other.shape)
    result_dtype = torch.result_type(self, other)
    assert out.dtype == result_dtype, "Out tensor dtype must match result dtype"
    # Resize out to target shape if needed
    if tuple(out.shape) != tuple(out_shape):
        out.resize_(out_shape)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    x = self.to(result_dtype)
    y = other.to(result_dtype)
    if x.is_contiguous() and y.is_contiguous() and out.is_contiguous() and x.shape == y.shape == tuple(out_shape):
        _launch_contiguous_tensor(x, y, out, float(alpha))
    else:
        _launch_strided_tensor(x, y, out, out_shape, float(alpha))
    return out

def rsub_Scalar_out(self: torch.Tensor, other, alpha=1, *, out: torch.Tensor):
    device = self.device
    assert device.type == 'cuda' and out.device == device, "Tensor and out must be on the same CUDA device"
    other_tensor = torch.tensor(other, device=device)
    result_dtype = torch.result_type(self, other_tensor)
    assert out.dtype == result_dtype, "Out tensor dtype must match result dtype"
    # Resize out to target shape if needed
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    x = self.to(result_dtype)
    other_val = float(other_tensor.item())
    alpha_val = float(alpha)
    if x.is_contiguous() and out.is_contiguous():
        _launch_contiguous_scalar(x, out, other_val, alpha_val)
    else:
        _launch_strided_scalar(x, out, alpha_val, other_val)
    return out
