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

def _broadcast_strides(in_shape, in_strides, out_shape):
    """Compute broadcasted strides aligning from the right."""
    out_ndim = len(out_shape)
    in_ndim = len(in_shape)
    b_strides = [0] * out_ndim
    for i in range(out_ndim):
        out_idx = out_ndim - 1 - i
        in_idx = in_ndim - 1 - i
        if in_idx >= 0:
            s = in_shape[in_idx]
            o = out_shape[out_idx]
            st = in_strides[in_idx]
            if s == o:
                b_strides[out_idx] = st
            elif s == 1:
                b_strides[out_idx] = 0
            else:
                # Non-broadcastable, but let kernel raise later on load if used
                b_strides[out_idx] = 0
        else:
            b_strides[out_idx] = 0
    return b_strides

def _select_block_size(n_elements: int):
    if n_elements < 1024:
        return 256
    elif n_elements < 1024 * 1024:
        return 1024
    else:
        return 2048

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def smooth_l1_bw_contiguous(
    self_ptr, target_ptr, grad_out_ptr, out_ptr,
    n_elements,
    beta,           # scalar float
    scale,          # scalar float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    a = tl.load(self_ptr + offs, mask=mask)
    b = tl.load(target_ptr + offs, mask=mask)
    go = tl.load(grad_out_ptr + offs, mask=mask)

    # Compute in fp32 for numerical stability, then cast back
    x = (a - b).to(tl.float32)
    sgn = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))

    # Handle beta == 0 without division
    grad_elem = sgn
    if beta != 0.0:
        cond = tl.abs(x) < beta
        grad_elem = tl.where(cond, x / beta, sgn)

    out32 = go.to(tl.float32) * grad_elem * scale
    out = out32.to(a.dtype)
    tl.store(out_ptr + offs, out, mask=mask)

@triton.jit
def smooth_l1_bw_strided(
    self_ptr, target_ptr, grad_out_ptr, out_ptr,
    n_elements,
    # Padded shape (reversed, row-major)
    d0, d1, d2, d3, d4, d5, d6, d7,
    # Cumulative products (reversed)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # Input strides (reversed)
    self_st0, self_st1, self_st2, self_st3, self_st4, self_st5, self_st6, self_st7,
    tgt_st0,  tgt_st1,  tgt_st2,  tgt_st3,  tgt_st4,  tgt_st5,  tgt_st6,  tgt_st7,
    go_st0,   go_st1,   go_st2,   go_st3,   go_st4,   go_st5,   go_st6,   go_st7,
    out_st0,  out_st1,  out_st2,  out_st3,  out_st4,  out_st5,  out_st6,  out_st7,
    beta,           # scalar float
    beta_is_zero,   # int32 flag
    scale,          # scalar float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    r = offs.to(tl.int64)

    i0 = r // s0 % d0
    i1 = r // s1 % d1
    i2 = r // s2 % d2
    i3 = r // s3 % d3
    i4 = r // s4 % d4
    i5 = r // s5 % d5
    i6 = r // s6 % d6
    i7 = r // s7 % d7

    self_off = i0*self_st0 + i1*self_st1 + i2*self_st2 + i3*self_st3 + i4*self_st4 + i5*self_st5 + i6*self_st6 + i7*self_st7
    tgt_off  = i0*tgt_st0  + i1*tgt_st1  + i2*tgt_st2  + i3*tgt_st3  + i4*tgt_st4  + i5*tgt_st5  + i6*tgt_st6  + i7*tgt_st7
    go_off   = i0*go_st0   + i1*go_st1   + i2*go_st2   + i3*go_st3   + i4*go_st4   + i5*go_st5   + i6*go_st6   + i7*go_st7
    out_off  = i0*out_st0  + i1*out_st1  + i2*out_st2  + i3*out_st3  + i4*out_st4  + i5*out_st5  + i6*out_st6  + i7*out_st7

    a = tl.load(self_ptr + self_off, mask=mask)
    b = tl.load(target_ptr + tgt_off, mask=mask)
    go = tl.load(grad_out_ptr + go_off, mask=mask)

    x = (a - b).to(tl.float32)
    sgn = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))

    grad_elem = sgn
    if beta_is_zero == 0:
        cond = tl.abs(x) < beta
        grad_elem = tl.where(cond, x / beta, sgn)

    out32 = go.to(tl.float32) * grad_elem * scale
    out = out32.to(a.dtype)
    tl.store(out_ptr + out_off, out, mask=mask)

def _prepare_shapes_and_strides(self: torch.Tensor, target: torch.Tensor, grad_output: torch.Tensor, out: torch.Tensor):
    out_shape = list(out.shape)
    out_shape_rev = out_shape[::-1]

    # Broadcasted strides for inputs
    self_bstrides = _broadcast_strides(list(self.shape), list(self.stride()), out_shape)
    tgt_bstrides  = _broadcast_strides(list(target.shape), list(target.stride()), out_shape)
    go_bstrides   = _broadcast_strides(list(grad_output.shape), list(grad_output.stride()), out_shape)

    # Reverse for row-major indexing in kernel
    self_bstrides_rev = self_bstrides[::-1]
    tgt_bstrides_rev  = tgt_bstrides[::-1]
    go_bstrides_rev   = go_bstrides[::-1]
    out_strides_rev   = list(out.stride())[::-1]

    # Cumulative products for linear index mapping
    cum_prod_rev = [1] * len(out_shape_rev)
    for i in range(1, len(out_shape_rev)):
        cum_prod_rev[i] = cum_prod_rev[i-1] * out_shape_rev[i-1]

    # Pad
    d = _pad_to_max_dims(out_shape_rev, fill=1)
    s = _pad_to_max_dims(cum_prod_rev, fill=1)
    self_st = _pad_to_max_dims(self_bstrides_rev, fill=0)
    tgt_st  = _pad_to_max_dims(tgt_bstrides_rev, fill=0)
    go_st   = _pad_to_max_dims(go_bstrides_rev, fill=0)
    out_st  = _pad_to_max_dims(out_strides_rev, fill=0)

    return d, s, self_st, tgt_st, go_st, out_st

def _common_output_shape(self: torch.Tensor, target: torch.Tensor):
    return torch.broadcast_shapes(self.shape, target.shape)

def _compute_scale(reduction: int, out_numel: int):
    # reduction: 0 -> none, 1 -> mean, 2 -> sum (typical ATen enum)
    if reduction == 1:
        return 1.0 / max(out_numel, 1)
    else:
        return 1.0

def _launch_kernel(self: torch.Tensor, target: torch.Tensor, grad_output: torch.Tensor, out: torch.Tensor, reduction: int, beta: float):
    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Scale factor based on reduction
    scale = _compute_scale(reduction, n_elements)
    beta_is_zero = 1 if beta == 0.0 else 0

    # Fast path: all contiguous and same shape
    fastpath = (
        self.is_contiguous()
        and target.is_contiguous()
        and grad_output.is_contiguous()
        and out.is_contiguous()
        and list(self.shape) == list(target.shape) == list(out.shape) == list(grad_output.shape)
    )

    if fastpath:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        smooth_l1_bw_contiguous[grid](
            self, target, grad_output, out,
            n_elements,
            beta,
            scale,
        )
        return out

    # General path: strided with broadcasting
    d, s, self_st, tgt_st, go_st, out_st = _prepare_shapes_and_strides(self, target, grad_output, out)
    BLOCK_SIZE = _select_block_size(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    smooth_l1_bw_strided[grid](
        self, target, grad_output, out,
        n_elements,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        self_st[0], self_st[1], self_st[2], self_st[3], self_st[4], self_st[5], self_st[6], self_st[7],
        tgt_st[0],  tgt_st[1],  tgt_st[2],  tgt_st[3],  tgt_st[4],  tgt_st[5],  tgt_st[6],  tgt_st[7],
        go_st[0],   go_st[1],   go_st[2],   go_st[3],   go_st[4],   go_st[5],   go_st[6],   go_st[7],
        out_st[0],  out_st[1],  out_st[2],  out_st[3],  out_st[4],  out_st[5],  out_st[6],  out_st[7],
        beta,
        beta_is_zero,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def smooth_l1_loss_backward_grad_input(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    target: torch.Tensor,
    reduction: int,
    beta: float,
    grad_input: torch.Tensor = None,
):
    """
    aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)
    """
    # Determine output shape via broadcasting self and target
    out_shape = _common_output_shape(self, target)
    device = self.device
    dtype = self.dtype

    if grad_input is None or list(grad_input.shape) != list(out_shape) or grad_input.device != device or grad_input.dtype != dtype:
        grad_input = torch.empty(out_shape, device=device, dtype=dtype)

    # Launch kernel
    _launch_kernel(self, target, grad_output, grad_input, reduction, beta)
    return grad_input

def smooth_l1_loss_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    target: torch.Tensor,
    reduction: int,
    beta: float,
):
    """
    aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor
    """
    out_shape = _common_output_shape(self, target)
    out = torch.empty(out_shape, device=self.device, dtype=self.dtype)
    _launch_kernel(self, target, grad_output, out, reduction, beta)
    return out
