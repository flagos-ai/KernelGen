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


@triton.jit
def stack_copy_kernel(
    in_ptr,                          # *Pointer* to input tensor data
    out_ptr,                         # *Pointer* to output tensor data
    in_shape_ptr,                    # int64[NIN] input shape
    in_strides_ptr,                  # int64[NIN] input strides (elements)
    out_strides_ptr,                 # int64[NOUT] output strides (elements)
    k_index,                         # int32 index along stacked dimension
    dim_out,                         # int32 stacked dimension in output
    numel_in,                        # int64 number of elements per input tensor
    in_storage_offset,               # int64
    out_storage_offset,              # int64
    BLOCK_SIZE: tl.constexpr,
    NDIM_IN: tl.constexpr,
    NDIM_OUT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    offs64 = offs.to(tl.int64)
    mask = offs64 < numel_in

    # Prepare scalar/vector parameters
    in_offset = tl.full([BLOCK_SIZE], in_storage_offset, dtype=tl.int64)
    out_offset = tl.full([BLOCK_SIZE], out_storage_offset, dtype=tl.int64)
    rem = offs64

    dim_out_vec = tl.full([BLOCK_SIZE], dim_out, dtype=tl.int32)

    # Decompose linear index into NDIM_IN coordinates and accumulate offsets on-the-fly
    for i in range(NDIM_IN - 1, -1, -1):
        s_i = tl.load(in_shape_ptr + i).to(tl.int64)
        idx_i = rem % s_i
        rem = rem // s_i

        # Input offset contribution
        stride_in_i = tl.load(in_strides_ptr + i).to(tl.int64)
        in_offset += idx_i * stride_in_i

        # Output offset contribution from non-stacked dims
        stride_out_i0 = tl.load(out_strides_ptr + i).to(tl.int64)        # j = i
        stride_out_i1 = tl.load(out_strides_ptr + (i + 1)).to(tl.int64)  # j = i + 1
        use_i = dim_out_vec > i  # if stacked dim is after i, use j=i; else j=i+1
        stride_j = tl.where(use_i, stride_out_i0, stride_out_i1)
        out_offset += idx_i * stride_j

    # Contribution from stacked dim
    k_vec = tl.full([BLOCK_SIZE], k_index, dtype=tl.int64)
    stride_stack = tl.load(out_strides_ptr + dim_out).to(tl.int64)
    out_offset += k_vec * stride_stack

    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + out_offset, x, mask=mask)


def _normalize_dim(dim: int, out_rank: int) -> int:
    if dim < 0:
        dim = dim + out_rank
    if not (0 <= dim < out_rank):
        raise IndexError(f"dim {dim} out of range for rank {out_rank}")
    return dim


def _check_tensors_compatible(tensors):
    if len(tensors) == 0:
        raise RuntimeError("stack expects a non-empty TensorList")
    ref = tensors[0]
    ref_device = ref.device
    ref_dtype = ref.dtype
    ref_shape = tuple(ref.shape)
    for t in tensors:
        if t.device != ref_device:
            raise RuntimeError("All tensors must be on the same device")
        if t.dtype != ref_dtype:
            raise RuntimeError("All tensors must have the same dtype")
        if tuple(t.shape) != ref_shape:
            raise RuntimeError("stack expects each tensor to be equal size")


def _launch_stack_kernels(tensors, dim, out):
    # Common metadata
    in_shape = tuple(tensors[0].shape)
    in_rank = len(in_shape)
    out_rank = in_rank + 1
    dim = _normalize_dim(dim, out_rank)

    # Early exit for zero-sized tensors
    numel_in = 1
    for s in in_shape:
        numel_in *= s
    if numel_in == 0 or out.numel() == 0:
        return out

    device = tensors[0].device

    # Device tensors for shapes/strides
    in_shape_t = torch.tensor(in_shape, dtype=torch.int64, device=device)
    out_strides_t = torch.tensor(out.stride(), dtype=torch.int64, device=device)

    out_storage_offset = out.storage_offset()
    grid = lambda meta: (triton.cdiv(numel_in, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = 1024

    for k, tin in enumerate(tensors):
        in_strides_t = torch.tensor(tin.stride(), dtype=torch.int64, device=device)
        in_storage_offset = tin.storage_offset()
        stack_copy_kernel[grid](
            tin,
            out,
            in_shape_t,
            in_strides_t,
            out_strides_t,
            k,
            dim,
            numel_in,
            in_storage_offset,
            out_storage_offset,
            BLOCK_SIZE=BLOCK_SIZE,
            NDIM_IN=in_rank,
            NDIM_OUT=out_rank,
        )
    return out


def stack(tensors, dim: int = 0):
    _check_tensors_compatible(tensors)
    device = tensors[0].device
    dtype = tensors[0].dtype
    in_shape = tuple(tensors[0].shape)
    out_shape = list(in_shape)
    out_shape.insert(_normalize_dim(dim, len(in_shape) + 1), len(tensors))
    out = torch.empty(out_shape, device=device, dtype=dtype)
    return _launch_stack_kernels(tensors, dim, out)


def stack_out(tensors, dim: int = 0, *, out):
    _check_tensors_compatible(tensors)
    in_shape = tuple(tensors[0].shape)
    out_rank = len(in_shape) + 1
    dim_norm = _normalize_dim(dim, out_rank)
    expected_shape = list(in_shape)
    expected_shape.insert(dim_norm, len(tensors))
    if tuple(out.shape) != tuple(expected_shape):
        raise RuntimeError(f"out tensor has incorrect shape, expected {tuple(expected_shape)} but got {tuple(out.shape)}")
    if out.device != tensors[0].device or out.dtype != tensors[0].dtype:
        raise RuntimeError("out tensor must have same device and dtype as inputs")
    return _launch_stack_kernels(tensors, dim, out)
