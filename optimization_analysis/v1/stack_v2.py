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


# Optimized kernel: batch process up to 4 tensors at once
@triton.jit
def stack_copy_kernel_batch4(
    out_ptr,
    in_ptr_0, in_ptr_1, in_ptr_2, in_ptr_3,
    dim_offset_0, dim_offset_1, dim_offset_2, dim_offset_3,
    numel_0, numel_1, numel_2, numel_3,
    dim_size_out,
    dim_prod_pre,      # product of dimensions before stack dim
    dim_prod_post,     # product of dimensions after stack dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch process up to 4 tensors in a single kernel launch.
    Uses 2D grid: (blocks_per_tensor, num_tensors_in_batch)
    """
    pid_x = tl.program_id(0)  # block index within tensor
    pid_y = tl.program_id(1)  # tensor index (0-3)

    # Select input pointer and metadata based on pid_y
    if pid_y == 0:
        in_ptr = in_ptr_0
        dim_offset = dim_offset_0
        numel = numel_0
    elif pid_y == 1:
        in_ptr = in_ptr_1
        dim_offset = dim_offset_1
        numel = numel_1
    elif pid_y == 2:
        in_ptr = in_ptr_2
        dim_offset = dim_offset_2
        numel = numel_2
    else:
        in_ptr = in_ptr_3
        dim_offset = dim_offset_3
        numel = numel_3

    # Compute offsets
    block_start = pid_x * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # For contiguous input: linear index = offsets
    # Decompose into (pre_idx, post_idx) for output
    pre_idx = offsets // dim_prod_post
    post_idx = offsets % dim_prod_post

    # Output index: pre_idx * (dim_size_out * dim_prod_post) + dim_offset * dim_prod_post + post_idx
    out_idx = pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx

    # Load and store
    data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, data, mask=mask)


# Fallback kernel for single tensor (when batch size < 4 remainder)
@triton.jit
def stack_copy_kernel_single(
    out_ptr,
    in_ptr,
    dim_offset,
    numel,
    dim_size_out,
    dim_prod_pre,
    dim_prod_post,
    BLOCK_SIZE: tl.constexpr,
):
    """Single tensor copy kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decompose linear index
    pre_idx = offsets // dim_prod_post
    post_idx = offsets % dim_prod_post

    # Output index
    out_idx = pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx

    # Load and store
    data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, data, mask=mask)


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
    """Optimized kernel launcher - only contiguous if needed."""
    in_shape = tuple(tensors[0].shape)
    in_rank = len(in_shape)
    out_rank = in_rank + 1
    dim = _normalize_dim(dim, out_rank)

    # Early exit for zero-sized tensors
    numel_in = tensors[0].numel()
    if numel_in == 0 or out.numel() == 0:
        return out

    # Compute dimension products for index calculation
    dim_prod_pre = 1
    for i in range(dim):
        dim_prod_pre *= in_shape[i] if i < len(in_shape) else 1

    dim_prod_post = 1
    for i in range(dim, len(in_shape)):
        dim_prod_post *= in_shape[i]

    dim_size_out = len(tensors)

    # Only make contiguous if not already
    tensors_contig = [t if t.is_contiguous() else t.contiguous() for t in tensors]

    BLOCK_SIZE = 1024
    grid_x = triton.cdiv(numel_in, BLOCK_SIZE)

    # For small number of tensors or small tensors, use single kernel
    # Batching overhead not worth it
    if len(tensors_contig) <= 2 or numel_in < 1024:
        for i, tensor in enumerate(tensors_contig):
            grid = (grid_x,)
            stack_copy_kernel_single[grid](
                out,
                tensor,
                i,
                numel_in,
                dim_size_out,
                dim_prod_pre,
                dim_prod_post,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return out

    # Process tensors in batches of 4 for larger cases
    i = 0
    while i < len(tensors_contig):
        batch = tensors_contig[i:i+4]
        batch_size = len(batch)

        if batch_size == 4:
            grid = (grid_x, 4)
            stack_copy_kernel_batch4[grid](
                out,
                batch[0], batch[1], batch[2], batch[3],
                i, i+1, i+2, i+3,
                numel_in, numel_in, numel_in, numel_in,
                dim_size_out,
                dim_prod_pre,
                dim_prod_post,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            i += 4
        else:
            for j, tensor in enumerate(batch):
                grid = (grid_x,)
                stack_copy_kernel_single[grid](
                    out,
                    tensor,
                    i + j,
                    numel_in,
                    dim_size_out,
                    dim_prod_pre,
                    dim_prod_post,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            i += batch_size

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
