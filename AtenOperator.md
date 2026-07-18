<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

| 算子名称 (Op Name) | 函数签名 (Schema) | FlagGems是否支持 (FlagGems Supported) |
| :--- | :--- | :--- |
| __and__ | `aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| __and__ | `aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| __and__ | `aten::__and__.bool(bool a, bool b) -> bool` | false |
| __and__ | `aten::__and__.int(int a, int b) -> int` | false |
| __doc__ | `default` | false |
| __iand__ | `aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| __iand__ | `aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| __ilshift__ | `aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| __ilshift__ | `aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| __ior__ | `aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| __ior__ | `aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| __irshift__ | `aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| __irshift__ | `aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| __ixor__ | `aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| __ixor__ | `aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| __loader__ | `default` | false |
| __lshift__ | `aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| __lshift__ | `aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| __lshift__ | `aten::__lshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| __lshift__ | `aten::__lshift__.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| __lshift__ | `aten::__lshift__.int(int a, int b) -> int` | false |
| __name__ | `default` | false |
| __or__ | `aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| __or__ | `aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| __or__ | `aten::__or__.bool(bool a, bool b) -> bool` | false |
| __or__ | `aten::__or__.int(int a, int b) -> int` | false |
| __package__ | `default` | false |
| __rshift__ | `aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| __rshift__ | `aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| __rshift__ | `aten::__rshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| __rshift__ | `aten::__rshift__.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| __rshift__ | `aten::__rshift__.int(int a, int b) -> int` | false |
| __spec__ | `default` | false |
| __xor__ | `aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| __xor__ | `aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| __xor__ | `aten::__xor__.bool(bool a, bool b) -> bool` | false |
| __xor__ | `aten::__xor__.int(int a, int b) -> int` | false |
| _adaptive_avg_pool2d | `aten::_adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor` | false |
| _adaptive_avg_pool2d | `aten::_adaptive_avg_pool2d.out(Tensor self, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _adaptive_avg_pool2d_backward | `aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor` | false |
| _adaptive_avg_pool2d_backward | `aten::_adaptive_avg_pool2d_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _adaptive_avg_pool3d | `aten::_adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor` | false |
| _adaptive_avg_pool3d | `aten::_adaptive_avg_pool3d.out(Tensor self, SymInt[3] output_size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _adaptive_avg_pool3d_backward | `aten::_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor` | false |
| _adaptive_avg_pool3d_backward | `aten::_adaptive_avg_pool3d_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _add_relu | `aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor` | false |
| _add_relu | `aten::_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| _add_relu | `aten::_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor` | false |
| _add_relu | `aten::_add_relu.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _addmm_activation | `aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor` | false |
| _addmm_activation | `aten::_addmm_activation.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False, Tensor(a!) out) -> Tensor(a!)` | false |
| _amp_foreach_non_finite_check_and_unscale_ | `aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()` | false |
| _assert_async | `aten::_assert_async(Tensor self) -> ()` | false |
| _assert_async | `aten::_assert_async.msg(Tensor self, str assert_msg) -> ()` | false |
| _assert_scalar | `aten::_assert_scalar(Scalar self, str assert_msg) -> ()` | false |
| _assert_tensor_metadata | `aten::_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]? stride=None, ScalarType? dtype=None, *, Device? device=None, Layout? layout=None) -> ()` | false |
| _batch_norm_no_update | `aten::_batch_norm_no_update(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)` | false |
| _batch_norm_no_update | `aten::_batch_norm_no_update.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))` | false |
| _batch_norm_with_update | `aten::_batch_norm_with_update.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd, Tensor(g!) reserve) -> (Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!))` | false |
| _batch_norm_with_update | `aten::_batch_norm_with_update(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)` | false |
| _batch_norm_with_update_functional | `aten::_batch_norm_with_update_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)` | false |
| _cdist_backward | `aten::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor` | false |
| _cdist_backward | `aten::_cdist_backward.out(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _cdist_forward | `aten::_cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor` | false |
| _cdist_forward | `aten::_cdist_forward.out(Tensor x1, Tensor x2, float p, int? compute_mode, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _cholesky_solve_helper | `aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor` | false |
| _cholesky_solve_helper | `aten::_cholesky_solve_helper.out(Tensor self, Tensor A, bool upper, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _chunk_cat | `aten::_chunk_cat(Tensor[] tensors, int dim, int num_chunks) -> Tensor` | false |
| _chunk_cat | `aten::_chunk_cat.out(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _convert_weight_to_int4pack | `aten::_convert_weight_to_int4pack(Tensor self, int innerKTiles) -> Tensor` | false |
| _convert_weight_to_int4pack_for_cpu | `aten::_convert_weight_to_int4pack_for_cpu(Tensor self, int innerKTiles) -> Tensor` | false |
| _cslt_sparse_mm | `aten::_cslt_sparse_mm(Tensor compressed_A, Tensor dense_B, Tensor? bias=None, Tensor? alpha=None, ScalarType? out_dtype=None, bool transpose_result=False, int alg_id=0, int split_k=1, int split_k_mode=-1) -> Tensor` | false |
| _dir | `default` | false |
| _dyn_quant_matmul_4bit | `aten::_dyn_quant_matmul_4bit(Tensor inp, Tensor packed_weights, int block_size, int in_features, int out_features) -> Tensor` | false |
| _dyn_quant_pack_4bit_weight | `aten::_dyn_quant_pack_4bit_weight(Tensor weights, Tensor scales_zeros, Tensor? bias, int block_size, int in_features, int out_features) -> Tensor` | false |
| _efficient_attention_backward | `aten::_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt max_seqlen_q, SymInt max_seqlen_k, Tensor logsumexp, float dropout_p, Tensor philox_seed, Tensor philox_offset, int custom_mask_type, bool bias_requires_grad, *, float? scale=None, int? num_splits_key=None, int? window_size=None, bool shared_storage_dqdkdv=False) -> (Tensor, Tensor, Tensor, Tensor)` | false |
| _efficient_attention_forward | `aten::_efficient_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt? max_seqlen_q, SymInt? max_seqlen_k, float dropout_p, int custom_mask_type, bool compute_log_sumexp=False, *, float? scale=None, Tensor? seqlen_k=None, int? window_size=None) -> (Tensor output, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, SymInt max_seqlen_batch_q, SymInt max_seqlen_batch_k)` | false |
| _embedding_bag | `aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)` | false |
| _embedding_bag | `aten::_embedding_bag.out(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))` | false |
| _embedding_bag_backward | `aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor` | false |
| _embedding_bag_dense_backward | `aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor` | false |
| _embedding_bag_dense_backward | `aten::_embedding_bag_dense_backward.out(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _embedding_bag_forward_only | `aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)` | false |
| _embedding_bag_forward_only | `aten::_embedding_bag_forward_only.out(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))` | false |
| _embedding_bag_per_sample_weights_backward | `aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor` | false |
| _embedding_bag_per_sample_weights_backward | `aten::_embedding_bag_per_sample_weights_backward.out(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _euclidean_dist | `aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor` | false |
| _euclidean_dist | `aten::_euclidean_dist.out(Tensor x1, Tensor x2, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _fft_c2c | `aten::_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor` | false |
| _fft_c2c | `aten::_fft_c2c.out(Tensor self, SymInt[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _fft_c2r | `aten::_fft_c2r(Tensor self, int[] dim, int normalization, SymInt last_dim_size) -> Tensor` | false |
| _fft_c2r | `aten::_fft_c2r.out(Tensor self, int[] dim, int normalization, SymInt last_dim_size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _fft_r2c | `aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor` | false |
| _fft_r2c | `aten::_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _flash_attention_backward | `aten::_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor rng_state, Tensor unused, *, float? scale=None, SymInt? window_size_left=None, SymInt? window_size_right=None) -> (Tensor, Tensor, Tensor)` | false |
| _flash_attention_forward | `aten::_flash_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, bool return_debug_mask, *, float? scale=None, SymInt? window_size_left=None, SymInt? window_size_right=None, Tensor? seqused_k=None, Tensor? alibi_slopes=None) -> (Tensor output, Tensor softmax_logsumexp, Tensor rng_state, Tensor unused, Tensor debug_attn_mask)` | false |
| _functional_assert_async | `aten::_functional_assert_async.msg(Tensor self, str assert_msg, Tensor dep_token) -> Tensor` | false |
| _functional_sym_constrain_range | `aten::_functional_sym_constrain_range(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor` | false |
| _functional_sym_constrain_range_for_size | `aten::_functional_sym_constrain_range_for_size(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor` | false |
| _fused_adam | `aten::_fused_adam(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)` | false |
| _fused_adam | `aten::_fused_adam.out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()` | false |
| _fused_adam | `aten::_fused_adam.tensor_lr(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)` | false |
| _fused_adam | `aten::_fused_adam.tensor_lr_out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()` | false |
| _fused_adam_ | `aten::_fused_adam_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()` | false |
| _fused_adam_ | `aten::_fused_adam_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()` | false |
| _fused_adamw_ | `aten::_fused_adamw_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()` | false |
| _fused_adamw_ | `aten::_fused_adamw_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()` | false |
| _fused_dropout | `aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)` | false |
| _fused_dropout | `aten::_fused_dropout.out(Tensor self, float p, Generator? generator=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| _fused_moving_avg_obs_fq_helper | `aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)` | false |
| _fused_moving_avg_obs_fq_helper | `aten::_fused_moving_avg_obs_fq_helper.out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))` | false |
| _fused_rms_norm_backward | `aten::_fused_rms_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor rstd, Tensor? weight, bool[2] output_mask) -> (Tensor, Tensor)` | false |
| _grouped_mm | `aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor` | false |
| _int_mm | `aten::_int_mm(Tensor self, Tensor mat2) -> Tensor` | false |
| _int_mm | `aten::_int_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _jagged_to_padded_dense_forward | `aten::_jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value=0.) -> Tensor` | false |
| _linalg_det | `aten::_linalg_det(Tensor A) -> (Tensor result, Tensor LU, Tensor pivots)` | false |
| _linalg_det | `aten::_linalg_det.result(Tensor A, *, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots)` | false |
| _linalg_eigh | `aten::_linalg_eigh(Tensor A, str UPLO="L", bool compute_v=True) -> (Tensor eigenvalues, Tensor eigenvectors)` | false |
| _linalg_eigh | `aten::_linalg_eigh.eigenvalues(Tensor A, str UPLO="L", bool compute_v=True, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)` | false |
| _linalg_eigvals | `aten::_linalg_eigvals(Tensor self) -> Tensor` | false |
| _linalg_slogdet | `aten::_linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)` | false |
| _linalg_slogdet | `aten::_linalg_slogdet.sign(Tensor A, *, Tensor(a!) sign, Tensor(b!) logabsdet, Tensor(c!) LU, Tensor(d!) pivots) -> (Tensor(a!) sign, Tensor(b!) logabsdet, Tensor(c!) LU, Tensor(d!) pivots)` | false |
| _linalg_solve_ex | `aten::_linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor LU, Tensor pivots, Tensor info)` | false |
| _linalg_solve_ex | `aten::_linalg_solve_ex.result(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots, Tensor(d!) info) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots, Tensor(d!) info)` | false |
| _linalg_svd | `aten::_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)` | false |
| _linalg_svd | `aten::_linalg_svd.U(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)` | false |
| _list_to_tensor | `aten::_list_to_tensor(int[] self) -> Tensor` | false |
| _local_scalar_dense | `aten::_local_scalar_dense(Tensor self) -> Scalar` | false |
| _log_softmax | `aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor` | false |
| _log_softmax | `aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _log_softmax_backward_data | `aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor` | false |
| _log_softmax_backward_data | `aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _make_dep_token | `aten::_make_dep_token(*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| _masked_scale | `aten::_masked_scale.out(Tensor self, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _masked_scale | `aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor` | false |
| _native_batch_norm_legit | `aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)` | false |
| _native_batch_norm_legit | `aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)` | false |
| _native_batch_norm_legit | `aten::_native_batch_norm_legit.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd) -> (Tensor(d!), Tensor(e!), Tensor(f!))` | false |
| _native_batch_norm_legit | `aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| _native_batch_norm_legit_functional | `aten::_native_batch_norm_legit_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)` | false |
| _native_batch_norm_legit_no_training | `aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)` | false |
| _native_batch_norm_legit_no_training | `aten::_native_batch_norm_legit_no_training.out(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| _nested_tensor_from_tensor_list | `aten::_nested_tensor_from_tensor_list(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| _nested_tensor_from_tensor_list | `aten::_nested_tensor_from_tensor_list.out(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _nested_view_from_buffer | `aten::_nested_view_from_buffer(Tensor(a) self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor(a)` | false |
| _nested_view_from_buffer_copy | `aten::_nested_view_from_buffer_copy(Tensor self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor` | false |
| _nested_view_from_buffer_copy | `aten::_nested_view_from_buffer_copy.out(Tensor self, Tensor nested_size, Tensor nested_strides, Tensor offsets, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _pack_padded_sequence | `aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)` | false |
| _pack_padded_sequence | `aten::_pack_padded_sequence.out(Tensor input, Tensor lengths, bool batch_first, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| _padded_dense_to_jagged_forward | `aten::_padded_dense_to_jagged_forward(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> Tensor` | false |
| _pdist_backward | `aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor` | false |
| _pdist_backward | `aten::_pdist_backward.out(Tensor grad, Tensor self, float p, Tensor pdist, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _pdist_forward | `aten::_pdist_forward(Tensor self, float p=2.) -> Tensor` | false |
| _pdist_forward | `aten::_pdist_forward.out(Tensor self, float p=2., *, Tensor(a!) out) -> Tensor(a!)` | false |
| _pin_memory | `aten::_pin_memory(Tensor self, Device? device=None) -> Tensor` | false |
| _pin_memory | `aten::_pin_memory.out(Tensor self, Device? device=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _prelu_kernel | `aten::_prelu_kernel(Tensor self, Tensor weight) -> Tensor` | false |
| _prelu_kernel_backward | `aten::_prelu_kernel_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)` | false |
| _print | `aten::_print(str s) -> ()` | false |
| _reshape_alias | `aten::_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)` | false |
| _resize_output | `aten::_resize_output(Tensor self, SymInt[] size, Device device) -> Tensor` | false |
| _resize_output | `aten::_resize_output.out(Tensor self, SymInt[] size, Device device, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _resize_output_ | `aten::_resize_output_(Tensor(a!) self, SymInt[] size, Device device) -> Tensor(a!)` | false |
| _safe_softmax | `aten::_safe_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor` | false |
| _scaled_dot_product_attention_math_for_mps | `aten::_scaled_dot_product_attention_math_for_mps(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None) -> (Tensor, Tensor)` | false |
| _scaled_dot_product_efficient_attention | `aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0., bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)` | false |
| _scaled_dot_product_efficient_attention_backward | `aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor attn_bias, Tensor out, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, float dropout_p, bool[4] grad_input_mask, bool is_causal=False, *, float? scale=None) -> (Tensor, Tensor, Tensor, Tensor)` | false |
| _scaled_dot_product_flash_attention | `aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor rng_state, Tensor unused, Tensor debug_attn_mask)` | false |
| _scaled_dot_product_flash_attention_backward | `aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)` | false |
| _scaled_dot_product_flash_attention_for_cpu | `aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)` | false |
| _scaled_dot_product_flash_attention_for_cpu_backward | `aten::_scaled_dot_product_flash_attention_for_cpu_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, float dropout_p, bool is_causal, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)` | false |
| _scaled_dot_product_fused_attention_overrideable | `aten::_scaled_dot_product_fused_attention_overrideable(Tensor query, Tensor key, Tensor value, Tensor? attn_bias=None, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)` | false |
| _scaled_grouped_mm | `aten::_scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor` | false |
| _scaled_mm | `aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor` | false |
| _scaled_mm | `aten::_scaled_mm.out(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _segment_reduce_backward | `aten::_segment_reduce_backward(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, Tensor? offsets=None, int axis=0, Scalar? initial=None) -> Tensor` | false |
| _segment_reduce_backward | `aten::_segment_reduce_backward.out(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, Tensor? offsets=None, int axis=0, Scalar? initial=None, Tensor(a!) out) -> Tensor(a!)` | false |
| _softmax | `aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor` | false |
| _softmax | `aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _softmax_backward_data | `aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor` | false |
| _softmax_backward_data | `aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| _sparse_coo_tensor_with_dims_and_tensors | `aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? is_coalesced=None) -> Tensor` | false |
| _sparse_coo_tensor_with_dims_and_tensors | `aten::_sparse_coo_tensor_with_dims_and_tensors.out(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, bool? is_coalesced=None, Tensor(a!) out) -> Tensor(a!)` | false |
| _sparse_semi_structured_addmm | `aten::_sparse_semi_structured_addmm(Tensor input, Tensor mat1, Tensor mat1_meta, Tensor mat2, *, Scalar alpha=1, Scalar beta=1, ScalarType? out_dtype=None) -> Tensor` | false |
| _sparse_semi_structured_linear | `aten::_sparse_semi_structured_linear(Tensor input, Tensor weight, Tensor meta, *, Tensor? bias=None, str? activation=None, ScalarType? out_dtype=None) -> Tensor` | false |
| _sparse_semi_structured_mm | `aten::_sparse_semi_structured_mm(Tensor mat1, Tensor mat1_meta, Tensor mat2, *, ScalarType? out_dtype=None) -> Tensor` | false |
| _thnn_fused_lstm_cell | `aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)` | false |
| _thnn_fused_lstm_cell | `aten::_thnn_fused_lstm_cell.out(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| _thnn_fused_lstm_cell_backward_impl | `aten::_thnn_fused_lstm_cell_backward_impl.out(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| _thnn_fused_lstm_cell_backward_impl | `aten::_thnn_fused_lstm_cell_backward_impl(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor)` | false |
| _to_copy | `aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor` | false |
| _to_copy | `aten::_to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| _unique2 | `aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)` | false |
| _unique2 | `aten::_unique2.out(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| _unsafe_index | `aten::_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor` | false |
| _unsafe_index | `aten::_unsafe_index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor` | false |
| _unsafe_index_put | `aten::_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor` | false |
| _unsafe_index_put | `aten::_unsafe_index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor` | false |
| _unsafe_masked_index | `aten::_unsafe_masked_index(Tensor self, Tensor mask, Tensor?[] indices, Scalar fill) -> Tensor` | false |
| _unsafe_masked_index_put_accumulate | `aten::_unsafe_masked_index_put_accumulate(Tensor self, Tensor mask, Tensor?[] indices, Tensor values) -> Tensor` | false |
| _unsafe_view | `aten::_unsafe_view(Tensor self, SymInt[] size) -> Tensor` | false |
| _unsafe_view | `aten::_unsafe_view.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _upsample_bicubic2d_aa | `aten::_upsample_bicubic2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| _upsample_bicubic2d_aa | `aten::_upsample_bicubic2d_aa.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor` | false |
| _upsample_bicubic2d_aa | `aten::_upsample_bicubic2d_aa.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _upsample_bilinear2d_aa | `aten::_upsample_bilinear2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| _upsample_bilinear2d_aa | `aten::_upsample_bilinear2d_aa.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor` | false |
| _upsample_bilinear2d_aa | `aten::_upsample_bilinear2d_aa.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _upsample_bilinear2d_aa_backward | `aten::_upsample_bilinear2d_aa_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| _upsample_bilinear2d_aa_backward | `aten::_upsample_bilinear2d_aa_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| _upsample_nearest_exact1d | `aten::_upsample_nearest_exact1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor` | false |
| _upsample_nearest_exact1d | `aten::_upsample_nearest_exact1d.out(Tensor self, SymInt[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _upsample_nearest_exact1d | `aten::_upsample_nearest_exact1d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor` | false |
| _upsample_nearest_exact2d | `aten::_upsample_nearest_exact2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| _upsample_nearest_exact2d | `aten::_upsample_nearest_exact2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _upsample_nearest_exact2d | `aten::_upsample_nearest_exact2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor` | false |
| _upsample_nearest_exact2d_backward | `aten::_upsample_nearest_exact2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| _upsample_nearest_exact2d_backward | `aten::_upsample_nearest_exact2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| _upsample_nearest_exact3d | `aten::_upsample_nearest_exact3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| _upsample_nearest_exact3d | `aten::_upsample_nearest_exact3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| _upsample_nearest_exact3d | `aten::_upsample_nearest_exact3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor` | false |
| _weight_int4pack_mm | `aten::_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros) -> Tensor` | false |
| _weight_int4pack_mm_for_cpu | `aten::_weight_int4pack_mm_for_cpu(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros) -> Tensor` | false |
| _weight_int4pack_mm_with_scales_and_zeros | `aten::_weight_int4pack_mm_with_scales_and_zeros(Tensor self, Tensor mat2, int qGroupSize, Tensor qScale, Tensor qZeros) -> Tensor` | false |
| _weight_int8pack_mm | `aten::_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales) -> Tensor` | false |
| _weight_norm_interface | `aten::_weight_norm_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)` | false |
| _weight_norm_interface | `aten::_weight_norm_interface.out(Tensor v, Tensor g, int dim=0, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| abs | `aten::abs(Tensor self) -> Tensor` | false |
| abs | `aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| abs_ | `aten::abs_(Tensor(a!) self) -> Tensor(a!)` | false |
| absolute | `aten::absolute(Tensor self) -> Tensor` | false |
| absolute | `aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| absolute_ | `aten::absolute_(Tensor(a!) self) -> Tensor(a!)` | false |
| acos | `aten::acos(Tensor self) -> Tensor` | false |
| acos | `aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| acos | `aten::acos.int(int a) -> float` | false |
| acos | `aten::acos.float(float a) -> float` | false |
| acos | `aten::acos.complex(complex a) -> complex` | false |
| acos | `aten::acos.Scalar(Scalar a) -> Scalar` | false |
| acos_ | `aten::acos_(Tensor(a!) self) -> Tensor(a!)` | false |
| acosh | `aten::acosh(Tensor self) -> Tensor` | false |
| acosh | `aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| acosh | `aten::acosh.int(int a) -> float` | false |
| acosh | `aten::acosh.float(float a) -> float` | false |
| acosh | `aten::acosh.complex(complex a) -> complex` | false |
| acosh | `aten::acosh.Scalar(Scalar a) -> Scalar` | false |
| acosh_ | `aten::acosh_(Tensor(a!) self) -> Tensor(a!)` | false |
| adaptive_max_pool2d | `aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)` | false |
| adaptive_max_pool2d | `aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))` | false |
| adaptive_max_pool2d_backward | `aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor` | false |
| adaptive_max_pool2d_backward | `aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| adaptive_max_pool3d | `aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)` | false |
| adaptive_max_pool3d | `aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))` | false |
| adaptive_max_pool3d_backward | `aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor` | false |
| adaptive_max_pool3d_backward | `aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| add | `aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor` | false |
| add | `aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor` | false |
| add | `aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| add | `aten::add.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| add | `aten::add.t(t[] a, t[] b) -> t[]` | false |
| add | `aten::add.str(str a, str b) -> str` | false |
| add | `aten::add.int(int a, int b) -> int` | false |
| add | `aten::add.complex(complex a, complex b) -> complex` | false |
| add | `aten::add.float(float a, float b) -> float` | false |
| add | `aten::add.int_complex(int a, complex b) -> complex` | false |
| add | `aten::add.complex_int(complex a, int b) -> complex` | false |
| add | `aten::add.float_complex(float a, complex b) -> complex` | false |
| add | `aten::add.complex_float(complex a, float b) -> complex` | false |
| add | `aten::add.int_float(int a, float b) -> float` | false |
| add | `aten::add.float_int(float a, int b) -> float` | false |
| add | `aten::add(Scalar a, Scalar b) -> Scalar` | false |
| add_ | `aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)` | false |
| add_ | `aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)` | false |
| add_ | `aten::add_.t(t[](a!) self, t[] b) -> t[]` | false |
| addbmm | `aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor` | false |
| addbmm | `aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| addbmm_ | `aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)` | false |
| addcdiv | `aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor` | false |
| addcdiv | `aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)` | false |
| addcdiv_ | `aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)` | false |
| addcmul | `aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor` | false |
| addcmul | `aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)` | false |
| addcmul_ | `aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)` | false |
| addmm | `aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor` | false |
| addmm | `aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| addmm | `aten::addmm.dtype_out(Tensor self, Tensor mat1, Tensor mat2, ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| addmm | `aten::addmm.dtype(Tensor self, Tensor mat1, Tensor mat2, ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1) -> Tensor` | false |
| addmm_ | `aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)` | false |
| addmv | `aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor` | false |
| addmv | `aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| addmv_ | `aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)` | false |
| addr | `aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor` | false |
| addr | `aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| affine_grid_generator | `aten::affine_grid_generator(Tensor theta, SymInt[] size, bool align_corners) -> Tensor` | false |
| affine_grid_generator | `aten::affine_grid_generator.out(Tensor theta, SymInt[] size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)` | false |
| alias | `aten::alias(Tensor(a) self) -> Tensor(a)` | false |
| alias_copy | `aten::alias_copy(Tensor self) -> Tensor` | false |
| alias_copy | `aten::alias_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| all | `aten::all(Tensor self) -> Tensor` | false |
| all | `aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor` | false |
| all | `aten::all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor` | false |
| all | `aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| all | `aten::all.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| all | `aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| all | `aten::all.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor` | false |
| all | `aten::all.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| all | `aten::all.int(int[] self) -> bool` | false |
| all | `aten::all.float(float[] self) -> bool` | false |
| all | `aten::all.bool(bool[] self) -> bool` | false |
| alpha_dropout | `aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor` | false |
| amax | `aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor` | false |
| amax | `aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| amin | `aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor` | false |
| amin | `aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| aminmax | `aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)` | false |
| aminmax | `aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)` | false |
| angle | `aten::angle(Tensor self) -> Tensor` | false |
| angle | `aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| angle | `aten::angle.int(int a) -> float` | false |
| angle | `aten::angle.float(float a) -> float` | false |
| angle | `aten::angle.complex(complex a) -> float` | false |
| angle | `aten::angle.Scalar(Scalar a) -> Scalar` | false |
| any | `aten::any(Tensor self) -> Tensor` | false |
| any | `aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor` | false |
| any | `aten::any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor` | false |
| any | `aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| any | `aten::any.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| any | `aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| any | `aten::any.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor` | false |
| any | `aten::any.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| any | `aten::any.str(str[] self) -> bool` | false |
| any | `aten::any.int(int[] self) -> bool` | false |
| any | `aten::any.float(float[] self) -> bool` | false |
| any | `aten::any.bool(bool[] self) -> bool` | false |
| arange | `aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| arange | `aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| arange | `aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| arange | `aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arange | `aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arccos | `aten::arccos(Tensor self) -> Tensor` | false |
| arccos | `aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arccos_ | `aten::arccos_(Tensor(a!) self) -> Tensor(a!)` | false |
| arccosh | `aten::arccosh(Tensor self) -> Tensor` | false |
| arccosh | `aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arccosh_ | `aten::arccosh_(Tensor(a!) self) -> Tensor(a!)` | false |
| arcsin | `aten::arcsin(Tensor self) -> Tensor` | false |
| arcsin | `aten::arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arcsin_ | `aten::arcsin_(Tensor(a!) self) -> Tensor(a!)` | false |
| arcsinh | `aten::arcsinh(Tensor self) -> Tensor` | false |
| arcsinh | `aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arcsinh_ | `aten::arcsinh_(Tensor(a!) self) -> Tensor(a!)` | false |
| arctan | `aten::arctan(Tensor self) -> Tensor` | false |
| arctan | `aten::arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arctan2 | `aten::arctan2(Tensor self, Tensor other) -> Tensor` | false |
| arctan2 | `aten::arctan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arctan2_ | `aten::arctan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| arctan_ | `aten::arctan_(Tensor(a!) self) -> Tensor(a!)` | false |
| arctanh | `aten::arctanh(Tensor self) -> Tensor` | false |
| arctanh | `aten::arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| arctanh_ | `aten::arctanh_(Tensor(a!) self) -> Tensor(a!)` | false |
| argmax | `aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor` | false |
| argmax | `aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| argmin | `aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor` | false |
| argmin | `aten::argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| as_strided | `aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)` | false |
| as_strided_ | `aten::as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)` | false |
| as_strided_copy | `aten::as_strided_copy(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor` | false |
| as_strided_copy | `aten::as_strided_copy.out(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| as_strided_scatter | `aten::as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor` | false |
| as_strided_scatter | `aten::as_strided_scatter.out(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| asin | `aten::asin(Tensor self) -> Tensor` | false |
| asin | `aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| asin | `aten::asin.int(int a) -> float` | false |
| asin | `aten::asin.float(float a) -> float` | false |
| asin | `aten::asin.complex(complex a) -> complex` | false |
| asin | `aten::asin.Scalar(Scalar a) -> Scalar` | false |
| asin_ | `aten::asin_(Tensor(a!) self) -> Tensor(a!)` | false |
| asinh | `aten::asinh(Tensor self) -> Tensor` | false |
| asinh | `aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| asinh | `aten::asinh.int(int a) -> float` | false |
| asinh | `aten::asinh.float(float a) -> float` | false |
| asinh | `aten::asinh.complex(complex a) -> complex` | false |
| asinh | `aten::asinh.Scalar(Scalar a) -> Scalar` | false |
| asinh_ | `aten::asinh_(Tensor(a!) self) -> Tensor(a!)` | false |
| atan | `aten::atan(Tensor self) -> Tensor` | false |
| atan | `aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| atan | `aten::atan.int(int a) -> float` | false |
| atan | `aten::atan.float(float a) -> float` | false |
| atan | `aten::atan.complex(complex a) -> complex` | false |
| atan | `aten::atan.Scalar(Scalar a) -> Scalar` | false |
| atan2 | `aten::atan2(Tensor self, Tensor other) -> Tensor` | false |
| atan2 | `aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| atan2 | `aten::atan2.int(int a, int b) -> float` | false |
| atan2 | `aten::atan2.float(float a, float b) -> float` | false |
| atan2 | `aten::atan2.int_float(int a, float b) -> float` | false |
| atan2 | `aten::atan2.float_int(float a, int b) -> float` | false |
| atan2 | `aten::atan2.Scalar_Scalar(Scalar a, Scalar b) -> float` | false |
| atan2_ | `aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| atan_ | `aten::atan_(Tensor(a!) self) -> Tensor(a!)` | false |
| atanh | `aten::atanh(Tensor self) -> Tensor` | false |
| atanh | `aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| atanh | `aten::atanh.int(int a) -> float` | false |
| atanh | `aten::atanh.float(float a) -> float` | false |
| atanh | `aten::atanh.complex(complex a) -> complex` | false |
| atanh | `aten::atanh.Scalar(Scalar a) -> Scalar` | false |
| atanh_ | `aten::atanh_(Tensor(a!) self) -> Tensor(a!)` | false |
| avg_pool2d | `aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor` | false |
| avg_pool2d | `aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| avg_pool2d_backward | `aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor` | false |
| avg_pool2d_backward | `aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| avg_pool3d | `aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor` | false |
| avg_pool3d | `aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| avg_pool3d_backward | `aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor` | false |
| avg_pool3d_backward | `aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| baddbmm | `aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor` | false |
| baddbmm | `aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| baddbmm | `aten::baddbmm.dtype_out(Tensor self, Tensor batch1, Tensor batch2, ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| baddbmm | `aten::baddbmm.dtype(Tensor self, Tensor batch1, Tensor batch2, ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1) -> Tensor` | false |
| baddbmm_ | `aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)` | false |
| batch_norm_backward | `aten::batch_norm_backward(Tensor grad_out, Tensor input, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, bool update, float eps, bool[3] output_mask, Tensor reserve) -> (Tensor, Tensor, Tensor)` | false |
| bernoulli | `aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor` | false |
| bernoulli | `aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| bernoulli | `aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor` | false |
| bernoulli | `aten::bernoulli.Tensor(Tensor self, Tensor p, *, Generator? generator=None) -> Tensor` | false |
| bernoulli | `aten::bernoulli.Tensor_out(Tensor self, Tensor p, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| bernoulli | `aten::bernoulli.float_out(Tensor self, float p=0.5, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| bernoulli_ | `aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)` | false |
| bernoulli_ | `aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)` | false |
| binary_cross_entropy | `aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> Tensor` | false |
| binary_cross_entropy | `aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| binary_cross_entropy_backward | `aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> Tensor` | false |
| binary_cross_entropy_backward | `aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=1, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| binary_cross_entropy_with_logits | `aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=1) -> Tensor` | false |
| binary_cross_entropy_with_logits | `aten::binary_cross_entropy_with_logits.out(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bincount | `aten::bincount(Tensor self, Tensor? weights=None, SymInt minlength=0) -> Tensor` | false |
| bincount | `aten::bincount.out(Tensor self, Tensor? weights=None, SymInt minlength=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_and | `aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| bitwise_and | `aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| bitwise_and | `aten::bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` | false |
| bitwise_and | `aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_and | `aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_and | `aten::bitwise_and.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_and_ | `aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| bitwise_and_ | `aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| bitwise_left_shift | `aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| bitwise_left_shift | `aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor` | false |
| bitwise_left_shift | `aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` | false |
| bitwise_left_shift | `aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_left_shift | `aten::bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_left_shift | `aten::bitwise_left_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_left_shift_ | `aten::bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| bitwise_left_shift_ | `aten::bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| bitwise_not | `aten::bitwise_not(Tensor self) -> Tensor` | false |
| bitwise_not | `aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_not_ | `aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)` | false |
| bitwise_or | `aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| bitwise_or | `aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| bitwise_or | `aten::bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` | false |
| bitwise_or | `aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_or | `aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_or | `aten::bitwise_or.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_or_ | `aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| bitwise_or_ | `aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| bitwise_right_shift | `aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| bitwise_right_shift | `aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor` | false |
| bitwise_right_shift | `aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` | false |
| bitwise_right_shift | `aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_right_shift | `aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_right_shift | `aten::bitwise_right_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_right_shift_ | `aten::bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| bitwise_right_shift_ | `aten::bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| bitwise_xor | `aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| bitwise_xor | `aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| bitwise_xor | `aten::bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` | false |
| bitwise_xor | `aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_xor | `aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_xor | `aten::bitwise_xor.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bitwise_xor_ | `aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| bitwise_xor_ | `aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| block_diag | `aten::block_diag(Tensor[] tensors) -> Tensor` | false |
| block_diag | `aten::block_diag.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bmm | `aten::bmm(Tensor self, Tensor mat2) -> Tensor` | false |
| bmm | `aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bmm | `aten::bmm.dtype_out(Tensor self, Tensor mat2, ScalarType out_dtype, *, Tensor(a!) out) -> Tensor(a!)` | false |
| bmm | `aten::bmm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor` | false |
| broadcast_tensors | `aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]` | false |
| bucketize | `aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor` | false |
| bucketize | `aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor` | false |
| bucketize | `aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)` | false |
| bucketize | `aten::bucketize.Scalar_out(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)` | false |
| cat | `aten::cat(Tensor[] tensors, int dim=0) -> Tensor` | false |
| cat | `aten::cat.names(Tensor[] tensors, str dim) -> Tensor` | false |
| cat | `aten::cat.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cat | `aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cauchy | `aten::cauchy(Tensor self, float median=0., float sigma=1., *, Generator? generator=None) -> Tensor` | false |
| cauchy | `aten::cauchy.out(Tensor self, float median=0., float sigma=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| cauchy_ | `aten::cauchy_(Tensor(a!) self, float median=0., float sigma=1., *, Generator? generator=None) -> Tensor(a!)` | false |
| ceil | `aten::ceil(Tensor self) -> Tensor` | false |
| ceil | `aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| ceil | `aten::ceil.int(int a) -> int` | false |
| ceil | `aten::ceil.float(float a) -> int` | false |
| ceil | `aten::ceil.Scalar(Scalar a) -> Scalar` | false |
| ceil_ | `aten::ceil_(Tensor(a!) self) -> Tensor(a!)` | false |
| celu | `aten::celu(Tensor self, Scalar alpha=1.) -> Tensor` | false |
| celu | `aten::celu.out(Tensor self, Scalar alpha=1., *, Tensor(a!) out) -> Tensor(a!)` | false |
| celu_ | `aten::celu_(Tensor(a!) self, Scalar alpha=1.) -> Tensor(a!)` | false |
| channel_shuffle | `aten::channel_shuffle(Tensor self, SymInt groups) -> Tensor` | false |
| channel_shuffle | `aten::channel_shuffle.out(Tensor self, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cholesky | `aten::cholesky(Tensor self, bool upper=False) -> Tensor` | false |
| cholesky | `aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cholesky_inverse | `aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor` | false |
| cholesky_inverse | `aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cholesky_solve | `aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor` | false |
| cholesky_solve | `aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clamp | `aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor` | false |
| clamp | `aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor` | false |
| clamp | `aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clamp | `aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clamp_ | `aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)` | false |
| clamp_ | `aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)` | false |
| clamp_max | `aten::clamp_max(Tensor self, Scalar max) -> Tensor` | false |
| clamp_max | `aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor` | false |
| clamp_max | `aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clamp_max | `aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clamp_max_ | `aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)` | false |
| clamp_max_ | `aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)` | false |
| clamp_min | `aten::clamp_min(Tensor self, Scalar min) -> Tensor` | false |
| clamp_min | `aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor` | false |
| clamp_min | `aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clamp_min | `aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clamp_min_ | `aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)` | false |
| clamp_min_ | `aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)` | false |
| clip | `aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor` | false |
| clip | `aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor` | false |
| clip | `aten::clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clip | `aten::clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| clip_ | `aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)` | false |
| clip_ | `aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)` | false |
| clone | `aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor` | false |
| clone | `aten::clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| col2im | `aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor` | false |
| col2im | `aten::col2im.out(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)` | false |
| complex | `aten::complex(Tensor real, Tensor imag) -> Tensor` | false |
| complex | `aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)` | false |
| conj | `aten::conj(Tensor(a) self) -> Tensor(a)` | false |
| conj_physical | `aten::conj_physical(Tensor self) -> Tensor` | false |
| conj_physical | `aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| conj_physical_ | `aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)` | false |
| constant_pad_nd | `aten::constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor` | false |
| constant_pad_nd | `aten::constant_pad_nd.out(Tensor self, SymInt[] pad, Scalar value=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| conv2d | `aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], SymInt[2] dilation=[1, 1], SymInt groups=1) -> Tensor` | false |
| conv2d | `aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1, 1], str padding="valid", SymInt[2] dilation=[1, 1], SymInt groups=1) -> Tensor` | false |
| convolution | `aten::convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor` | false |
| convolution | `aten::convolution.out(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)` | false |
| convolution_backward | `aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)` | false |
| convolution_backward | `aten::convolution_backward.out(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| copy | `aten::copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor` | false |
| copy | `aten::copy.out(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| copy | `aten::copy.t(t[](a) self) -> t[]` | false |
| copy | `aten::copy.Dict_str(Dict(str, t)(a) self) -> Dict(str, t)` | false |
| copy | `aten::copy.Dict_int(Dict(int, t)(a) self) -> Dict(int, t)` | false |
| copy | `aten::copy.Dict_bool(Dict(bool, t)(a) self) -> Dict(bool, t)` | false |
| copy | `aten::copy.Dict_float(Dict(float, t)(a) self) -> Dict(float, t)` | false |
| copy | `aten::copy.Dict_complex(Dict(complex, t)(a) self) -> Dict(complex, t)` | false |
| copy | `aten::copy.Dict_Tensor(Dict(Tensor, t)(a) self) -> Dict(Tensor, t)` | false |
| copy_ | `aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)` | false |
| copy_ | `aten::copy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| copy_ | `aten::copy_.int(Tensor(a!) self, int other) -> Tensor(a!)` | false |
| copy_ | `aten::copy_.float(Tensor(a!) self, float other) -> Tensor(a!)` | false |
| copysign | `aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| copysign | `aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| copysign | `aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| copysign | `aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| copysign | `aten::copysign.int(int a, int b) -> float` | false |
| copysign | `aten::copysign.float(float a, float b) -> float` | false |
| copysign | `aten::copysign.int_float(int a, float b) -> float` | false |
| copysign | `aten::copysign.float_int(float a, int b) -> float` | false |
| copysign | `aten::copysign(Scalar a, Scalar b) -> float` | false |
| copysign_ | `aten::copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| copysign_ | `aten::copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| cos | `aten::cos(Tensor self) -> Tensor` | false |
| cos | `aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cos | `aten::cos.int(int a) -> float` | false |
| cos | `aten::cos.float(float a) -> float` | false |
| cos | `aten::cos.complex(complex a) -> complex` | false |
| cos | `aten::cos.Scalar(Scalar a) -> Scalar` | false |
| cos_ | `aten::cos_(Tensor(a!) self) -> Tensor(a!)` | false |
| cosh | `aten::cosh(Tensor self) -> Tensor` | false |
| cosh | `aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cosh | `aten::cosh.int(int a) -> float` | false |
| cosh | `aten::cosh.float(float a) -> float` | false |
| cosh | `aten::cosh.complex(complex a) -> complex` | false |
| cosh | `aten::cosh.Scalar(Scalar a) -> Scalar` | false |
| cosh_ | `aten::cosh_(Tensor(a!) self) -> Tensor(a!)` | false |
| count_nonzero | `aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor` | false |
| count_nonzero | `aten::count_nonzero.dim_IntList_out(Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!)` | false |
| count_nonzero | `aten::count_nonzero(Tensor self, int? dim=None) -> Tensor` | false |
| count_nonzero | `aten::count_nonzero.out(Tensor self, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| cummax | `aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)` | false |
| cummax | `aten::cummax.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices)` | false |
| cummax | `aten::cummax.dimname_out(Tensor self, str dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| cummax | `aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| cummin | `aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)` | false |
| cummin | `aten::cummin.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices)` | false |
| cummin | `aten::cummin.dimname_out(Tensor self, str dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| cummin | `aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| cumprod | `aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor` | false |
| cumprod | `aten::cumprod.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor` | false |
| cumprod | `aten::cumprod.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| cumprod | `aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| cumprod_ | `aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)` | false |
| cumprod_ | `aten::cumprod_.dimname(Tensor(a!) self, str dim, *, ScalarType? dtype=None) -> Tensor(a!)` | false |
| cumsum | `aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor` | false |
| cumsum | `aten::cumsum.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor` | false |
| cumsum | `aten::cumsum.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| cumsum | `aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| cumsum_ | `aten::cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)` | false |
| cumsum_ | `aten::cumsum_.dimname(Tensor(a!) self, str dim, *, ScalarType? dtype=None) -> Tensor(a!)` | false |
| deg2rad | `aten::deg2rad(Tensor self) -> Tensor` | false |
| deg2rad | `aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| deg2rad_ | `aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)` | false |
| dense_dim | `aten::dense_dim(Tensor self) -> int` | false |
| detach | `aten::detach(Tensor(a) self) -> Tensor(a)` | false |
| diag | `aten::diag(Tensor self, int diagonal=0) -> Tensor` | false |
| diag | `aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| diag_embed | `aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor` | false |
| diag_embed | `aten::diag_embed.out(Tensor self, int offset=0, int dim1=-2, int dim2=-1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| diagonal | `aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)` | false |
| diagonal | `aten::diagonal.Dimname(Tensor(a) self, *, str outdim, str dim1, str dim2, int offset=0) -> Tensor(a)` | false |
| diagonal_backward | `aten::diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor` | false |
| diagonal_backward | `aten::diagonal_backward.out(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2, *, Tensor(a!) out) -> Tensor(a!)` | false |
| diagonal_copy | `aten::diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor` | false |
| diagonal_copy | `aten::diagonal_copy.out(Tensor self, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| diagonal_scatter | `aten::diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor` | false |
| diagonal_scatter | `aten::diagonal_scatter.out(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| digamma | `aten::digamma(Tensor self) -> Tensor` | false |
| digamma | `aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| digamma_ | `aten::digamma_(Tensor(a!) self) -> Tensor(a!)` | false |
| dim | `aten::dim(Tensor self) -> int` | false |
| dist | `aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor` | false |
| dist | `aten::dist.out(Tensor self, Tensor other, Scalar p=2, *, Tensor(a!) out) -> Tensor(a!)` | false |
| div | `aten::div.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| div | `aten::div.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| div | `aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor` | false |
| div | `aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor` | false |
| div | `aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| div | `aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)` | false |
| div | `aten::div.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| div | `aten::div.Scalar_mode_out(Tensor self, Scalar other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)` | false |
| div | `aten::div.int(int a, int b) -> float` | false |
| div | `aten::div.complex(complex a, complex b) -> complex` | false |
| div | `aten::div.float(float a, float b) -> float` | false |
| div | `aten::div(Scalar a, Scalar b) -> float` | false |
| div_ | `aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| div_ | `aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)` | false |
| div_ | `aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| div_ | `aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)` | false |
| divide | `aten::divide.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| divide | `aten::divide.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| divide | `aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor` | false |
| divide | `aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor` | false |
| divide | `aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| divide | `aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)` | false |
| divide_ | `aten::divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| divide_ | `aten::divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)` | false |
| divide_ | `aten::divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)` | false |
| divide_ | `aten::divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| dot | `aten::dot(Tensor self, Tensor tensor) -> Tensor` | false |
| dot | `aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)` | false |
| dropout | `aten::dropout(Tensor input, float p, bool train) -> Tensor` | false |
| elu | `aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor` | false |
| elu | `aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| elu_ | `aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)` | false |
| elu_backward | `aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor` | false |
| elu_backward | `aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| embedding | `aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor` | false |
| embedding | `aten::embedding.out(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| embedding_dense_backward | `aten::embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor` | false |
| embedding_dense_backward | `aten::embedding_dense_backward.out(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, *, Tensor(a!) out) -> Tensor(a!)` | false |
| empty | `aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| empty | `aten::empty.out(SymInt[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| empty | `aten::empty.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| empty | `aten::empty.names_out(int[] size, *, str[]? names, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| empty_like | `aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| empty_like | `aten::empty_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| empty_permuted | `aten::empty_permuted(SymInt[] size, int[] physical_layout, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| empty_permuted | `aten::empty_permuted.out(SymInt[] size, int[] physical_layout, *, Tensor(a!) out) -> Tensor(a!)` | false |
| empty_strided | `aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| empty_strided | `aten::empty_strided.out(SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!)` | false |
| eq | `aten::eq.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| eq | `aten::eq.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| eq | `aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| eq | `aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| eq | `aten::eq.int_list(int[] a, int[] b) -> bool` | false |
| eq | `aten::eq.device(Device a, Device b) -> bool` | false |
| eq | `aten::eq.bool(bool a, bool b) -> bool` | false |
| eq | `aten::eq.enum(AnyEnumType a, AnyEnumType b) -> bool` | false |
| eq | `aten::eq.int(int a, int b) -> bool` | false |
| eq | `aten::eq.complex(complex a, complex b) -> bool` | false |
| eq | `aten::eq.float(float a, float b) -> bool` | false |
| eq | `aten::eq.int_float(int a, float b) -> bool` | false |
| eq | `aten::eq.float_int(float a, int b) -> bool` | false |
| eq | `aten::eq.float_complex(float a, complex b) -> bool` | false |
| eq | `aten::eq.complex_float(complex a, float b) -> bool` | false |
| eq | `aten::eq(Scalar a, Scalar b) -> bool` | false |
| eq | `aten::eq.str(str a, str b) -> bool` | false |
| eq | `aten::eq.float_list(float[] a, float[] b) -> bool` | false |
| eq | `aten::eq.Tensor_list(Tensor[] a, Tensor[] b) -> bool` | false |
| eq | `aten::eq.bool_list(bool[] a, bool[] b) -> bool` | false |
| eq | `aten::eq.str_list(str[] a, str[] b) -> bool` | false |
| eq_ | `aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| eq_ | `aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| erf | `aten::erf(Tensor self) -> Tensor` | false |
| erf | `aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| erf | `aten::erf.int(int a) -> float` | false |
| erf | `aten::erf.float(float a) -> float` | false |
| erf | `aten::erf.Scalar(Scalar a) -> Scalar` | false |
| erf_ | `aten::erf_(Tensor(a!) self) -> Tensor(a!)` | false |
| erfc | `aten::erfc(Tensor self) -> Tensor` | false |
| erfc | `aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| erfc | `aten::erfc.int(int a) -> float` | false |
| erfc | `aten::erfc.float(float a) -> float` | false |
| erfc | `aten::erfc.Scalar(Scalar a) -> Scalar` | false |
| erfc_ | `aten::erfc_(Tensor(a!) self) -> Tensor(a!)` | false |
| erfinv | `aten::erfinv(Tensor self) -> Tensor` | false |
| erfinv | `aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| erfinv_ | `aten::erfinv_(Tensor(a!) self) -> Tensor(a!)` | false |
| exp | `aten::exp(Tensor self) -> Tensor` | false |
| exp | `aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| exp | `aten::exp.int(int a) -> float` | false |
| exp | `aten::exp.float(float a) -> float` | false |
| exp | `aten::exp.complex(complex a) -> complex` | false |
| exp | `aten::exp.Scalar(Scalar a) -> Scalar` | false |
| exp2 | `aten::exp2(Tensor self) -> Tensor` | false |
| exp2 | `aten::exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| exp2_ | `aten::exp2_(Tensor(a!) self) -> Tensor(a!)` | false |
| exp_ | `aten::exp_(Tensor(a!) self) -> Tensor(a!)` | false |
| expand | `aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)` | false |
| expand_copy | `aten::expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor` | false |
| expand_copy | `aten::expand_copy.out(Tensor self, SymInt[] size, *, bool implicit=False, Tensor(a!) out) -> Tensor(a!)` | false |
| expm1 | `aten::expm1(Tensor self) -> Tensor` | false |
| expm1 | `aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| expm1 | `aten::expm1.int(int a) -> float` | false |
| expm1 | `aten::expm1.float(float a) -> float` | false |
| expm1 | `aten::expm1.Scalar(Scalar a) -> Scalar` | false |
| expm1_ | `aten::expm1_(Tensor(a!) self) -> Tensor(a!)` | false |
| exponential | `aten::exponential(Tensor self, float lambd=1., *, Generator? generator=None) -> Tensor` | false |
| exponential | `aten::exponential.out(Tensor self, float lambd=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| exponential_ | `aten::exponential_(Tensor(a!) self, float lambd=1., *, Generator? generator=None) -> Tensor(a!)` | false |
| eye | `aten::eye(SymInt n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| eye | `aten::eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| eye | `aten::eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| eye | `aten::eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_fft | `aten::fft_fft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor` | false |
| fft_fft | `aten::fft_fft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_fft2 | `aten::fft_fft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor` | false |
| fft_fft2 | `aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_fftn | `aten::fft_fftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor` | false |
| fft_fftn | `aten::fft_fftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_fftshift | `aten::fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor` | false |
| fft_hfft | `aten::fft_hfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor` | false |
| fft_hfft | `aten::fft_hfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_hfft2 | `aten::fft_hfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor` | false |
| fft_hfft2 | `aten::fft_hfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_hfftn | `aten::fft_hfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor` | false |
| fft_hfftn | `aten::fft_hfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_ifft | `aten::fft_ifft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor` | false |
| fft_ifft | `aten::fft_ifft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_ifft2 | `aten::fft_ifft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor` | false |
| fft_ifft2 | `aten::fft_ifft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_ifftn | `aten::fft_ifftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor` | false |
| fft_ifftn | `aten::fft_ifftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_ifftshift | `aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor` | false |
| fft_ihfft | `aten::fft_ihfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor` | false |
| fft_ihfft | `aten::fft_ihfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_ihfft2 | `aten::fft_ihfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor` | false |
| fft_ihfft2 | `aten::fft_ihfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_ihfftn | `aten::fft_ihfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor` | false |
| fft_ihfftn | `aten::fft_ihfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_irfft | `aten::fft_irfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor` | false |
| fft_irfft | `aten::fft_irfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_irfft2 | `aten::fft_irfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor` | false |
| fft_irfft2 | `aten::fft_irfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_irfftn | `aten::fft_irfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor` | false |
| fft_irfftn | `aten::fft_irfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_rfft | `aten::fft_rfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor` | false |
| fft_rfft | `aten::fft_rfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_rfft2 | `aten::fft_rfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor` | false |
| fft_rfft2 | `aten::fft_rfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fft_rfftn | `aten::fft_rfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor` | false |
| fft_rfftn | `aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fill | `aten::fill.Scalar(Tensor self, Scalar value) -> Tensor` | false |
| fill | `aten::fill.Scalar_out(Tensor self, Scalar value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fill | `aten::fill.Tensor(Tensor self, Tensor value) -> Tensor` | false |
| fill | `aten::fill.Tensor_out(Tensor self, Tensor value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fill_ | `aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)` | false |
| fill_ | `aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)` | false |
| fix | `aten::fix(Tensor self) -> Tensor` | false |
| fix | `aten::fix.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fix_ | `aten::fix_(Tensor(a!) self) -> Tensor(a!)` | false |
| flip | `aten::flip(Tensor self, int[] dims) -> Tensor` | false |
| flip | `aten::flip.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)` | false |
| float_power_ | `aten::float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)` | false |
| float_power_ | `aten::float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)` | false |
| floor | `aten::floor(Tensor self) -> Tensor` | false |
| floor | `aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| floor | `aten::floor.int(int a) -> int` | false |
| floor | `aten::floor.float(float a) -> int` | false |
| floor | `aten::floor.Scalar(Scalar a) -> Scalar` | false |
| floor_ | `aten::floor_(Tensor(a!) self) -> Tensor(a!)` | false |
| floor_divide | `aten::floor_divide(Tensor self, Tensor other) -> Tensor` | false |
| floor_divide | `aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| floor_divide | `aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| floor_divide | `aten::floor_divide.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| floor_divide_ | `aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| floor_divide_ | `aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| fmax | `aten::fmax(Tensor self, Tensor other) -> Tensor` | false |
| fmax | `aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fmin | `aten::fmin(Tensor self, Tensor other) -> Tensor` | false |
| fmin | `aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fmod | `aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| fmod | `aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| fmod | `aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fmod | `aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| fmod | `aten::fmod.int(int a, int b) -> float` | false |
| fmod | `aten::fmod.float(float a, float b) -> float` | false |
| fmod | `aten::fmod.int_float(int a, float b) -> float` | false |
| fmod | `aten::fmod.float_int(float a, int b) -> float` | false |
| fmod | `aten::fmod(Scalar a, Scalar b) -> float` | false |
| fmod_ | `aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| fmod_ | `aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| frac | `aten::frac(Tensor self) -> Tensor` | false |
| frac | `aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| frac_ | `aten::frac_(Tensor(a!) self) -> Tensor(a!)` | false |
| fractional_max_pool2d | `aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)` | false |
| fractional_max_pool2d | `aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))` | false |
| frexp | `aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)` | false |
| frexp | `aten::frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)` | false |
| frexp | `aten::frexp(float a) -> (float, int)` | false |
| full | `aten::full.names(int[] size, Scalar fill_value, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| full | `aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| full | `aten::full.names_out(int[] size, Scalar fill_value, *, str[]? names, Tensor(a!) out) -> Tensor(a!)` | false |
| full | `aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| full_like | `aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| full_like | `aten::full_like.out(Tensor self, Scalar fill_value, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| gather | `aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor` | false |
| gather | `aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)` | false |
| gather | `aten::gather.dimname(Tensor self, str dim, Tensor index, *, bool sparse_grad=False) -> Tensor` | false |
| gather | `aten::gather.dimname_out(Tensor self, str dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)` | false |
| gcd | `aten::gcd(Tensor self, Tensor other) -> Tensor` | false |
| gcd | `aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| gcd | `aten::gcd.int(int a, int b) -> int` | false |
| gcd_ | `aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| ge | `aten::ge.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| ge | `aten::ge.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| ge | `aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| ge | `aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| ge | `aten::ge.int(int a, int b) -> bool` | false |
| ge | `aten::ge.float(float a, float b) -> bool` | false |
| ge | `aten::ge.int_float(int a, float b) -> bool` | false |
| ge | `aten::ge.float_int(float a, int b) -> bool` | false |
| ge | `aten::ge(Scalar a, Scalar b) -> bool` | false |
| ge | `aten::ge.str(str a, str b) -> bool` | false |
| ge_ | `aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| ge_ | `aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| gelu | `aten::gelu(Tensor self, *, str approximate="none") -> Tensor` | false |
| gelu | `aten::gelu.out(Tensor self, *, str approximate="none", Tensor(a!) out) -> Tensor(a!)` | false |
| gelu_ | `aten::gelu_(Tensor(a!) self, *, str approximate="none") -> Tensor(a!)` | false |
| gelu_backward | `aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate="none") -> Tensor` | false |
| gelu_backward | `aten::gelu_backward.grad_input(Tensor grad_output, Tensor self, *, str approximate="none", Tensor(a!) grad_input) -> Tensor(a!)` | false |
| geometric | `aten::geometric(Tensor self, float p, *, Generator? generator=None) -> Tensor` | false |
| geometric | `aten::geometric.out(Tensor self, float p, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| geometric_ | `aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)` | false |
| glu | `aten::glu(Tensor self, int dim=-1) -> Tensor` | false |
| glu | `aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| glu_backward | `aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor` | false |
| glu_backward | `aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| greater | `aten::greater.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| greater | `aten::greater.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| greater | `aten::greater.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| greater | `aten::greater.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| greater_ | `aten::greater_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| greater_ | `aten::greater_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| greater_equal | `aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| greater_equal | `aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| greater_equal | `aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| greater_equal | `aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| greater_equal_ | `aten::greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| greater_equal_ | `aten::greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| grid_sampler_2d | `aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor` | false |
| grid_sampler_2d | `aten::grid_sampler_2d.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)` | false |
| grid_sampler_2d_backward | `aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)` | false |
| grid_sampler_2d_backward | `aten::grid_sampler_2d_backward.out(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| grid_sampler_3d | `aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor` | false |
| grid_sampler_3d | `aten::grid_sampler_3d.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)` | false |
| grid_sampler_3d_backward | `aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)` | false |
| grid_sampler_3d_backward | `aten::grid_sampler_3d_backward.out(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| gru | `aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)` | false |
| gru | `aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)` | false |
| gt | `aten::gt.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| gt | `aten::gt.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| gt | `aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| gt | `aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| gt | `aten::gt.int(int a, int b) -> bool` | false |
| gt | `aten::gt.float(float a, float b) -> bool` | false |
| gt | `aten::gt.int_float(int a, float b) -> bool` | false |
| gt | `aten::gt.float_int(float a, int b) -> bool` | false |
| gt | `aten::gt(Scalar a, Scalar b) -> bool` | false |
| gt | `aten::gt.str(str a, str b) -> bool` | false |
| gt_ | `aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| gt_ | `aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| hardshrink | `aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor` | false |
| hardshrink | `aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)` | false |
| hardsigmoid | `aten::hardsigmoid(Tensor self) -> Tensor` | false |
| hardsigmoid | `aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| hardsigmoid_ | `aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)` | false |
| hardsigmoid_backward | `aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor` | false |
| hardsigmoid_backward | `aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| hardswish | `aten::hardswish(Tensor self) -> Tensor` | false |
| hardswish | `aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| hardswish_ | `aten::hardswish_(Tensor(a!) self) -> Tensor(a!)` | false |
| hardswish_backward | `aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor` | false |
| hardswish_backward | `aten::hardswish_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| hardtanh | `aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor` | false |
| hardtanh | `aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| hardtanh_ | `aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)` | false |
| hardtanh_backward | `aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor` | false |
| hardtanh_backward | `aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| heaviside | `aten::heaviside(Tensor self, Tensor values) -> Tensor` | false |
| heaviside | `aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)` | false |
| heaviside_ | `aten::heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)` | false |
| hinge_embedding_loss | `aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1., int reduction=1) -> Tensor` | false |
| histc | `aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor` | false |
| histc | `aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| huber_loss | `aten::huber_loss(Tensor self, Tensor target, int reduction=1, float delta=1.) -> Tensor` | false |
| huber_loss | `aten::huber_loss.out(Tensor self, Tensor target, int reduction=1, float delta=1., *, Tensor(a!) out) -> Tensor(a!)` | false |
| huber_loss_backward | `aten::huber_loss_backward.out(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| huber_loss_backward | `aten::huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor` | false |
| hypot | `aten::hypot(Tensor self, Tensor other) -> Tensor` | false |
| hypot | `aten::hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| hypot_ | `aten::hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| i0 | `aten::i0(Tensor self) -> Tensor` | false |
| i0 | `aten::i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| i0_ | `aten::i0_(Tensor(a!) self) -> Tensor(a!)` | false |
| igamma | `aten::igamma(Tensor self, Tensor other) -> Tensor` | false |
| igamma | `aten::igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| igamma_ | `aten::igamma_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| igammac | `aten::igammac(Tensor self, Tensor other) -> Tensor` | false |
| igammac | `aten::igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| igammac_ | `aten::igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| im2col | `aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor` | false |
| im2col | `aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)` | false |
| imag | `aten::imag(Tensor(a) self) -> Tensor(a)` | false |
| index | `aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor` | false |
| index | `aten::index.Tensor_out(Tensor self, Tensor?[] indices, *, Tensor(a!) out) -> Tensor(a!)` | false |
| index | `aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor` | false |
| index | `aten::index.str(str self, str substr, int start=0, int end=-1) -> int` | false |
| index | `aten::index.list_int(int[] self, int el) -> int` | false |
| index | `aten::index.list_float(float[] self, float el) -> int` | false |
| index | `aten::index.list_bool(bool[] self, bool el) -> int` | false |
| index | `aten::index.list_Tensor(Tensor[] self, Tensor el) -> int` | false |
| index | `aten::index.list_str(str[] self, str el) -> int` | false |
| index_add | `aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor` | false |
| index_add | `aten::index_add.out(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| index_add | `aten::index_add.dimname(Tensor self, str dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor` | false |
| index_add_ | `aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor(a!)` | false |
| index_copy | `aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor` | false |
| index_copy | `aten::index_copy.dimname(Tensor self, str dim, Tensor index, Tensor source) -> Tensor` | false |
| index_copy | `aten::index_copy.out(Tensor self, int dim, Tensor index, Tensor source, *, Tensor(a!) out) -> Tensor(a!)` | false |
| index_copy_ | `aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)` | false |
| index_copy_ | `aten::index_copy_.dimname(Tensor(a!) self, str dim, Tensor index, Tensor source) -> Tensor(a!)` | false |
| index_fill | `aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor` | false |
| index_fill | `aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor` | false |
| index_fill | `aten::index_fill.Dimname_Scalar(Tensor self, str dim, Tensor index, Scalar value) -> Tensor` | false |
| index_fill | `aten::index_fill.Dimname_Tensor(Tensor self, str dim, Tensor index, Tensor value) -> Tensor` | false |
| index_fill | `aten::index_fill.int_Scalar_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| index_fill | `aten::index_fill.int_Tensor_out(Tensor self, int dim, Tensor index, Tensor value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| index_fill_ | `aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)` | false |
| index_fill_ | `aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)` | false |
| index_fill_ | `aten::index_fill_.Dimname_Scalar(Tensor(a!) self, str dim, Tensor index, Scalar value) -> Tensor(a!)` | false |
| index_fill_ | `aten::index_fill_.Dimname_Tensor(Tensor(a!) self, str dim, Tensor index, Tensor value) -> Tensor(a!)` | false |
| index_put | `aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor` | false |
| index_put | `aten::index_put.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| index_put | `aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor` | false |
| index_put_ | `aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)` | false |
| index_put_ | `aten::index_put_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)` | false |
| index_reduce | `aten::index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor` | false |
| index_reduce | `aten::index_reduce.out(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)` | false |
| index_reduce_ | `aten::index_reduce_(Tensor(a!) self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor(a!)` | false |
| index_select | `aten::index_select(Tensor self, int dim, Tensor index) -> Tensor` | false |
| index_select | `aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)` | false |
| index_select | `aten::index_select.dimname(Tensor self, str dim, Tensor index) -> Tensor` | false |
| index_select | `aten::index_select.dimname_out(Tensor self, str dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)` | false |
| is_coalesced | `aten::is_coalesced(Tensor self) -> bool` | false |
| is_complex | `aten::is_complex(Tensor self) -> bool` | false |
| is_contiguous | `aten::is_contiguous(Tensor self) -> bool` | false |
| is_contiguous | `aten::is_contiguous.memory_format(Tensor self, MemoryFormat memory_format) -> bool` | false |
| is_non_overlapping_and_dense | `aten::is_non_overlapping_and_dense(Tensor self) -> bool` | false |
| is_pinned | `aten::is_pinned(Tensor self, Device? device=None) -> bool` | false |
| is_same_size | `aten::is_same_size(Tensor self, Tensor other) -> bool` | false |
| is_strides_like_format | `aten::is_strides_like_format(Tensor self, MemoryFormat memory_format) -> bool` | false |
| isfinite | `aten::isfinite(Tensor self) -> Tensor` | false |
| isfinite | `aten::isfinite.float(float a) -> bool` | false |
| isfinite | `aten::isfinite.complex(complex a) -> bool` | false |
| isin | `aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor` | false |
| isin | `aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)` | false |
| isin | `aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor` | false |
| isin | `aten::isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)` | false |
| isin | `aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor` | false |
| isin | `aten::isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)` | false |
| isinf | `aten::isinf(Tensor self) -> Tensor` | false |
| isinf | `aten::isinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| isinf | `aten::isinf.float(float a) -> bool` | false |
| isinf | `aten::isinf.complex(complex a) -> bool` | false |
| isnan | `aten::isnan(Tensor self) -> Tensor` | false |
| isnan | `aten::isnan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| isnan | `aten::isnan.float(float a) -> bool` | false |
| isnan | `aten::isnan.complex(complex a) -> bool` | false |
| isneginf | `aten::isneginf(Tensor self) -> Tensor` | false |
| isneginf | `aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| isposinf | `aten::isposinf(Tensor self) -> Tensor` | false |
| isposinf | `aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| istft | `aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor` | false |
| item | `aten::item(Tensor self) -> Scalar` | false |
| kthvalue | `aten::kthvalue(Tensor self, SymInt k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| kthvalue | `aten::kthvalue.dimname(Tensor self, SymInt k, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| kthvalue | `aten::kthvalue.dimname_out(Tensor self, SymInt k, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| kthvalue | `aten::kthvalue.values(Tensor self, SymInt k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| lcm | `aten::lcm(Tensor self, Tensor other) -> Tensor` | false |
| lcm | `aten::lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lcm_ | `aten::lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| le | `aten::le.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| le | `aten::le.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| le | `aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| le | `aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| le | `aten::le.int(int a, int b) -> bool` | false |
| le | `aten::le.float(float a, float b) -> bool` | false |
| le | `aten::le.int_float(int a, float b) -> bool` | false |
| le | `aten::le.float_int(float a, int b) -> bool` | false |
| le | `aten::le(Scalar a, Scalar b) -> bool` | false |
| le | `aten::le.str(str a, str b) -> bool` | false |
| le_ | `aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| le_ | `aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| leaky_relu | `aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor` | false |
| leaky_relu | `aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)` | false |
| leaky_relu_ | `aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)` | false |
| leaky_relu_backward | `aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor` | false |
| leaky_relu_backward | `aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| lerp | `aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor` | false |
| lerp | `aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor` | false |
| lerp | `aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lerp | `aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lerp_ | `aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)` | false |
| lerp_ | `aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)` | false |
| less | `aten::less.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| less | `aten::less.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| less | `aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| less | `aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| less_ | `aten::less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| less_ | `aten::less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| less_equal | `aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| less_equal | `aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| less_equal | `aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| less_equal | `aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| less_equal_ | `aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| less_equal_ | `aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| lgamma | `aten::lgamma(Tensor self) -> Tensor` | false |
| lgamma | `aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lgamma | `aten::lgamma.int(int a) -> float` | false |
| lgamma | `aten::lgamma.float(float a) -> float` | false |
| lgamma | `aten::lgamma.Scalar(Scalar a) -> Scalar` | false |
| lgamma_ | `aten::lgamma_(Tensor(a!) self) -> Tensor(a!)` | false |
| lift | `aten::lift(Tensor self) -> Tensor` | false |
| lift | `aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lift_fresh | `aten::lift_fresh(Tensor(a) self) -> Tensor(a)` | false |
| lift_fresh_copy | `aten::lift_fresh_copy(Tensor self) -> Tensor` | false |
| lift_fresh_copy | `aten::lift_fresh_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_cholesky_ex | `aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)` | false |
| linalg_cholesky_ex | `aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)` | false |
| linalg_cross | `aten::linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor` | false |
| linalg_cross | `aten::linalg_cross.out(Tensor self, Tensor other, *, int dim=-1, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_eig | `aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)` | false |
| linalg_eig | `aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)` | false |
| linalg_eigvals | `aten::linalg_eigvals(Tensor self) -> Tensor` | false |
| linalg_eigvals | `aten::linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_householder_product | `aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor` | false |
| linalg_householder_product | `aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_inv_ex | `aten::linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)` | false |
| linalg_inv_ex | `aten::linalg_inv_ex.inverse(Tensor A, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)` | false |
| linalg_ldl_factor_ex | `aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)` | false |
| linalg_ldl_factor_ex | `aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info)` | false |
| linalg_ldl_solve | `aten::linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> Tensor` | false |
| linalg_ldl_solve | `aten::linalg_ldl_solve.out(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_lu | `aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)` | false |
| linalg_lu | `aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)` | false |
| linalg_lu_factor_ex | `aten::linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)` | false |
| linalg_lu_factor_ex | `aten::linalg_lu_factor_ex.out(Tensor A, *, bool pivot=True, bool check_errors=False, Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info)` | false |
| linalg_lu_solve | `aten::linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor` | false |
| linalg_lu_solve | `aten::linalg_lu_solve.out(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_matrix_exp | `aten::linalg_matrix_exp(Tensor self) -> Tensor` | false |
| linalg_matrix_exp | `aten::linalg_matrix_exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_qr | `aten::linalg_qr(Tensor A, str mode="reduced") -> (Tensor Q, Tensor R)` | false |
| linalg_qr | `aten::linalg_qr.out(Tensor A, str mode="reduced", *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)` | false |
| linalg_solve_triangular | `aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor` | false |
| linalg_solve_triangular | `aten::linalg_solve_triangular.out(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False, Tensor(a!) out) -> Tensor(a!)` | false |
| linalg_vector_norm | `aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| linalg_vector_norm | `aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| linear | `aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor` | false |
| linear | `aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linear_backward | `aten::linear_backward.out(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| linear_backward | `aten::linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)` | false |
| linspace | `aten::linspace.Tensor_Tensor(Tensor start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| linspace | `aten::linspace.Tensor_Scalar(Tensor start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| linspace | `aten::linspace.Scalar_Tensor(Scalar start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| linspace | `aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| linspace | `aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linspace | `aten::linspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linspace | `aten::linspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)` | false |
| linspace | `aten::linspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)` | false |
| log | `aten::log(Tensor self) -> Tensor` | false |
| log | `aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| log | `aten::log.int(int a) -> float` | false |
| log | `aten::log.float(float a) -> float` | false |
| log | `aten::log.complex(complex a) -> complex` | false |
| log | `aten::log.Scalar(Scalar a) -> Scalar` | false |
| log | `aten::log.int_int(int a, int b) -> float` | false |
| log | `aten::log.float_float(float a, float b) -> float` | false |
| log | `aten::log.complex_complex(complex a, complex b) -> complex` | false |
| log | `aten::log.int_float(int a, float b) -> float` | false |
| log | `aten::log.float_int(float a, int b) -> float` | false |
| log | `aten::log.int_complex(int a, complex b) -> complex` | false |
| log | `aten::log.complex_int(complex a, int b) -> complex` | false |
| log | `aten::log.float_complex(float a, complex b) -> complex` | false |
| log | `aten::log.complex_float(complex a, float b) -> complex` | false |
| log | `aten::log.Scalar_Scalar(Scalar a, Scalar b) -> float` | false |
| log10 | `aten::log10(Tensor self) -> Tensor` | false |
| log10 | `aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| log10 | `aten::log10.int(int a) -> float` | false |
| log10 | `aten::log10.float(float a) -> float` | false |
| log10 | `aten::log10.complex(complex a) -> complex` | false |
| log10 | `aten::log10.Scalar(Scalar a) -> Scalar` | false |
| log10_ | `aten::log10_(Tensor(a!) self) -> Tensor(a!)` | false |
| log1p | `aten::log1p(Tensor self) -> Tensor` | false |
| log1p | `aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| log1p | `aten::log1p.int(int a) -> float` | false |
| log1p | `aten::log1p.float(float a) -> float` | false |
| log1p | `aten::log1p.Scalar(Scalar a) -> Scalar` | false |
| log1p_ | `aten::log1p_(Tensor(a!) self) -> Tensor(a!)` | false |
| log2 | `aten::log2(Tensor self) -> Tensor` | false |
| log2 | `aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| log2_ | `aten::log2_(Tensor(a!) self) -> Tensor(a!)` | false |
| log_ | `aten::log_(Tensor(a!) self) -> Tensor(a!)` | false |
| log_normal | `aten::log_normal(Tensor self, float mean=1., float std=2., *, Generator? generator=None) -> Tensor` | false |
| log_normal | `aten::log_normal.out(Tensor self, float mean=1., float std=2., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| log_normal_ | `aten::log_normal_(Tensor(a!) self, float mean=1., float std=2., *, Generator? generator=None) -> Tensor(a!)` | false |
| log_sigmoid_backward | `aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor` | false |
| log_sigmoid_backward | `aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| log_sigmoid_forward | `aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)` | false |
| log_sigmoid_forward | `aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))` | false |
| logaddexp | `aten::logaddexp(Tensor self, Tensor other) -> Tensor` | false |
| logaddexp | `aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logaddexp2 | `aten::logaddexp2(Tensor self, Tensor other) -> Tensor` | false |
| logaddexp2 | `aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logcumsumexp | `aten::logcumsumexp(Tensor self, int dim) -> Tensor` | false |
| logcumsumexp | `aten::logcumsumexp.dimname(Tensor self, str dim) -> Tensor` | false |
| logcumsumexp | `aten::logcumsumexp.dimname_out(Tensor self, str dim, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logcumsumexp | `aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logical_and | `aten::logical_and(Tensor self, Tensor other) -> Tensor` | false |
| logical_and | `aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logical_and_ | `aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| logical_not | `aten::logical_not(Tensor self) -> Tensor` | false |
| logical_not | `aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logical_not_ | `aten::logical_not_(Tensor(a!) self) -> Tensor(a!)` | false |
| logical_or | `aten::logical_or(Tensor self, Tensor other) -> Tensor` | false |
| logical_or | `aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logical_or_ | `aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| logical_xor | `aten::logical_xor(Tensor self, Tensor other) -> Tensor` | false |
| logical_xor | `aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logical_xor_ | `aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| logit | `aten::logit(Tensor self, float? eps=None) -> Tensor` | false |
| logit | `aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logit_ | `aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)` | false |
| logit_backward | `aten::logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor` | false |
| logit_backward | `aten::logit_backward.grad_input(Tensor grad_output, Tensor self, float? eps=None, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| logspace | `aten::logspace.Tensor_Tensor(Tensor start, Tensor end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| logspace | `aten::logspace.Tensor_Scalar(Tensor start, Scalar end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| logspace | `aten::logspace.Scalar_Tensor(Scalar start, Tensor end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| logspace | `aten::logspace(Scalar start, Scalar end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| logspace | `aten::logspace.out(Scalar start, Scalar end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)` | false |
| logspace | `aten::logspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)` | false |
| logspace | `aten::logspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)` | false |
| logspace | `aten::logspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)` | false |
| logsumexp | `aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor` | false |
| logsumexp | `aten::logsumexp.names(Tensor self, str[1] dim, bool keepdim=False) -> Tensor` | false |
| logsumexp | `aten::logsumexp.names_out(Tensor self, str[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| logsumexp | `aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lstm | `aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)` | false |
| lstm | `aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)` | false |
| lt | `aten::lt.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| lt | `aten::lt.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| lt | `aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lt | `aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| lt | `aten::lt.int(int a, int b) -> bool` | false |
| lt | `aten::lt.float(float a, float b) -> bool` | false |
| lt | `aten::lt.int_float(int a, float b) -> bool` | false |
| lt | `aten::lt.float_int(float a, int b) -> bool` | false |
| lt | `aten::lt(Scalar a, Scalar b) -> bool` | false |
| lt | `aten::lt.str(str a, str b) -> bool` | false |
| lt_ | `aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| lt_ | `aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| lu_unpack | `aten::lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)` | false |
| lu_unpack | `aten::lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)` | false |
| margin_ranking_loss | `aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0., int reduction=1) -> Tensor` | false |
| masked_fill | `aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor` | false |
| masked_fill | `aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor` | false |
| masked_fill | `aten::masked_fill.Scalar_out(Tensor self, Tensor mask, Scalar value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| masked_fill | `aten::masked_fill.Tensor_out(Tensor self, Tensor mask, Tensor value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| masked_fill_ | `aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)` | false |
| masked_fill_ | `aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)` | false |
| masked_scatter | `aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor` | false |
| masked_scatter | `aten::masked_scatter.out(Tensor self, Tensor mask, Tensor source, *, Tensor(a!) out) -> Tensor(a!)` | false |
| masked_scatter_ | `aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)` | false |
| masked_scatter_backward | `aten::masked_scatter_backward(Tensor grad_output, Tensor mask, SymInt[] sizes) -> Tensor` | false |
| masked_select | `aten::masked_select(Tensor self, Tensor mask) -> Tensor` | false |
| masked_select | `aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)` | false |
| matmul | `aten::matmul(Tensor self, Tensor other) -> Tensor` | false |
| matmul | `aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| max | `aten::max.other(Tensor self, Tensor other) -> Tensor` | false |
| max | `aten::max(Tensor self) -> Tensor` | false |
| max | `aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| max | `aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| max | `aten::max.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| max | `aten::max.names_dim_max(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| max | `aten::max.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| max | `aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| max_pool2d_with_indices | `aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)` | false |
| max_pool2d_with_indices | `aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))` | false |
| max_pool2d_with_indices_backward | `aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor` | false |
| max_pool2d_with_indices_backward | `aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| max_pool3d_with_indices | `aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)` | false |
| max_pool3d_with_indices | `aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))` | false |
| max_pool3d_with_indices_backward | `aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor` | false |
| max_pool3d_with_indices_backward | `aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| max_unpool2d | `aten::max_unpool2d(Tensor self, Tensor indices, SymInt[2] output_size) -> Tensor` | false |
| max_unpool2d | `aten::max_unpool2d.out(Tensor self, Tensor indices, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| max_unpool3d | `aten::max_unpool3d(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding) -> Tensor` | false |
| max_unpool3d | `aten::max_unpool3d.out(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)` | false |
| maximum | `aten::maximum(Tensor self, Tensor other) -> Tensor` | false |
| maximum | `aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mean | `aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor` | false |
| mean | `aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| mean | `aten::mean.names_dim(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| mean | `aten::mean.names_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| mean | `aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| mean | `aten::mean.dtype_out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| median | `aten::median(Tensor self) -> Tensor` | false |
| median | `aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| median | `aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| median | `aten::median.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| median | `aten::median.names_dim_values(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| median | `aten::median.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| meshgrid | `aten::meshgrid(Tensor[] tensors) -> Tensor[]` | false |
| meshgrid | `aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]` | false |
| min | `aten::min.other(Tensor self, Tensor other) -> Tensor` | false |
| min | `aten::min(Tensor self) -> Tensor` | false |
| min | `aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| min | `aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| min | `aten::min.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| min | `aten::min.names_dim_min(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| min | `aten::min.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| min | `aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| minimum | `aten::minimum(Tensor self, Tensor other) -> Tensor` | false |
| minimum | `aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mish | `aten::mish(Tensor self) -> Tensor` | false |
| mish | `aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mish_ | `aten::mish_(Tensor(a!) self) -> Tensor(a!)` | false |
| mish_backward | `aten::mish_backward(Tensor grad_output, Tensor self) -> Tensor` | false |
| mm | `aten::mm(Tensor self, Tensor mat2) -> Tensor` | false |
| mm | `aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mm | `aten::mm.dtype_out(Tensor self, Tensor mat2, ScalarType out_dtype, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mm | `aten::mm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor` | false |
| mode | `aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| mode | `aten::mode.dimname(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| mode | `aten::mode.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| mode | `aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| mse_loss | `aten::mse_loss(Tensor self, Tensor target, int reduction=1) -> Tensor` | false |
| mse_loss | `aten::mse_loss.out(Tensor self, Tensor target, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mse_loss_backward | `aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor` | false |
| mse_loss_backward | `aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| mul | `aten::mul.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| mul | `aten::mul.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| mul | `aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mul | `aten::mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mul | `aten::mul.left_t(t[] l, int n) -> t[]` | false |
| mul | `aten::mul.right_(int n, t[] l) -> t[]` | false |
| mul | `aten::mul.int(int a, int b) -> int` | false |
| mul | `aten::mul.complex(complex a, complex b) -> complex` | false |
| mul | `aten::mul.float(float a, float b) -> float` | false |
| mul | `aten::mul.int_complex(int a, complex b) -> complex` | false |
| mul | `aten::mul.complex_int(complex a, int b) -> complex` | false |
| mul | `aten::mul.float_complex(float a, complex b) -> complex` | false |
| mul | `aten::mul.complex_float(complex a, float b) -> complex` | false |
| mul | `aten::mul.int_float(int a, float b) -> float` | false |
| mul | `aten::mul.float_int(float a, int b) -> float` | false |
| mul | `aten::mul(Scalar a, Scalar b) -> Scalar` | false |
| mul_ | `aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| mul_ | `aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| mul_ | `aten::mul_.t(t[](a!) l, int n) -> t[](a!)` | false |
| multi_margin_loss | `aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=1) -> Tensor` | false |
| multi_margin_loss | `aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| multilabel_margin_loss_forward | `aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)` | false |
| multilabel_margin_loss_forward | `aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))` | false |
| multinomial | `aten::multinomial(Tensor self, SymInt num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor` | false |
| multinomial | `aten::multinomial.out(Tensor self, SymInt num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| multiply | `aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| multiply | `aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| multiply | `aten::multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| multiply_ | `aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| multiply_ | `aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| mv | `aten::mv(Tensor self, Tensor vec) -> Tensor` | false |
| mv | `aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mvlgamma | `aten::mvlgamma(Tensor self, int p) -> Tensor` | false |
| mvlgamma | `aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)` | false |
| mvlgamma_ | `aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)` | false |
| name | `default` | false |
| nan_to_num | `aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor` | false |
| nan_to_num | `aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| nan_to_num_ | `aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)` | false |
| nanmedian | `aten::nanmedian(Tensor self) -> Tensor` | false |
| nanmedian | `aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| nanmedian | `aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| nanmedian | `aten::nanmedian.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)` | false |
| nanmedian | `aten::nanmedian.names_dim_values(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| nanmedian | `aten::nanmedian.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| nansum | `aten::nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| nansum | `aten::nansum.out(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| narrow | `aten::narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)` | false |
| narrow | `aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length) -> Tensor(a)` | false |
| narrow_copy | `aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor` | false |
| narrow_copy | `aten::narrow_copy.out(Tensor self, int dim, SymInt start, SymInt length, *, Tensor(a!) out) -> Tensor(a!)` | false |
| native_batch_norm | `aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)` | false |
| native_batch_norm | `aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| native_batch_norm_backward | `aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)` | false |
| native_batch_norm_backward | `aten::native_batch_norm_backward.out(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| native_dropout | `aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)` | false |
| native_dropout | `aten::native_dropout.out(Tensor input, float p, bool? train, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| native_dropout_backward | `aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor` | false |
| native_dropout_backward | `aten::native_dropout_backward.out(Tensor grad_output, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!)` | false |
| native_group_norm | `aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)` | false |
| native_group_norm | `aten::native_group_norm.out(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| native_group_norm_backward | `aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)` | false |
| native_group_norm_backward | `aten::native_group_norm_backward.out(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| native_layer_norm | `aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)` | false |
| native_layer_norm | `aten::native_layer_norm.out(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| native_layer_norm_backward | `aten::native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)` | false |
| native_layer_norm_backward | `aten::native_layer_norm_backward.out(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| ne | `aten::ne.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| ne | `aten::ne.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| ne | `aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| ne | `aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| ne | `aten::ne.int_list(int[] a, int[] b) -> bool` | false |
| ne | `aten::ne.device(Device a, Device b) -> bool` | false |
| ne | `aten::ne.bool(bool a, bool b) -> bool` | false |
| ne | `aten::ne.enum(AnyEnumType a, AnyEnumType b) -> bool` | false |
| ne | `aten::ne.int(int a, int b) -> bool` | false |
| ne | `aten::ne.complex(complex a, complex b) -> bool` | false |
| ne | `aten::ne.float(float a, float b) -> bool` | false |
| ne | `aten::ne.int_float(int a, float b) -> bool` | false |
| ne | `aten::ne.float_int(float a, int b) -> bool` | false |
| ne | `aten::ne.float_complex(float a, complex b) -> bool` | false |
| ne | `aten::ne.complex_float(complex a, float b) -> bool` | false |
| ne | `aten::ne(Scalar a, Scalar b) -> bool` | false |
| ne | `aten::ne.str(str a, str b) -> bool` | false |
| ne | `aten::ne.float_list(float[] a, float[] b) -> bool` | false |
| ne | `aten::ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool` | false |
| ne | `aten::ne.bool_list(bool[] a, bool[] b) -> bool` | false |
| ne | `aten::ne.str_list(str[] a, str[] b) -> bool` | false |
| ne_ | `aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| ne_ | `aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| neg | `aten::neg(Tensor self) -> Tensor` | false |
| neg | `aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| neg | `aten::neg.int(int a) -> int` | false |
| neg | `aten::neg.float(float a) -> float` | false |
| neg | `aten::neg.complex(complex a) -> complex` | false |
| neg | `aten::neg.Scalar(Scalar a) -> Scalar` | false |
| neg_ | `aten::neg_(Tensor(a!) self) -> Tensor(a!)` | false |
| negative | `aten::negative(Tensor self) -> Tensor` | false |
| negative | `aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| negative_ | `aten::negative_(Tensor(a!) self) -> Tensor(a!)` | false |
| new_empty | `aten::new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| new_empty | `aten::new_empty.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| new_empty_strided | `aten::new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| new_empty_strided | `aten::new_empty_strided.out(Tensor self, SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!)` | false |
| new_full | `aten::new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| new_full | `aten::new_full.out(Tensor self, SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| new_ones | `aten::new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| new_ones | `aten::new_ones.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| new_zeros | `aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| new_zeros | `aten::new_zeros.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| nextafter | `aten::nextafter(Tensor self, Tensor other) -> Tensor` | false |
| nextafter | `aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| nextafter_ | `aten::nextafter_(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| nll_loss | `aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100) -> Tensor` | false |
| nll_loss | `aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)` | false |
| nll_loss2d_backward | `aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor` | false |
| nll_loss2d_backward | `aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| nll_loss2d_forward | `aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)` | false |
| nll_loss2d_forward | `aten::nll_loss2d_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))` | false |
| nll_loss_backward | `aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor` | false |
| nll_loss_backward | `aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| nll_loss_forward | `aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)` | false |
| nll_loss_forward | `aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))` | false |
| nonzero | `aten::nonzero(Tensor self) -> Tensor` | false |
| nonzero | `aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| nonzero_numpy | `aten::nonzero_numpy(Tensor self) -> Tensor[]` | false |
| nonzero_static | `aten::nonzero_static(Tensor self, *, SymInt size, int fill_value=-1) -> Tensor` | false |
| nonzero_static | `aten::nonzero_static.out(Tensor self, *, SymInt size, int fill_value=-1, Tensor(a!) out) -> Tensor(a!)` | false |
| norm | `aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor` | false |
| norm | `aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor` | false |
| norm | `aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, str[1] dim, bool keepdim=False) -> Tensor` | false |
| norm | `aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor` | false |
| norm | `aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)` | false |
| norm | `aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| norm | `aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor` | false |
| norm | `aten::norm.ScalarOpt_dtype_out(Tensor self, Scalar? p, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)` | false |
| norm | `aten::norm.Scalar_out(Tensor self, Scalar p=2, *, Tensor(a!) out) -> Tensor(a!)` | false |
| norm | `aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, str[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor` | false |
| norm | `aten::norm.names_dtype_out(Tensor self, Scalar? p, str[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)` | false |
| norm | `aten::norm.names_out(Tensor self, Scalar? p, str[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| normal | `aten::normal.Tensor_float(Tensor mean, float std=1., *, Generator? generator=None) -> Tensor` | false |
| normal | `aten::normal.Tensor_float_out(Tensor mean, float std=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| normal | `aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| normal | `aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor` | false |
| normal | `aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor` | false |
| normal | `aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| normal | `aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| normal | `aten::normal.float_float_out(float mean, float std, SymInt[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| normal | `aten::normal.out(Tensor self, float mean=0., float std=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| normal_ | `aten::normal_(Tensor(a!) self, float mean=0., float std=1., *, Generator? generator=None) -> Tensor(a!)` | false |
| not_equal | `aten::not_equal.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| not_equal | `aten::not_equal.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| not_equal | `aten::not_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| not_equal | `aten::not_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| not_equal_ | `aten::not_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| not_equal_ | `aten::not_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| numel | `aten::numel(Tensor self) -> int` | false |
| ones | `aten::ones.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| ones | `aten::ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| ones | `aten::ones.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)` | false |
| ones | `aten::ones.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| ones_like | `aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| ones_like | `aten::ones_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| ormqr | `aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor` | false |
| ormqr | `aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| pad_sequence | `aten::pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0., str padding_side="right") -> Tensor` | false |
| pairwise_distance | `aten::pairwise_distance(Tensor x1, Tensor x2, float p=2., float eps=9.9999999999999995e-07, bool keepdim=False) -> Tensor` | false |
| pdist | `aten::pdist(Tensor self, float p=2.) -> Tensor` | false |
| permute | `aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)` | false |
| permute_copy | `aten::permute_copy(Tensor self, int[] dims) -> Tensor` | false |
| permute_copy | `aten::permute_copy.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)` | false |
| pin_memory | `aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)` | false |
| pixel_shuffle | `aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor` | false |
| pixel_shuffle | `aten::pixel_shuffle.out(Tensor self, int upscale_factor, *, Tensor(a!) out) -> Tensor(a!)` | false |
| pixel_unshuffle | `aten::pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor` | false |
| pixel_unshuffle | `aten::pixel_unshuffle.out(Tensor self, int downscale_factor, *, Tensor(a!) out) -> Tensor(a!)` | false |
| poisson | `aten::poisson(Tensor self, Generator? generator=None) -> Tensor` | false |
| poisson | `aten::poisson.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| polar | `aten::polar(Tensor abs, Tensor angle) -> Tensor` | false |
| polar | `aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)` | false |
| polar | `aten::polar.int(int a, int b) -> complex` | false |
| polar | `aten::polar.float(float a, float b) -> complex` | false |
| polar | `aten::polar.int_float(int a, float b) -> complex` | false |
| polar | `aten::polar.float_int(float a, int b) -> complex` | false |
| polar | `aten::polar.Scalar_Scalar(Scalar a, Scalar b) -> Scalar` | false |
| polygamma | `aten::polygamma(int n, Tensor self) -> Tensor` | false |
| polygamma | `aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| positive | `aten::positive(Tensor(a) self) -> Tensor(a)` | false |
| pow | `aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor` | false |
| pow | `aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor` | false |
| pow | `aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor` | false |
| pow | `aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)` | false |
| pow | `aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)` | false |
| pow | `aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)` | false |
| pow | `aten::pow.int(int a, int b) -> float` | false |
| pow | `aten::pow.complex(complex a, complex b) -> complex` | false |
| pow | `aten::pow.float(float a, float b) -> float` | false |
| pow | `aten::pow.int_float(int a, float b) -> float` | false |
| pow | `aten::pow.float_int(float a, int b) -> float` | false |
| pow | `aten::pow.float_complex(float a, complex b) -> complex` | false |
| pow | `aten::pow.complex_float(complex a, float b) -> complex` | false |
| pow | `aten::pow.Scalar_Scalar(Scalar a, Scalar b) -> float` | false |
| pow | `aten::pow.int_to_int(int a, int b) -> int` | false |
| pow_ | `aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)` | false |
| pow_ | `aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)` | false |
| prelu | `aten::prelu(Tensor self, Tensor weight) -> Tensor` | false |
| prod | `aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor` | false |
| prod | `aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| prod | `aten::prod.dim_Dimname(Tensor self, str dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| prod | `aten::prod.Dimname_out(Tensor self, str dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| prod | `aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| prod | `aten::prod.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| quantized_gru | `aten::quantized_gru.input(Tensor input, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)` | false |
| quantized_gru | `aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)` | false |
| quantized_gru | `aten::quantized_gru.input_legacy(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)` | false |
| quantized_gru | `aten::quantized_gru.data_legacy(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)` | false |
| quantized_lstm | `aten::quantized_lstm.input(Tensor input, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)` | false |
| quantized_lstm | `aten::quantized_lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)` | false |
| quantized_lstm | `aten::quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)` | false |
| quantized_lstm | `aten::quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)` | false |
| rad2deg | `aten::rad2deg(Tensor self) -> Tensor` | false |
| rad2deg | `aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| rad2deg_ | `aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)` | false |
| rand | `aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| rand | `aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| rand | `aten::rand.names(SymInt[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| rand | `aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| rand | `aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| rand | `aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)` | false |
| rand | `aten::rand.names_out(SymInt[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)` | false |
| rand | `aten::rand.generator_with_names_out(SymInt[] size, *, Generator? generator, str[]? names, Tensor(a!) out) -> Tensor(a!)` | false |
| rand_like | `aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| rand_like | `aten::rand_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| randint | `aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randint | `aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randint | `aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randint | `aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randint | `aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| randint | `aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)` | false |
| randint | `aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| randint | `aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)` | false |
| randint_like | `aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| randint_like | `aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| randint_like | `aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| randint_like | `aten::randint_like.Tensor(Tensor self, Tensor high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| randint_like | `aten::randint_like.Tensor_out(Tensor self, Tensor high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| randint_like | `aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| randn | `aten::randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randn | `aten::randn.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randn | `aten::randn.names(SymInt[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randn | `aten::randn.generator_with_names(SymInt[] size, *, Generator? generator, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randn | `aten::randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| randn | `aten::randn.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)` | false |
| randn | `aten::randn.names_out(SymInt[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)` | false |
| randn | `aten::randn.generator_with_names_out(SymInt[] size, *, Generator? generator, str[]? names, Tensor(a!) out) -> Tensor(a!)` | false |
| randn_like | `aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| randn_like | `aten::randn_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| randperm | `aten::randperm(SymInt n, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randperm | `aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| randperm | `aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| randperm | `aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)` | false |
| real | `aten::real(Tensor(a) self) -> Tensor(a)` | false |
| reciprocal | `aten::reciprocal(Tensor self) -> Tensor` | false |
| reciprocal | `aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| reciprocal_ | `aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)` | false |
| reflection_pad1d | `aten::reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor` | false |
| reflection_pad1d | `aten::reflection_pad1d.out(Tensor self, SymInt[2] padding, *, Tensor(a!) out) -> Tensor(a!)` | false |
| reflection_pad1d_backward | `aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor` | false |
| reflection_pad1d_backward | `aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| reflection_pad2d | `aten::reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor` | false |
| reflection_pad2d | `aten::reflection_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)` | false |
| reflection_pad2d_backward | `aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor` | false |
| reflection_pad2d_backward | `aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| reflection_pad3d | `aten::reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor` | false |
| reflection_pad3d | `aten::reflection_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)` | false |
| reflection_pad3d_backward | `aten::reflection_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor` | false |
| reflection_pad3d_backward | `aten::reflection_pad3d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| relu | `aten::relu(Tensor self) -> Tensor` | false |
| relu | `aten::relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| relu6 | `aten::relu6(Tensor self) -> Tensor` | false |
| relu_ | `aten::relu_(Tensor(a!) self) -> Tensor(a!)` | false |
| remainder | `aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| remainder | `aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| remainder | `aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` | false |
| remainder | `aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| remainder | `aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| remainder | `aten::remainder.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| remainder | `aten::remainder.int(int a, int b) -> int` | false |
| remainder | `aten::remainder.float(float a, float b) -> float` | false |
| remainder | `aten::remainder.int_float(int a, float b) -> float` | false |
| remainder | `aten::remainder.float_int(float a, int b) -> float` | false |
| remainder | `aten::remainder(Scalar a, Scalar b) -> Scalar` | false |
| remainder_ | `aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| remainder_ | `aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| renorm | `aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor` | false |
| renorm | `aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)` | false |
| renorm_ | `aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)` | false |
| repeat | `aten::repeat(Tensor self, SymInt[] repeats) -> Tensor` | false |
| repeat | `aten::repeat.out(Tensor self, SymInt[] repeats, *, Tensor(a!) out) -> Tensor(a!)` | false |
| repeat_interleave | `aten::repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor` | false |
| repeat_interleave | `aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor` | false |
| repeat_interleave | `aten::repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor` | false |
| repeat_interleave | `aten::repeat_interleave.Tensor_out(Tensor repeats, *, SymInt? output_size=None, Tensor(a!) out) -> Tensor(a!)` | false |
| replication_pad1d | `aten::replication_pad1d(Tensor self, SymInt[2] padding) -> Tensor` | false |
| replication_pad1d | `aten::replication_pad1d.out(Tensor self, SymInt[2] padding, *, Tensor(a!) out) -> Tensor(a!)` | false |
| replication_pad1d_backward | `aten::replication_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor` | false |
| replication_pad1d_backward | `aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| replication_pad2d | `aten::replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor` | false |
| replication_pad2d | `aten::replication_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)` | false |
| replication_pad2d_backward | `aten::replication_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor` | false |
| replication_pad2d_backward | `aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| replication_pad3d | `aten::replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor` | false |
| replication_pad3d | `aten::replication_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)` | false |
| replication_pad3d_backward | `aten::replication_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor` | false |
| replication_pad3d_backward | `aten::replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| reshape | `aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)` | false |
| resize | `aten::resize(Tensor self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor` | false |
| resize | `aten::resize.out(Tensor self, SymInt[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| resize_as | `aten::resize_as(Tensor self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor` | false |
| resize_as | `aten::resize_as.out(Tensor self, Tensor the_template, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
| resize_as_ | `aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)` | false |
| rnn_relu | `aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)` | false |
| rnn_relu | `aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)` | false |
| rnn_tanh | `aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)` | false |
| rnn_tanh | `aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)` | false |
| roll | `aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor` | false |
| roll | `aten::roll.out(Tensor self, SymInt[1] shifts, int[1] dims=[], *, Tensor(a!) out) -> Tensor(a!)` | false |
| rot90 | `aten::rot90(Tensor self, int k=1, int[] dims=[0, 1]) -> Tensor` | false |
| rot90 | `aten::rot90.out(Tensor self, int k=1, int[] dims=[0, 1], *, Tensor(a!) out) -> Tensor(a!)` | false |
| round | `aten::round(Tensor self) -> Tensor` | false |
| round | `aten::round.decimals(Tensor self, *, int decimals) -> Tensor` | false |
| round | `aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| round | `aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)` | false |
| round | `aten::round.int(int a) -> float` | false |
| round | `aten::round.float(float a) -> float` | false |
| round | `aten::round.Scalar(Scalar a) -> Scalar` | false |
| round_ | `aten::round_(Tensor(a!) self) -> Tensor(a!)` | false |
| round_ | `aten::round_.decimals(Tensor(a!) self, *, int decimals) -> Tensor(a!)` | false |
| rrelu_with_noise | `aten::rrelu_with_noise(Tensor self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor` | false |
| rrelu_with_noise | `aten::rrelu_with_noise.out(Tensor self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| rrelu_with_noise_ | `aten::rrelu_with_noise_(Tensor(a!) self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor(a!)` | false |
| rrelu_with_noise_backward | `aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor` | false |
| rrelu_with_noise_backward | `aten::rrelu_with_noise_backward.out(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result, *, Tensor(a!) out) -> Tensor(a!)` | false |
| rrelu_with_noise_functional | `aten::rrelu_with_noise_functional(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> (Tensor, Tensor noise_out)` | false |
| rsqrt | `aten::rsqrt(Tensor self) -> Tensor` | false |
| rsqrt | `aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| rsqrt_ | `aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)` | false |
| rsub | `aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor` | false |
| rsub | `aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor` | false |
| rsub | `aten::rsub.Tensor_out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| rsub | `aten::rsub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| scalar_tensor | `aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| scalar_tensor | `aten::scalar_tensor.out(Scalar s, *, Tensor(a!) out) -> Tensor(a!)` | false |
| scatter | `aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor` | false |
| scatter | `aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor` | false |
| scatter | `aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor` | false |
| scatter | `aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor` | false |
| scatter | `aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)` | false |
| scatter | `aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| scatter | `aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)` | false |
| scatter | `aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)` | false |
| scatter | `aten::scatter.dimname_src(Tensor self, str dim, Tensor index, Tensor src) -> Tensor` | false |
| scatter | `aten::scatter.dimname_value(Tensor self, str dim, Tensor index, Scalar value) -> Tensor` | false |
| scatter_ | `aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)` | false |
| scatter_ | `aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)` | false |
| scatter_ | `aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)` | false |
| scatter_ | `aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)` | false |
| scatter_add | `aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor` | false |
| scatter_add | `aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)` | false |
| scatter_add | `aten::scatter_add.dimname(Tensor self, str dim, Tensor index, Tensor src) -> Tensor` | false |
| scatter_add_ | `aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)` | false |
| scatter_reduce | `aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor` | false |
| scatter_reduce | `aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)` | false |
| scatter_reduce_ | `aten::scatter_reduce_.two(Tensor(a!) self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor(a!)` | false |
| searchsorted | `aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor` | false |
| searchsorted | `aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)` | false |
| searchsorted | `aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor` | false |
| searchsorted | `aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)` | false |
| segment_reduce | `aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor` | false |
| segment_reduce | `aten::segment_reduce.out(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False, Scalar? initial=None, Tensor(a!) out) -> Tensor(a!)` | false |
| select | `aten::select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a)` | false |
| select | `aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)` | false |
| select | `aten::select.t(t[](a) list, int idx) -> t(*)` | false |
| select_backward | `aten::select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index) -> Tensor` | false |
| select_backward | `aten::select_backward.out(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)` | false |
| select_scatter | `aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor` | false |
| select_scatter | `aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)` | false |
| selu | `aten::selu(Tensor self) -> Tensor` | false |
| selu_ | `aten::selu_(Tensor(a!) self) -> Tensor(a!)` | false |
| set_ | `aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)` | false |
| set_ | `aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)` | false |
| set_ | `aten::set_(Tensor(a!) self) -> Tensor(a!)` | false |
| set_ | `aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)` | false |
| set_ | `aten::set_.source_Tensor_storage_offset(Tensor(a!) self, Tensor source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)` | false |
| sgn | `aten::sgn(Tensor self) -> Tensor` | false |
| sgn | `aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sgn_ | `aten::sgn_(Tensor(a!) self) -> Tensor(a!)` | false |
| sigmoid | `aten::sigmoid(Tensor self) -> Tensor` | false |
| sigmoid | `aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sigmoid_ | `aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)` | false |
| sigmoid_backward | `aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor` | false |
| sigmoid_backward | `aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| sign | `aten::sign(Tensor self) -> Tensor` | false |
| sign | `aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sign_ | `aten::sign_(Tensor(a!) self) -> Tensor(a!)` | false |
| signbit | `aten::signbit(Tensor self) -> Tensor` | false |
| signbit | `aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| silu | `aten::silu(Tensor self) -> Tensor` | false |
| silu | `aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| silu_ | `aten::silu_(Tensor(a!) self) -> Tensor(a!)` | false |
| silu_backward | `aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor` | false |
| silu_backward | `aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| sin | `aten::sin(Tensor self) -> Tensor` | false |
| sin | `aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sin | `aten::sin.int(int a) -> float` | false |
| sin | `aten::sin.float(float a) -> float` | false |
| sin | `aten::sin.complex(complex a) -> complex` | false |
| sin | `aten::sin.Scalar(Scalar a) -> Scalar` | false |
| sin_ | `aten::sin_(Tensor(a!) self) -> Tensor(a!)` | false |
| sinc | `aten::sinc(Tensor self) -> Tensor` | false |
| sinc | `aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sinc_ | `aten::sinc_(Tensor(a!) self) -> Tensor(a!)` | false |
| sinh | `aten::sinh(Tensor self) -> Tensor` | false |
| sinh | `aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sinh | `aten::sinh.int(int a) -> float` | false |
| sinh | `aten::sinh.float(float a) -> float` | false |
| sinh | `aten::sinh.complex(complex a) -> complex` | false |
| sinh | `aten::sinh.Scalar(Scalar a) -> Scalar` | false |
| sinh_ | `aten::sinh_(Tensor(a!) self) -> Tensor(a!)` | false |
| size | `aten::size.int(Tensor self, int dim) -> int` | false |
| size | `aten::size.Dimname(Tensor self, str dim) -> int` | false |
| size | `aten::size(Tensor self) -> int[]` | false |
| slice | `aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)` | false |
| slice | `aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> t[]` | false |
| slice | `aten::slice.str(str string, int? start=None, int? end=None, int step=1) -> str` | false |
| slice_backward | `aten::slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor` | false |
| slice_backward | `aten::slice_backward.out(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step, *, Tensor(a!) out) -> Tensor(a!)` | false |
| slice_scatter | `aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor` | false |
| slice_scatter | `aten::slice_scatter.out(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| smooth_l1_loss | `aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=1, float beta=1.) -> Tensor` | false |
| smooth_l1_loss | `aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=1, float beta=1., *, Tensor(a!) out) -> Tensor(a!)` | false |
| smooth_l1_loss_backward | `aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| smooth_l1_loss_backward | `aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor` | false |
| soft_margin_loss | `aten::soft_margin_loss(Tensor self, Tensor target, int reduction=1) -> Tensor` | false |
| soft_margin_loss | `aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| soft_margin_loss_backward | `aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor` | false |
| soft_margin_loss_backward | `aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| softplus | `aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor` | false |
| softplus | `aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)` | false |
| softplus_backward | `aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold) -> Tensor` | false |
| softplus_backward | `aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| softshrink | `aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor` | false |
| softshrink | `aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sort | `aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)` | false |
| sort | `aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)` | false |
| sort | `aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| sort | `aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| sort | `aten::sort.dimname(Tensor self, str dim, bool descending=False) -> (Tensor values, Tensor indices)` | false |
| sort | `aten::sort.dimname_values(Tensor self, str dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| sort | `aten::sort.dimname_stable(Tensor self, *, bool? stable, str dim, bool descending=False) -> (Tensor values, Tensor indices)` | false |
| sort | `aten::sort.dimname_values_stable(Tensor self, *, bool? stable, str dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| sort | `aten::sort.int(int[](a!) self, bool reverse=False) -> ()` | false |
| sort | `aten::sort.float(float[](a!) self, bool reverse=False) -> ()` | false |
| sort | `aten::sort.Tensor(Tensor[](a!) self, bool reverse=False) -> ()` | false |
| sort | `aten::sort.bool(bool[](a!) self, bool reverse=False) -> ()` | false |
| sort | `aten::sort.str(str[](a!) self, bool reverse=False) -> ()` | false |
| sort | `aten::sort.any(t[](a!) self, bool reverse=False) -> ()` | false |
| sparse_dim | `aten::sparse_dim(Tensor self) -> int` | false |
| special_airy_ai | `aten::special_airy_ai(Tensor x) -> Tensor` | false |
| special_airy_ai | `aten::special_airy_ai.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_bessel_j0 | `aten::special_bessel_j0(Tensor self) -> Tensor` | false |
| special_bessel_j0 | `aten::special_bessel_j0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_bessel_j1 | `aten::special_bessel_j1(Tensor self) -> Tensor` | false |
| special_bessel_j1 | `aten::special_bessel_j1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_bessel_y0 | `aten::special_bessel_y0(Tensor self) -> Tensor` | false |
| special_bessel_y0 | `aten::special_bessel_y0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_bessel_y1 | `aten::special_bessel_y1(Tensor self) -> Tensor` | false |
| special_bessel_y1 | `aten::special_bessel_y1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_t | `aten::special_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_t | `aten::special_chebyshev_polynomial_t.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_t | `aten::special_chebyshev_polynomial_t.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_t | `aten::special_chebyshev_polynomial_t.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_t | `aten::special_chebyshev_polynomial_t.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_chebyshev_polynomial_t | `aten::special_chebyshev_polynomial_t.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_u | `aten::special_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_u | `aten::special_chebyshev_polynomial_u.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_u | `aten::special_chebyshev_polynomial_u.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_u | `aten::special_chebyshev_polynomial_u.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_u | `aten::special_chebyshev_polynomial_u.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_chebyshev_polynomial_u | `aten::special_chebyshev_polynomial_u.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_v | `aten::special_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_v | `aten::special_chebyshev_polynomial_v.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_v | `aten::special_chebyshev_polynomial_v.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_v | `aten::special_chebyshev_polynomial_v.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_v | `aten::special_chebyshev_polynomial_v.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_chebyshev_polynomial_v | `aten::special_chebyshev_polynomial_v.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_w | `aten::special_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_w | `aten::special_chebyshev_polynomial_w.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_w | `aten::special_chebyshev_polynomial_w.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_chebyshev_polynomial_w | `aten::special_chebyshev_polynomial_w.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_chebyshev_polynomial_w | `aten::special_chebyshev_polynomial_w.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_chebyshev_polynomial_w | `aten::special_chebyshev_polynomial_w.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_entr | `aten::special_entr(Tensor self) -> Tensor` | false |
| special_entr | `aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_erfcx | `aten::special_erfcx(Tensor self) -> Tensor` | false |
| special_erfcx | `aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_hermite_polynomial_h | `aten::special_hermite_polynomial_h(Tensor x, Tensor n) -> Tensor` | false |
| special_hermite_polynomial_h | `aten::special_hermite_polynomial_h.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_hermite_polynomial_h | `aten::special_hermite_polynomial_h.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_hermite_polynomial_h | `aten::special_hermite_polynomial_h.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_hermite_polynomial_h | `aten::special_hermite_polynomial_h.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_hermite_polynomial_h | `aten::special_hermite_polynomial_h.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_hermite_polynomial_he | `aten::special_hermite_polynomial_he(Tensor x, Tensor n) -> Tensor` | false |
| special_hermite_polynomial_he | `aten::special_hermite_polynomial_he.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_hermite_polynomial_he | `aten::special_hermite_polynomial_he.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_hermite_polynomial_he | `aten::special_hermite_polynomial_he.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_hermite_polynomial_he | `aten::special_hermite_polynomial_he.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_hermite_polynomial_he | `aten::special_hermite_polynomial_he.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_i0e | `aten::special_i0e(Tensor self) -> Tensor` | false |
| special_i0e | `aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_i1 | `aten::special_i1(Tensor self) -> Tensor` | false |
| special_i1 | `aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_i1e | `aten::special_i1e(Tensor self) -> Tensor` | false |
| special_i1e | `aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_laguerre_polynomial_l | `aten::special_laguerre_polynomial_l(Tensor x, Tensor n) -> Tensor` | false |
| special_laguerre_polynomial_l | `aten::special_laguerre_polynomial_l.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_laguerre_polynomial_l | `aten::special_laguerre_polynomial_l.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_laguerre_polynomial_l | `aten::special_laguerre_polynomial_l.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_laguerre_polynomial_l | `aten::special_laguerre_polynomial_l.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_laguerre_polynomial_l | `aten::special_laguerre_polynomial_l.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_legendre_polynomial_p | `aten::special_legendre_polynomial_p(Tensor x, Tensor n) -> Tensor` | false |
| special_legendre_polynomial_p | `aten::special_legendre_polynomial_p.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_legendre_polynomial_p | `aten::special_legendre_polynomial_p.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_legendre_polynomial_p | `aten::special_legendre_polynomial_p.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_legendre_polynomial_p | `aten::special_legendre_polynomial_p.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_legendre_polynomial_p | `aten::special_legendre_polynomial_p.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_log_ndtr | `aten::special_log_ndtr(Tensor self) -> Tensor` | false |
| special_log_ndtr | `aten::special_log_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_modified_bessel_i0 | `aten::special_modified_bessel_i0(Tensor self) -> Tensor` | false |
| special_modified_bessel_i0 | `aten::special_modified_bessel_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_modified_bessel_i1 | `aten::special_modified_bessel_i1(Tensor self) -> Tensor` | false |
| special_modified_bessel_i1 | `aten::special_modified_bessel_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_modified_bessel_k0 | `aten::special_modified_bessel_k0(Tensor self) -> Tensor` | false |
| special_modified_bessel_k0 | `aten::special_modified_bessel_k0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_modified_bessel_k1 | `aten::special_modified_bessel_k1(Tensor self) -> Tensor` | false |
| special_modified_bessel_k1 | `aten::special_modified_bessel_k1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_ndtr | `aten::special_ndtr(Tensor self) -> Tensor` | false |
| special_ndtr | `aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_ndtri | `aten::special_ndtri(Tensor self) -> Tensor` | false |
| special_ndtri | `aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_scaled_modified_bessel_k0 | `aten::special_scaled_modified_bessel_k0(Tensor x) -> Tensor` | false |
| special_scaled_modified_bessel_k0 | `aten::special_scaled_modified_bessel_k0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_scaled_modified_bessel_k1 | `aten::special_scaled_modified_bessel_k1(Tensor x) -> Tensor` | false |
| special_scaled_modified_bessel_k1 | `aten::special_scaled_modified_bessel_k1.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_t | `aten::special_shifted_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_t | `aten::special_shifted_chebyshev_polynomial_t.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_t | `aten::special_shifted_chebyshev_polynomial_t.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_t | `aten::special_shifted_chebyshev_polynomial_t.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_t | `aten::special_shifted_chebyshev_polynomial_t.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_t | `aten::special_shifted_chebyshev_polynomial_t.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_u | `aten::special_shifted_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_u | `aten::special_shifted_chebyshev_polynomial_u.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_u | `aten::special_shifted_chebyshev_polynomial_u.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_u | `aten::special_shifted_chebyshev_polynomial_u.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_u | `aten::special_shifted_chebyshev_polynomial_u.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_u | `aten::special_shifted_chebyshev_polynomial_u.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_v | `aten::special_shifted_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_v | `aten::special_shifted_chebyshev_polynomial_v.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_v | `aten::special_shifted_chebyshev_polynomial_v.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_v | `aten::special_shifted_chebyshev_polynomial_v.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_v | `aten::special_shifted_chebyshev_polynomial_v.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_v | `aten::special_shifted_chebyshev_polynomial_v.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_w | `aten::special_shifted_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_w | `aten::special_shifted_chebyshev_polynomial_w.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_w | `aten::special_shifted_chebyshev_polynomial_w.x_scalar(Scalar x, Tensor n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_w | `aten::special_shifted_chebyshev_polynomial_w.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_shifted_chebyshev_polynomial_w | `aten::special_shifted_chebyshev_polynomial_w.n_scalar(Tensor x, Scalar n) -> Tensor` | false |
| special_shifted_chebyshev_polynomial_w | `aten::special_shifted_chebyshev_polynomial_w.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_spherical_bessel_j0 | `aten::special_spherical_bessel_j0(Tensor x) -> Tensor` | false |
| special_spherical_bessel_j0 | `aten::special_spherical_bessel_j0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_xlog1py | `aten::special_xlog1py(Tensor self, Tensor other) -> Tensor` | false |
| special_xlog1py | `aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor` | false |
| special_xlog1py | `aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor` | false |
| special_xlog1py | `aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_xlog1py | `aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_xlog1py | `aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_zeta | `aten::special_zeta(Tensor self, Tensor other) -> Tensor` | false |
| special_zeta | `aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor` | false |
| special_zeta | `aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor` | false |
| special_zeta | `aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_zeta | `aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| special_zeta | `aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| split | `aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]` | false |
| split | `aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]` | false |
| split | `aten::split.str(str self, str? separator=None, int max=-1) -> str[]` | false |
| split | `aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]` | false |
| split_with_sizes | `aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]` | false |
| split_with_sizes_copy | `aten::split_with_sizes_copy(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]` | false |
| split_with_sizes_copy | `aten::split_with_sizes_copy.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()` | false |
| sqrt | `aten::sqrt(Tensor self) -> Tensor` | false |
| sqrt | `aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sqrt | `aten::sqrt.int(int a) -> float` | false |
| sqrt | `aten::sqrt.float(float a) -> float` | false |
| sqrt | `aten::sqrt.complex(complex a) -> complex` | false |
| sqrt | `aten::sqrt.Scalar(Scalar a) -> Scalar` | false |
| sqrt_ | `aten::sqrt_(Tensor(a!) self) -> Tensor(a!)` | false |
| square | `aten::square(Tensor self) -> Tensor` | false |
| square | `aten::square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| square_ | `aten::square_(Tensor(a!) self) -> Tensor(a!)` | false |
| squeeze | `aten::squeeze(Tensor(a) self) -> Tensor(a)` | false |
| squeeze | `aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)` | false |
| squeeze | `aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)` | false |
| squeeze | `aten::squeeze.dimname(Tensor(a) self, str dim) -> Tensor(a)` | false |
| squeeze_copy | `aten::squeeze_copy(Tensor self) -> Tensor` | false |
| squeeze_copy | `aten::squeeze_copy.dim(Tensor self, int dim) -> Tensor` | false |
| squeeze_copy | `aten::squeeze_copy.dims(Tensor self, int[] dim) -> Tensor` | false |
| squeeze_copy | `aten::squeeze_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| squeeze_copy | `aten::squeeze_copy.dim_out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)` | false |
| squeeze_copy | `aten::squeeze_copy.dims_out(Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!)` | false |
| stack | `aten::stack(Tensor[] tensors, int dim=0) -> Tensor` | false |
| stack | `aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| std | `aten::std(Tensor self, bool unbiased=True) -> Tensor` | false |
| std | `aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor` | false |
| std | `aten::std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor` | false |
| std | `aten::std.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor` | false |
| std | `aten::std.names_out(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| std | `aten::std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| std | `aten::std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)` | false |
| std | `aten::std.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor` | false |
| std | `aten::std.correction_names_out(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)` | false |
| std_mean | `aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)` | false |
| std_mean | `aten::std_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)` | false |
| std_mean | `aten::std_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)` | false |
| std_mean | `aten::std_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)` | false |
| std_mean | `aten::std_mean.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)` | false |
| std_mean | `aten::std_mean.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| stft | `aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None) -> Tensor` | false |
| stft | `aten::stft.center(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, str pad_mode="reflect", bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None) -> Tensor` | false |
| storage_offset | `aten::storage_offset(Tensor self) -> int` | false |
| stride | `aten::stride.int(Tensor self, int dim) -> int` | false |
| stride | `aten::stride.Dimname(Tensor self, str dim) -> int` | false |
| stride | `aten::stride(Tensor self) -> int[]` | false |
| sub | `aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor` | false |
| sub | `aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor` | false |
| sub | `aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| sub | `aten::sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| sub | `aten::sub.int(int a, int b) -> int` | false |
| sub | `aten::sub.complex(complex a, complex b) -> complex` | false |
| sub | `aten::sub.float(float a, float b) -> float` | false |
| sub | `aten::sub.int_complex(int a, complex b) -> complex` | false |
| sub | `aten::sub.complex_int(complex a, int b) -> complex` | false |
| sub | `aten::sub.float_complex(float a, complex b) -> complex` | false |
| sub | `aten::sub.complex_float(complex a, float b) -> complex` | false |
| sub | `aten::sub.int_float(int a, float b) -> float` | false |
| sub | `aten::sub.float_int(float a, int b) -> float` | false |
| sub | `aten::sub(Scalar a, Scalar b) -> Scalar` | false |
| sub_ | `aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)` | false |
| sub_ | `aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)` | false |
| subtract | `aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor` | false |
| subtract | `aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)` | false |
| subtract | `aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor` | false |
| subtract_ | `aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)` | false |
| subtract_ | `aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)` | false |
| sum | `aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| sum | `aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor` | false |
| sum | `aten::sum.dim_DimnameList(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor` | false |
| sum | `aten::sum.DimnameList_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| sum | `aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| sum | `aten::sum.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)` | false |
| sum | `aten::sum.int(int[] self) -> int` | false |
| sum | `aten::sum.float(float[] self) -> float` | false |
| sum | `aten::sum.complex(complex[] self) -> complex` | false |
| sum | `aten::sum.bool(bool[] self) -> int` | false |
| svd | `aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)` | false |
| svd | `aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)` | false |
| sym_constrain_range | `aten::sym_constrain_range(Scalar size, *, int? min=None, int? max=None) -> ()` | false |
| sym_constrain_range_for_size | `aten::sym_constrain_range_for_size(Scalar size, *, int? min=None, int? max=None) -> ()` | false |
| sym_numel | `aten::sym_numel(Tensor self) -> SymInt` | false |
| sym_size | `aten::sym_size.int(Tensor self, int dim) -> SymInt` | false |
| sym_size | `aten::sym_size(Tensor self) -> SymInt[]` | false |
| sym_storage_offset | `aten::sym_storage_offset(Tensor self) -> SymInt` | false |
| sym_stride | `aten::sym_stride.int(Tensor self, int dim) -> SymInt` | false |
| sym_stride | `aten::sym_stride(Tensor self) -> SymInt[]` | false |
| t | `aten::t(Tensor(a) self) -> Tensor(a)` | false |
| t_ | `aten::t_(Tensor(a!) self) -> Tensor(a!)` | false |
| t_copy | `aten::t_copy(Tensor self) -> Tensor` | false |
| t_copy | `aten::t_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| take | `aten::take(Tensor self, Tensor index) -> Tensor` | false |
| take | `aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)` | false |
| tan | `aten::tan(Tensor self) -> Tensor` | false |
| tan | `aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| tan | `aten::tan.int(int a) -> float` | false |
| tan | `aten::tan.float(float a) -> float` | false |
| tan | `aten::tan.complex(complex a) -> complex` | false |
| tan | `aten::tan.Scalar(Scalar a) -> Scalar` | false |
| tan_ | `aten::tan_(Tensor(a!) self) -> Tensor(a!)` | false |
| tanh | `aten::tanh(Tensor self) -> Tensor` | false |
| tanh | `aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| tanh | `aten::tanh.int(int a) -> float` | false |
| tanh | `aten::tanh.float(float a) -> float` | false |
| tanh | `aten::tanh.complex(complex a) -> complex` | false |
| tanh | `aten::tanh.Scalar(Scalar a) -> Scalar` | false |
| tanh_ | `aten::tanh_(Tensor(a!) self) -> Tensor(a!)` | false |
| tanh_backward | `aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor` | false |
| tanh_backward | `aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| tensor_split | `aten::tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]` | false |
| tensor_split | `aten::tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]` | false |
| tensor_split | `aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]` | false |
| threshold | `aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor` | false |
| threshold | `aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)` | false |
| threshold_ | `aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)` | false |
| threshold_backward | `aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor` | false |
| threshold_backward | `aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| to | `aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)` | false |
| to | `aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)` | false |
| to | `aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)` | false |
| to | `aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)` | false |
| to | `aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(b\|a)` | false |
| to | `aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(b\|a)` | false |
| to | `aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(b\|a)` | false |
| topk | `aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)` | false |
| topk | `aten::topk.values(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)` | false |
| trace | `aten::trace(Tensor self) -> Tensor` | false |
| trace | `aten::trace.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| transpose | `aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)` | false |
| transpose | `aten::transpose.Dimname(Tensor(a) self, str dim0, str dim1) -> Tensor(a)` | false |
| transpose_ | `aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)` | false |
| transpose_copy | `aten::transpose_copy.int(Tensor self, int dim0, int dim1) -> Tensor` | false |
| transpose_copy | `aten::transpose_copy.int_out(Tensor self, int dim0, int dim1, *, Tensor(a!) out) -> Tensor(a!)` | false |
| triangular_solve | `aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)` | false |
| triangular_solve | `aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)` | false |
| tril | `aten::tril(Tensor self, int diagonal=0) -> Tensor` | false |
| tril | `aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| tril_ | `aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)` | false |
| tril_indices | `aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| tril_indices | `aten::tril_indices.out(int row, int col, int offset=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| triu | `aten::triu(Tensor self, int diagonal=0) -> Tensor` | false |
| triu | `aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| triu_ | `aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)` | false |
| triu_indices | `aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| triu_indices | `aten::triu_indices.out(int row, int col, int offset=0, *, Tensor(a!) out) -> Tensor(a!)` | false |
| true_divide | `aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| true_divide | `aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor` | false |
| true_divide | `aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| true_divide_ | `aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| true_divide_ | `aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| trunc | `aten::trunc(Tensor self) -> Tensor` | false |
| trunc | `aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| trunc_ | `aten::trunc_(Tensor(a!) self) -> Tensor(a!)` | false |
| unbind | `aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]` | false |
| unbind | `aten::unbind.Dimname(Tensor(a -> *) self, str dim) -> Tensor(a)[]` | false |
| unbind_copy | `aten::unbind_copy.int(Tensor self, int dim=0) -> Tensor[]` | false |
| unbind_copy | `aten::unbind_copy.int_out(Tensor self, int dim=0, *, Tensor(a!)[] out) -> ()` | false |
| unfold | `aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)` | false |
| unfold_backward | `aten::unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor` | false |
| unfold_backward | `aten::unfold_backward.out(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step, *, Tensor(a!) out) -> Tensor(a!)` | false |
| unfold_copy | `aten::unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor` | false |
| unfold_copy | `aten::unfold_copy.out(Tensor self, int dimension, int size, int step, *, Tensor(a!) out) -> Tensor(a!)` | false |
| uniform | `aten::uniform(Tensor self, float from=0., float to=1., *, Generator? generator=None) -> Tensor` | false |
| uniform | `aten::uniform.out(Tensor self, float from=0., float to=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)` | false |
| uniform_ | `aten::uniform_(Tensor(a!) self, float from=0., float to=1., *, Generator? generator=None) -> Tensor(a!)` | false |
| unique_consecutive | `aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)` | false |
| unique_consecutive | `aten::unique_consecutive.out(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| unique_dim | `aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)` | false |
| unique_dim | `aten::unique_dim.out(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))` | false |
| unsafe_chunk | `aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]` | false |
| unsafe_split | `aten::unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]` | false |
| unsafe_split | `aten::unsafe_split.Tensor_out(Tensor self, SymInt split_size, int dim=0, *, Tensor(a!)[] out) -> ()` | false |
| unsafe_split_with_sizes | `aten::unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]` | false |
| unsafe_split_with_sizes | `aten::unsafe_split_with_sizes.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()` | false |
| unsqueeze | `aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)` | false |
| unsqueeze_ | `aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)` | false |
| unsqueeze_copy | `aten::unsqueeze_copy(Tensor self, int dim) -> Tensor` | false |
| unsqueeze_copy | `aten::unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_bicubic2d | `aten::upsample_bicubic2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| upsample_bicubic2d | `aten::upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor` | false |
| upsample_bicubic2d | `aten::upsample_bicubic2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_bilinear2d | `aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| upsample_bilinear2d | `aten::upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor` | false |
| upsample_bilinear2d | `aten::upsample_bilinear2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_bilinear2d | `aten::upsample_bilinear2d.vec_out(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_linear1d | `aten::upsample_linear1d(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None) -> Tensor` | false |
| upsample_linear1d | `aten::upsample_linear1d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor` | false |
| upsample_linear1d | `aten::upsample_linear1d.out(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_nearest1d | `aten::upsample_nearest1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor` | false |
| upsample_nearest1d | `aten::upsample_nearest1d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor` | false |
| upsample_nearest1d | `aten::upsample_nearest1d.out(Tensor self, SymInt[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_nearest2d | `aten::upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| upsample_nearest2d | `aten::upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor` | false |
| upsample_nearest2d | `aten::upsample_nearest2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_nearest2d | `aten::upsample_nearest2d.vec_out(Tensor input, SymInt[]? output_size, float[]? scale_factors, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_nearest2d_backward | `aten::upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| upsample_nearest2d_backward | `aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)` | false |
| upsample_nearest3d | `aten::upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| upsample_nearest3d | `aten::upsample_nearest3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor` | false |
| upsample_nearest3d | `aten::upsample_nearest3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| upsample_trilinear3d | `aten::upsample_trilinear3d(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor` | false |
| upsample_trilinear3d | `aten::upsample_trilinear3d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor` | false |
| upsample_trilinear3d | `aten::upsample_trilinear3d.out(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)` | false |
| var | `aten::var(Tensor self, bool unbiased=True) -> Tensor` | false |
| var | `aten::var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor` | false |
| var | `aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor` | false |
| var | `aten::var.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor` | false |
| var | `aten::var.names_out(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| var | `aten::var.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)` | false |
| var | `aten::var.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)` | false |
| var | `aten::var.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor` | false |
| var | `aten::var.correction_names_out(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)` | false |
| var_mean | `aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)` | false |
| var_mean | `aten::var_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)` | false |
| var_mean | `aten::var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)` | false |
| var_mean | `aten::var_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)` | false |
| var_mean | `aten::var_mean.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)` | false |
| var_mean | `aten::var_mean.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))` | false |
| vdot | `aten::vdot(Tensor self, Tensor other) -> Tensor` | false |
| vdot | `aten::vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| view | `aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)` | false |
| view | `aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)` | false |
| view_as_complex | `aten::view_as_complex(Tensor(a) self) -> Tensor(a)` | false |
| view_as_real | `aten::view_as_real(Tensor(a) self) -> Tensor(a)` | false |
| view_copy | `aten::view_copy(Tensor self, SymInt[] size) -> Tensor` | false |
| view_copy | `aten::view_copy.dtype(Tensor self, ScalarType dtype) -> Tensor` | false |
| view_copy | `aten::view_copy.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| view_copy | `aten::view_copy.dtype_out(Tensor self, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)` | false |
| where | `aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor` | false |
| where | `aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor` | false |
| where | `aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor` | false |
| where | `aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor` | false |
| where | `aten::where(Tensor condition) -> Tensor[]` | false |
| where | `aten::where.self_out(Tensor condition, Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| xlogy | `aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor` | false |
| xlogy | `aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor` | false |
| xlogy | `aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor` | false |
| xlogy | `aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| xlogy | `aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| xlogy | `aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)` | false |
| xlogy_ | `aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` | false |
| xlogy_ | `aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)` | false |
| zero | `aten::zero(Tensor self) -> Tensor` | false |
| zero | `aten::zero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` | false |
| zero_ | `aten::zero_(Tensor(a!) self) -> Tensor(a!)` | false |
| zeros | `aten::zeros.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| zeros | `aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor` | false |
| zeros | `aten::zeros.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)` | false |
| zeros | `aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)` | false |
| zeros_like | `aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor` | false |
| zeros_like | `aten::zeros_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)` | false |
