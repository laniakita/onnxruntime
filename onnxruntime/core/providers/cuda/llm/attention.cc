// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/attention.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cuda/llm/attention.h"
#include "core/providers/cuda/llm/attention_mask_impl.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "core/providers/cuda/cuda_type_conversion.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                            \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      23,                                                             \
      23,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#define REGISTER_KERNEL_TYPED_24(T)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      24,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED_24(float)
REGISTER_KERNEL_TYPED_24(MLFloat16)
REGISTER_KERNEL_TYPED_24(BFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode_ = info.node().OutputDefs().size() >= 4 && info.node().OutputDefs()[3]->Exists()
                               ? static_cast<attention_helper::QKMatMulOutputMode>(mode)
                               : attention_helper::QKMatMulOutputMode::kNone;
  ORT_ENFORCE(qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKMask ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftCap ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftMax,
              "qk_matmul_output_mode must be 0, 1, 2, or 3.");
  scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<T>::quiet_NaN());
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  softmax_precision_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
  ORT_ENFORCE(scale_ > 0 || std::isnan(scale_), "scale must be greater than 0 if specified");
}

// ============================================================================
// RunFlashAttention: Direct flash attention kernel call
// ============================================================================
template <typename T>
Status Attention<T>::RunFlashAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    const attention_helper::AttentionParameters& parameters) const {
#if USE_FLASH_ATTENTION
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
  const bool is_bf16 = std::is_same<T, BFloat16>::value;
  const bool is_bsnh = parameters.transpose_output;  // 3D inputs → BSNH

  // Allocate softmax_lse and accumulation buffers
  size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(
      parameters.q_sequence_length, parameters.batch_size, parameters.q_num_heads);

  auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      onnxruntime::flash::get_num_splits_and_buffer_sizes(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.total_sequence_length, parameters.q_num_heads,
          parameters.head_size, device_prop.multiProcessorCount);

  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());

  if (softmax_lse_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(softmax_lse_accum_buffer.get(), 0,
                                         softmax_lse_accum_bytes, cuda_stream));
  }
  if (out_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(out_accum_buffer.get(), 0,
                                         out_accum_bytes, cuda_stream));
  }

  // Handle nonpad_kv_seqlen: external KV cache path (opset 24)
  if (nonpad_kv_seqlen != nullptr) {
    ORT_ENFORCE(parameters.past_sequence_length == 0,
                "RunFlashAttention with nonpad_kv_seqlen requires K/V to be the full cache "
                "(past_sequence_length must be 0, got ", parameters.past_sequence_length, ").");

    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
        nonpad_kv_seqlen->Data<int64_t>(),
        seqlens_k_buffer.get(),
        parameters.batch_size,
        parameters.total_sequence_length,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    // K/V are the full cache in BSNH. No new tokens to append (k=nullptr, v=nullptr).
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
        device_prop, cuda_stream,
        const_cast<void*>(static_cast<const void*>(Q->Data<T>())),
        const_cast<void*>(static_cast<const void*>(K->Data<T>())),
        const_cast<void*>(static_cast<const void*>(V->Data<T>())),
        /*k=*/nullptr, /*v=*/nullptr,
        static_cast<void*>(Y->MutableData<T>()),
        softmax_lse_buffer.get(),
        const_cast<void*>(static_cast<const void*>(seqlens_k_buffer.get())),
        /*rotary_cos=*/nullptr, /*rotary_sin=*/nullptr,
        /*cache_batch_idx=*/nullptr, /*leftpad_k=*/nullptr,
        /*head_sink=*/nullptr, /*block_table=*/nullptr,
        parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
        parameters.head_size,
        parameters.q_sequence_length, parameters.kv_sequence_length,
        /*seqlen_k_new=*/0, /*rotary_dim=*/0,
        parameters.scale, parameters.softcap,
        parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
        /*past_bsnh=*/is_bsnh,
        static_cast<int>(num_splits),
        softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
        /*local_window_size=*/-1, /*is_rotary_interleaved=*/false,
        /*is_packed_qkv=*/false));

    // Populate present_key/value (BNSH) from external cache K/V (BSNH)
    if (present_key != nullptr && is_bsnh) {
      if constexpr (std::is_same_v<T, MLFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            reinterpret_cast<const half*>(K->Data<T>()),
            reinterpret_cast<half*>(present_key->MutableData<T>()),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else if constexpr (std::is_same_v<T, BFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            K->Data<BFloat16>(), present_key->MutableData<BFloat16>(),
            cuda_stream, device_prop.maxThreadsPerBlock));
      }
    }
    if (present_value != nullptr && is_bsnh) {
      if constexpr (std::is_same_v<T, MLFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            reinterpret_cast<const half*>(V->Data<T>()),
            reinterpret_cast<half*>(present_value->MutableData<T>()),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else if constexpr (std::is_same_v<T, BFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            V->Data<BFloat16>(), present_value->MutableData<BFloat16>(),
            cuda_stream, device_prop.maxThreadsPerBlock));
      }
    }
    return Status::OK();
  }

  // Note: Flash with past_key is excluded by flash_eligible (requires past_key == nullptr).
  // Those cases fall through to unfused attention which handles past concatenation.

  // No past, no nonpad: prompt-only flash attention
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
      device_prop, cuda_stream,
      const_cast<void*>(static_cast<const void*>(Q->Data<T>())),
      const_cast<void*>(static_cast<const void*>(K->Data<T>())),
      const_cast<void*>(static_cast<const void*>(V->Data<T>())),
      static_cast<void*>(Y->MutableData<T>()),
      softmax_lse_buffer.get(),
      parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
      parameters.head_size,
      parameters.q_sequence_length, parameters.kv_sequence_length,
      parameters.scale, parameters.softcap,
      parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
      static_cast<int>(num_splits),
      softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
      is_bsnh));

      // Populate present_key/present_value (BNSH) from K/V (BSNH) for no-past case
  if (present_key != nullptr && is_bsnh) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          reinterpret_cast<const half*>(K->Data<T>()),
          reinterpret_cast<half*>(present_key->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          K->Data<BFloat16>(), present_key->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }
  if (present_value != nullptr && is_bsnh) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          reinterpret_cast<const half*>(V->Data<T>()),
          reinterpret_cast<half*>(present_value->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          V->Data<BFloat16>(), present_value->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(Q);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(V);
  ORT_UNUSED_PARAMETER(attn_mask);
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  ORT_UNUSED_PARAMETER(nonpad_kv_seqlen);
  ORT_UNUSED_PARAMETER(Y);
  ORT_UNUSED_PARAMETER(present_key);
  ORT_UNUSED_PARAMETER(present_value);
  ORT_UNUSED_PARAMETER(parameters);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Flash attention is not available in this build.");
#endif
}

// ============================================================================
// RunMemoryEfficientAttention: Direct memory-efficient attention kernel call
// ============================================================================
template <typename T>
Status Attention<T>::RunMemoryEfficientAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    const attention_helper::AttentionParameters& parameters) const {
#if USE_MEMORY_EFFICIENT_ATTENTION
  typedef typename ToCudaType<T>::MappedType CudaT;
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
  const bool is_bsnh = parameters.transpose_output;
  const int sm = device_prop.major * 10 + device_prop.minor;

  // Q/K/V pointers — MEA expects BSNH format
  const void* q_data = Q->Data<T>();
  const void* k_data = K->Data<T>();
  const void* v_data = V->Data<T>();

  // Note: MEA with past_key/value is handled by the unfused fallback.
  // The cascade in ComputeInternal ensures past_key == nullptr when we reach here.

  // Handle attention mask → attention_bias conversion
  IAllocatorUniquePtr<void> converted_mask_buffer;
  IAllocatorUniquePtr<void> nonpad_bias_buffer;
  const void* attn_bias_data = nullptr;
  bool broadcast_bias_dim_0 = false;
  bool broadcast_bias_dim_1 = false;

  if (nonpad_kv_seqlen != nullptr) {
    // Convert nonpad_kv_seqlen to seqlens_k for custom right padding.
    // MEA expects actual token count (not count-1), so use FlashSeqlensK variant.
    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
        nonpad_kv_seqlen->Data<int64_t>(),
        seqlens_k_buffer.get(),
        parameters.batch_size,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    onnxruntime::contrib::cuda::MemoryEfficientAttentionParams p;
    p.sm = sm;
    p.is_half = std::is_same<T, MLFloat16>::value;
    p.is_bf16 = std::is_same<T, BFloat16>::value;
    p.is_kv_bsnh = is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    p.scale = parameters.scale;
    p.seqlen_k_ptr = seqlens_k_buffer.get();
    p.has_custom_right_padding = true;
    p.query = q_data;
    p.key = k_data;
    p.value = v_data;
    p.attn_bias = nullptr;
    p.stream = cuda_stream;
    p.output = Y->MutableData<T>();

    IAllocatorUniquePtr<void> workspace_buffer;
    if (onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
            parameters.v_head_size, sizeof(T) == sizeof(float))) {
      size_t workspace_bytes = sizeof(float) * parameters.batch_size * parameters.q_sequence_length *
                               parameters.q_num_heads * parameters.v_head_size;
      workspace_buffer = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());
      p.workspace = workspace_buffer.get();
    } else {
      p.workspace = nullptr;
    }
    onnxruntime::contrib::cuda::run_memory_efficient_attention(p);
  } else {
    // Standard MEA path (no nonpad)
    if (attn_mask != nullptr) {
      if (attn_mask->IsDataType<bool>()) {
        using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
        int64_t num_elements = attn_mask->Shape().Size();
        converted_mask_buffer = GetScratchBuffer<void>(
            num_elements * sizeof(NativeCudaT), context->GetComputeStream());
        float mask_filter_value = static_cast<float>(std::numeric_limits<T>::lowest());
        ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
            attn_mask->Data<bool>(),
            reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
            num_elements, mask_filter_value, cuda_stream,
            device_prop.maxThreadsPerBlock));
        attn_bias_data = converted_mask_buffer.get();
      } else {
        attn_bias_data = attn_mask->Data<T>();
      }

      // Determine broadcast flags
      size_t mask_dims = attn_mask->Shape().NumDimensions();
      auto dims = attn_mask->Shape().GetDims();
      if (mask_dims == 2) {
        broadcast_bias_dim_0 = true;
        broadcast_bias_dim_1 = true;
      } else if (mask_dims == 3) {
        broadcast_bias_dim_0 = true;
        broadcast_bias_dim_1 = dims[0] == 1;
      } else {
        broadcast_bias_dim_0 = dims[0] == 1;
        broadcast_bias_dim_1 = dims[1] == 1;
      }
    }

    onnxruntime::contrib::cuda::MemoryEfficientAttentionParams p;
    p.sm = sm;
    p.is_half = std::is_same<T, MLFloat16>::value;
    p.is_bf16 = std::is_same<T, BFloat16>::value;
    p.is_kv_bsnh = is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    p.scale = parameters.scale;
    p.broadcast_attn_bias_dim_0 = broadcast_bias_dim_0;
    p.broadcast_attn_bias_dim_1 = broadcast_bias_dim_1;
    p.query = q_data;
    p.key = k_data;
    p.value = v_data;
    p.attn_bias = attn_bias_data;
    p.stream = cuda_stream;
    p.output = Y->MutableData<T>();

    if (onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
            parameters.v_head_size, sizeof(T) == sizeof(float))) {
      size_t workspace_bytes = sizeof(float) * parameters.batch_size * parameters.q_sequence_length *
                               parameters.q_num_heads * parameters.v_head_size;
      auto workspace_buffer = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());
      p.workspace = workspace_buffer.get();
      onnxruntime::contrib::cuda::run_memory_efficient_attention(p);
    } else {
      p.workspace = nullptr;
      onnxruntime::contrib::cuda::run_memory_efficient_attention(p);
    }
  }

  // Populate present_key/present_value (BNSH) if requested
  if (present_key != nullptr && is_bsnh) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          reinterpret_cast<const half*>(K->Data<T>()),
          reinterpret_cast<half*>(present_key->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          K->Data<BFloat16>(), present_key->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          K->Data<float>(), present_key->MutableData<float>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }
  if (present_value != nullptr && is_bsnh) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          reinterpret_cast<const half*>(V->Data<T>()),
          reinterpret_cast<half*>(present_value->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          V->Data<BFloat16>(), present_value->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          V->Data<float>(), present_value->MutableData<float>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(Q);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(V);
  ORT_UNUSED_PARAMETER(attn_mask);
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  ORT_UNUSED_PARAMETER(nonpad_kv_seqlen);
  ORT_UNUSED_PARAMETER(Y);
  ORT_UNUSED_PARAMETER(present_key);
  ORT_UNUSED_PARAMETER(present_value);
  ORT_UNUSED_PARAMETER(parameters);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Memory efficient attention is not available in this build.");
#endif
}

// ============================================================================
// RunUnfusedAttention: Delegates to MHA's QkvToContext (unfused GEMM+softmax+GEMM)
// ============================================================================
template <typename T>
Status Attention<T>::RunUnfusedAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    Tensor* output_qk,
    const attention_helper::AttentionParameters& parameters) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

  // Bridge to contrib::AttentionParameters for the MHA unfused path
  onnxruntime::contrib::AttentionParameters contribop_parameters;

  if (!parameters.transpose_output) {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    contribop_parameters.is_output_bnsh = true;
  } else {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH;
    contribop_parameters.is_output_bnsh = false;
  }

  contribop_parameters.batch_size = parameters.batch_size;
  contribop_parameters.sequence_length = parameters.q_sequence_length;
  contribop_parameters.kv_sequence_length = parameters.kv_sequence_length;
  contribop_parameters.past_sequence_length = parameters.past_sequence_length;
  contribop_parameters.total_sequence_length = parameters.total_sequence_length;
  contribop_parameters.max_sequence_length = parameters.total_sequence_length;
  contribop_parameters.input_hidden_size = 0;
  contribop_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
  contribop_parameters.head_size = parameters.head_size;
  contribop_parameters.v_head_size = parameters.v_head_size;
  contribop_parameters.v_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
  contribop_parameters.num_heads = parameters.q_num_heads;
  contribop_parameters.rotary_dim = 0;
  contribop_parameters.num_splits = 1;
  contribop_parameters.beam_width = 1;
  contribop_parameters.is_unidirectional = parameters.is_causal;
  contribop_parameters.past_present_share_buffer = false;
  contribop_parameters.is_packed_qkv = false;
  contribop_parameters.do_rotary = false;
  contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;
  contribop_parameters.mask_filter_value = static_cast<float>(std::numeric_limits<T>::lowest());
  contribop_parameters.scale = parameters.scale;
  contribop_parameters.use_tf32 = UseTF32();

  // Determine broadcast flags for attention_bias
  if (attn_mask != nullptr) {
    size_t attn_mask_dims_size = attn_mask->Shape().NumDimensions();
    auto attn_mask_dims = attn_mask->Shape().GetDims();
    if (attn_mask_dims_size == 2) {
      contribop_parameters.broadcast_attn_bias_dim_0 = true;
      contribop_parameters.broadcast_attn_bias_dim_1 = true;
    } else if (attn_mask_dims_size == 3) {
      contribop_parameters.broadcast_attn_bias_dim_0 = true;
      contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[0] == 1;
    } else {
      contribop_parameters.broadcast_attn_bias_dim_0 = attn_mask_dims[0] == 1;
      contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[1] == 1;
    }
  } else {
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = false;
  }

  // Construct AttentionData
  onnxruntime::contrib::cuda::AttentionData<CudaT> data;
  data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
  data.mask_index = nullptr;
  data.mask_index_dims = gsl::span<const int64_t>();
  data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  data.present_key = (present_key == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (present_value == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  if (output_qk != nullptr) {
    data.output_qk = reinterpret_cast<CudaT*>(output_qk->MutableData<T>());
  }
  data.bias = nullptr;

  // Handle attention mask / nonpad_kv_seqlen → attention_bias
  IAllocatorUniquePtr<void> converted_mask_buffer;
  if (nonpad_kv_seqlen != nullptr) {
    // Convert nonpad_kv_seqlen to additive attention bias: [B, q_seq, total_seq]
    using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
    int64_t bias_elements = static_cast<int64_t>(parameters.batch_size) *
                            parameters.q_sequence_length *
                            parameters.total_sequence_length;
    converted_mask_buffer = GetScratchBuffer<void>(bias_elements * sizeof(NativeCudaT), context->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToAttentionBias<NativeCudaT>(
        nonpad_kv_seqlen->Data<int64_t>(),
        reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
        parameters.batch_size,
        parameters.q_sequence_length,
        parameters.total_sequence_length,
        contribop_parameters.mask_filter_value,
        cuda_stream,
        device_prop.maxThreadsPerBlock));
    data.attention_bias = reinterpret_cast<const CudaT*>(converted_mask_buffer.get());
    // nonpad bias is [B, q_seq, total_seq] → broadcasts over heads but not batch
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = true;
  } else if (attn_mask != nullptr) {
    if (attn_mask->IsDataType<bool>()) {
      using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
      int64_t num_elements = attn_mask->Shape().Size();
      converted_mask_buffer = GetScratchBuffer<void>(num_elements * sizeof(NativeCudaT), context->GetComputeStream());
      ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
          attn_mask->Data<bool>(),
          reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
          num_elements,
          contribop_parameters.mask_filter_value,
          cuda_stream,
          device_prop.maxThreadsPerBlock));
      data.attention_bias = reinterpret_cast<const CudaT*>(converted_mask_buffer.get());
    } else {
      data.attention_bias = reinterpret_cast<const CudaT*>(attn_mask->Data<T>());
    }
  }

  data.qkv_format = contribop_parameters.qkv_format;
  data.use_flash_attention = false;
  data.use_memory_efficient_attention = false;
  data.fused_runner = nullptr;
  data.fused_cross_attention_kernel = nullptr;
  data.kernel_type = onnxruntime::contrib::AttentionKernelType::AttentionKernel_Unfused;

  // Allocate workspace
  const bool no_qkv_workspace = onnxruntime::contrib::cuda::NoQkvWorkspace(contribop_parameters, data);
  size_t workspace_bytes = onnxruntime::contrib::cuda::GetAttentionWorkspaceSize(
      sizeof(T),
      contribop_parameters.batch_size,
      contribop_parameters.num_heads,
      contribop_parameters.head_size,
      contribop_parameters.v_head_size,
      contribop_parameters.sequence_length,
      contribop_parameters.kv_sequence_length,
      contribop_parameters.total_sequence_length,
      nullptr, false, false, false, false, false,
      no_qkv_workspace);
  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workspace_bytes;

  cublasHandle_t cublas = GetCublasHandle(context);
  cudnnHandle_t cudnn = GetCudnnHandle(context);

  return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
      device_prop, cublas, cudnn, context->GetComputeStream(), contribop_parameters, data);
}

// ============================================================================
// ComputeInternal: Dispatch to appropriate attention kernel
// ============================================================================
// MHA path (q_num_heads == kv_num_heads): uses direct kernel dispatch cascade
//   flash → memory efficient → unfused
// GQA path (q_num_heads != kv_num_heads): routes through GQA dispatch (kept for now)
// ============================================================================
template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  const Tensor* attn_mask = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);
  const Tensor* past_value = context->Input<Tensor>(5);
  const Tensor* nonpad_kv_seqlen = context->Input<Tensor>(6);  // optional, Opset 24

  attention_helper::AttentionParameters parameters;
  TensorShape y_shape;
  TensorShape present_key_shape;
  TensorShape present_value_shape;
  TensorShape output_qk_shape;

  ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
                  Q, K, V, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                  is_causal_, softcap_, softmax_precision_,
                  qk_matmul_output_mode_, kv_num_heads_, q_num_heads_, scale_,
                  parameters, y_shape, present_key_shape, present_value_shape, output_qk_shape,
                  true /* skip_nonpad_data_validation: data is on GPU */)
                  .IsOK(),
              "Output shapes for Attention could not be computed.");

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = context->Output(3, output_qk_shape);

  const bool is_gqa = parameters.kv_num_heads != parameters.q_num_heads;

  if (is_gqa) {
    // GQA path: routes through existing GQA dispatch kernel.
    // Will be replaced with the direct dispatch cascade in a follow-up.
    return ComputeGQA(context, Q, K, V, attn_mask, past_key, past_value,
                      nonpad_kv_seqlen, Y, present_key, present_value, parameters);
  }

  // === MHA KERNEL SELECTION CASCADE ===
  // Priority: flash attention > memory efficient attention > unfused attention
  const bool has_output_qk = (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone);

#if USE_FLASH_ATTENTION
  {
    auto& device_prop = GetDeviceProp();
    bool flash_eligible =
        !std::is_same<T, float>::value &&
        onnxruntime::flash::is_supported<T>(device_prop, parameters.head_size,
                                            parameters.q_num_heads, parameters.kv_num_heads) &&
        parameters.head_size == parameters.v_head_size &&
        !has_output_qk &&
        parameters.softcap == 0.0f &&
        parameters.softmax_precision == 0 &&
        past_key == nullptr &&  // Flash with past requires buffer management; use unfused
        attn_mask == nullptr;   // Flash prompt path does not support attention mask

    if (flash_eligible) {
      return RunFlashAttention(context, Q, K, V, attn_mask, past_key, past_value,
                               nonpad_kv_seqlen, Y, present_key, present_value, parameters);
    }
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  {
    auto& device_prop = GetDeviceProp();
    int sm = device_prop.major * 10 + device_prop.minor;
    bool mea_eligible =
        onnxruntime::contrib::cuda::has_memory_efficient_attention(
            sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value,
            parameters.head_size, parameters.v_head_size) &&
        !has_output_qk &&
        parameters.softcap == 0.0f &&
        parameters.softmax_precision == 0 &&
        past_key == nullptr;

    if (mea_eligible) {
      return RunMemoryEfficientAttention(context, Q, K, V, attn_mask, past_key, past_value,
                                         nonpad_kv_seqlen, Y, present_key, present_value, parameters);
    }
  }
#endif

  // Fallback: unfused attention
  if (parameters.softcap != 0.0f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "softcap is not supported yet in Attention op (CUDA).");
  }
  if (parameters.softmax_precision != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "softmax_precision is not supported yet in Attention op (CUDA).");
  }
  if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone &&
      qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kQK) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "qk_matmul_output_mode other than kNone and kQK is not supported yet "
                           "in Attention op (CUDA).");
  }

  return RunUnfusedAttention(context, Q, K, V, attn_mask, past_key, past_value,
                             nonpad_kv_seqlen, Y, present_key, present_value, output_qk, parameters);
}

// ============================================================================
// ComputeGQA: Preserved GQA dispatch path (routes through contrib GQA kernel)
// ============================================================================
template <typename T>
Status Attention<T>::ComputeGQA(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    const attention_helper::AttentionParameters& parameters) const {
  // GQA path does not support nonpad_kv_seqlen (opset 24 inplace KV cache).
  // Fail loudly rather than silently producing wrong results.
  ORT_ENFORCE(nonpad_kv_seqlen == nullptr,
              "nonpad_kv_seqlen is not supported in the GQA path of Attention op. "
              "Use flash or memory efficient attention instead.");
  // GQA only supports float16 and bfloat16 types
  if constexpr (std::is_same<T, float>::value) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GQA in Attention op (CUDA) does not support float32. "
                           "Please use float16 or bfloat16.");
  } else {
    // GQA only supports 3D inputs (BSNH), not 4D (BNSH)
    if (!parameters.transpose_output) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "4D QKV inputs (BNSH format) are not supported yet in GQA path of Attention op (CUDA). "
                             "Please use 3D inputs (B, S, hidden_size) instead.");
    }
    if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "qk_matmul_output_mode is not supported yet in GQA path of Attention op (CUDA).");
    }
    if (parameters.softmax_precision != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "softmax_precision is not supported yet in GQA path of Attention op (CUDA).");
    }
    if (!parameters.is_causal) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Non-causal attention is not supported yet in GQA path of Attention op (CUDA).");
    }
    if (parameters.kv_sequence_length != parameters.q_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Cross-attention (kv_sequence_length != q_sequence_length) is not supported in "
                             "GQA path of Attention op (CUDA). kv_sequence_length=",
                             parameters.kv_sequence_length,
                             ", q_sequence_length=", parameters.q_sequence_length);
    }

    auto& device_prop = GetDeviceProp();

    // Bridge to GroupQueryAttentionParameters
    onnxruntime::contrib::GroupQueryAttentionParameters gqa_parameters;
    gqa_parameters.batch_size = parameters.batch_size;
    gqa_parameters.sequence_length = parameters.q_sequence_length;
    gqa_parameters.seqlen_past_kv_cache = parameters.past_sequence_length;
    gqa_parameters.seqlen_present_kv_cache = parameters.total_sequence_length;
    gqa_parameters.total_sequence_length = parameters.total_sequence_length;
    gqa_parameters.kv_sequence_length = parameters.kv_sequence_length;
    gqa_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
    gqa_parameters.num_heads = parameters.q_num_heads;
    gqa_parameters.head_size = parameters.head_size;
    gqa_parameters.v_head_size = parameters.v_head_size;
    gqa_parameters.kv_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
    gqa_parameters.kv_num_heads = parameters.kv_num_heads;
    gqa_parameters.scale = parameters.scale;
    gqa_parameters.softcap = parameters.softcap;
    gqa_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH;
    gqa_parameters.rotary_dim = 0;
    gqa_parameters.is_unidirectional = true;
    gqa_parameters.is_packed_qkv = false;
    gqa_parameters.is_subsequent_prompt = false;
    gqa_parameters.is_first_prompt = parameters.past_sequence_length == 0;
    gqa_parameters.do_rotary = false;
    gqa_parameters.rotary_interleaved = false;
    gqa_parameters.use_smooth_softmax = false;
    gqa_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;
    gqa_parameters.past_kv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    gqa_parameters.local_window_size = -1;
    gqa_parameters.zeros_count = 0;
    gqa_parameters.zero_ptr = nullptr;
    gqa_parameters.num_splits = 1;

    typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
    onnxruntime::contrib::cuda::GroupQueryAttentionData<CudaT, CudaT> gqa_data;

    IAllocatorUniquePtr<void> k_buffer;
    IAllocatorUniquePtr<void> v_buffer;
    IAllocatorUniquePtr<void> fmha_buffer;
    IAllocatorUniquePtr<void> unpacked_qkv_buffer;
    IAllocatorUniquePtr<int> seq_lens_buffer;
    IAllocatorUniquePtr<int> seqlens_k_buffer;
    IAllocatorUniquePtr<void> present_key_scratch;
    IAllocatorUniquePtr<void> present_value_scratch;

    gqa_data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
    gqa_data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
    gqa_data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
    gqa_data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
    gqa_data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
    gqa_data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());

    size_t present_kv_size = static_cast<size_t>(parameters.batch_size) *
                             static_cast<size_t>(parameters.kv_num_heads) *
                             static_cast<size_t>(parameters.total_sequence_length) *
                             static_cast<size_t>(parameters.head_size) * sizeof(CudaT);
    if (present_key != nullptr) {
      gqa_data.present_key = reinterpret_cast<CudaT*>(present_key->MutableData<T>());
    } else {
      present_key_scratch = GetScratchBuffer<void>(present_kv_size, context->GetComputeStream());
      gqa_data.present_key = reinterpret_cast<CudaT*>(present_key_scratch.get());
    }
    if (present_value != nullptr) {
      gqa_data.present_value = reinterpret_cast<CudaT*>(present_value->MutableData<T>());
    } else {
      present_value_scratch = GetScratchBuffer<void>(present_kv_size, context->GetComputeStream());
      gqa_data.present_value = reinterpret_cast<CudaT*>(present_value_scratch.get());
    }

    gqa_parameters.past_present_share_buffer = (gqa_data.past_key == gqa_data.present_key);

    IAllocatorUniquePtr<void> softmax_lse_buffer;
    IAllocatorUniquePtr<void> softmax_lse_accum_buffer;
    IAllocatorUniquePtr<void> out_accum_buffer;

#if USE_FLASH_ATTENTION
    bool use_flash_attention = onnxruntime::flash::is_supported<T>(device_prop,
                                                                   gqa_parameters.head_size,
                                                                   gqa_parameters.num_heads,
                                                                   gqa_parameters.kv_num_heads);
    gqa_data.use_flash_attention = use_flash_attention;
    gqa_data.use_flash_attention_fast_decode = use_flash_attention &&
                                               !gqa_parameters.is_first_prompt &&
                                               gqa_parameters.past_present_share_buffer;

    if (use_flash_attention) {
      size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(
          gqa_parameters.sequence_length, gqa_parameters.batch_size, gqa_parameters.num_heads);

      int num_heads_for_split = gqa_data.use_flash_attention_fast_decode
                                    ? gqa_parameters.kv_num_heads
                                    : gqa_parameters.num_heads;
      auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
          onnxruntime::flash::get_num_splits_and_buffer_sizes(
              gqa_parameters.batch_size, gqa_parameters.sequence_length,
              gqa_parameters.total_sequence_length, num_heads_for_split,
              gqa_parameters.head_size, device_prop.multiProcessorCount);

      gqa_parameters.num_splits = static_cast<int>(num_splits);

      if (gqa_data.use_flash_attention_fast_decode && num_splits > 1) {
        softmax_lse_accum_bytes = onnxruntime::flash::get_softmax_lse_accum_size(
            num_splits, gqa_parameters.batch_size, gqa_parameters.num_heads, gqa_parameters.sequence_length);
        auto round_multiple = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
        out_accum_bytes = onnxruntime::flash::get_out_accum_size(
            num_splits, gqa_parameters.batch_size, gqa_parameters.num_heads, gqa_parameters.sequence_length,
            round_multiple(gqa_parameters.head_size, 32));
      }

      softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
      softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
      out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());

      gqa_data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
      gqa_data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
      gqa_data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
    } else {
      gqa_data.softmax_lse = nullptr;
      gqa_data.softmax_lse_accum = nullptr;
      gqa_data.out_accum = nullptr;
    }
#else
    gqa_data.use_flash_attention = false;
    gqa_data.use_flash_attention_fast_decode = false;
    gqa_data.softmax_lse = nullptr;
    gqa_data.softmax_lse_accum = nullptr;
    gqa_data.out_accum = nullptr;
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
    if (!gqa_data.use_flash_attention) {
      int sm = (device_prop.major * 10) + device_prop.minor;
      bool use_memory_efficient_attention =
          onnxruntime::contrib::cuda::has_memory_efficient_attention(
              sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value,
              gqa_parameters.head_size, gqa_parameters.head_size);
      gqa_data.use_memory_efficient_attention = use_memory_efficient_attention;

      size_t kv_buffer_bytes = (use_memory_efficient_attention &&
                                (gqa_parameters.num_heads != gqa_parameters.kv_num_heads))
                                   ? (sizeof(T) * gqa_parameters.batch_size * gqa_parameters.num_heads *
                                      gqa_parameters.seqlen_present_kv_cache * gqa_parameters.head_size)
                                   : 0;
      size_t fmha_buffer_bytes =
          (use_memory_efficient_attention &&
           onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
               gqa_parameters.head_size, sizeof(T) == sizeof(float)))
              ? (sizeof(float) * gqa_parameters.batch_size * gqa_parameters.sequence_length *
                 gqa_parameters.num_heads * gqa_parameters.head_size)
              : 0;

      k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
      v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
      fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, context->GetComputeStream());

      gqa_data.k = reinterpret_cast<CudaT*>(k_buffer.get());
      gqa_data.v = reinterpret_cast<CudaT*>(v_buffer.get());
      gqa_data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
    } else {
      gqa_data.use_memory_efficient_attention = false;
      gqa_data.k = nullptr;
      gqa_data.v = nullptr;
      gqa_data.fmha_buffer = nullptr;
    }
#else
    gqa_data.use_memory_efficient_attention = false;
    gqa_data.k = nullptr;
    gqa_data.v = nullptr;
    gqa_data.fmha_buffer = nullptr;
#endif

    auto buffer_req = onnxruntime::contrib::cuda::GQABufferRequirements::Compute<T>(
        gqa_parameters, false, gqa_data.use_flash_attention,
        gqa_data.use_flash_attention_fast_decode, gqa_data.use_memory_efficient_attention);

    if (buffer_req.qkv_buffer_bytes > 0) {
      unpacked_qkv_buffer = GetScratchBuffer<void>(buffer_req.qkv_buffer_bytes, context->GetComputeStream());
      gqa_data.qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
    } else {
      gqa_data.qkv_buffer = nullptr;
    }

    seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
    auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

    if (attn_mask != nullptr && attn_mask->IsDataType<bool>()) {
      const auto& mask_shape = attn_mask->Shape();
      int mask_dims = static_cast<int>(mask_shape.NumDimensions());
      int64_t mask_dim0 = mask_shape[0];
      int64_t mask_dim1 = mask_dims >= 3 ? mask_shape[1] : 0;
      int64_t mask_dim2 = mask_dims >= 4 ? mask_shape[2] : 0;

      ORT_RETURN_IF_ERROR(LaunchConvertMaskToSeqlensK(
          attn_mask->Data<bool>(), seqlens_k_buffer.get(),
          parameters.batch_size, parameters.total_sequence_length,
          mask_dims, mask_dim0, mask_dim1, mask_dim2,
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if (attn_mask != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Non-boolean attn_mask is not supported yet in GQA path of Attention op (CUDA).");
    } else {
      std::vector<int> seqlens_k_host(parameters.batch_size, parameters.total_sequence_length - 1);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(seqlens_k_buffer.get(), seqlens_k_host.data(),
                                           sizeof(int) * parameters.batch_size,
                                           cudaMemcpyHostToDevice, cuda_stream));
    }

    seq_lens_buffer = GetScratchBuffer<int>(3 * parameters.batch_size, context->GetComputeStream());
    gqa_data.past_seq_lens = seq_lens_buffer.get();
    gqa_data.total_seq_lens = seq_lens_buffer.get() + parameters.batch_size;
    gqa_data.padded_seq_lens = gqa_data.total_seq_lens + parameters.batch_size;

    ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchGetSequenceLengths(
        seqlens_k_buffer.get(), gqa_data.past_seq_lens,
        gqa_data.total_seq_lens, gqa_data.padded_seq_lens,
        parameters.batch_size, parameters.q_sequence_length,
        gqa_parameters.is_first_prompt, cuda_stream,
        device_prop.maxThreadsPerBlock));

    gqa_data.cos_cache = nullptr;
    gqa_data.sin_cache = nullptr;
    gqa_data.head_sink = nullptr;
    gqa_data.position_ids = nullptr;

    cublasHandle_t cublas = GetCublasHandle(context);

    return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
        device_prop, cublas, context->GetComputeStream(), gqa_parameters, gqa_data);
  }
}
}  // namespace cuda
}  // namespace onnxruntime
