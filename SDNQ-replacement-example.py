from functools import wraps

# inference only kernel:
from sdnq.kernels.triton_atten import sdnq_triton_atten

# for training:
#from sdnq.kernels.triton_atten_backward import sdnq_triton_atten_with_backward as sdnq_triton_atten

sdpa_pre_sdnq_atten = torch.nn.functional.scaled_dot_product_attention
@wraps(sdpa_pre_sdnq_atten)
def sdpa_sdnq_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.FloatTensor:
    if (
        query.device.type != "cpu"
        and (query.shape[-2] >= 32 and key.shape[-2] >= 32)
        and (query.shape[-2] > 512 or key.shape[-2] > 512) # Skip TE
        and query.shape[-3] > 1 # Skip VAE
    ):
        return sdnq_triton_atten(
            query=query, key=key, value=value, attn_mask=attn_mask,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            matmul_dtype="int8", # can be one of "disabled", "int8", "float8_e4m3fn", "float16".
            pv_matmul_dtype="disabled", # can be one of "disabled", "int8", "float8_e4m3fn", "float16".
            smooth_k=True,
            use_hadamard=False,
            hadamard_group_size=256,
            do_quantize=True, # Set this to False to disable the quantized matmul usage
            quantize_fp32=True, # Set this to False to disable upcasting to FP32 when quantizing
            use_fp16_accum=False, # Set this to True to use FP16 accumulaton with matmul_dtype="float16" and pv_matmul_dtype="float16" or "disabled"
            out_dtype=None, # Set this to a torch.dtype like torch.float32 if you want the output dtype to be different than inputs
        )
    else:
        if enable_gqa:
            kwargs["enable_gqa"] = enable_gqa
        return sdpa_pre_sdnq_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
torch.nn.functional.scaled_dot_product_attention = sdpa_sdnq_atten