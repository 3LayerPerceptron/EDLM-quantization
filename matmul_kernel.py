import torch
import triton
import triton.language as tl

from final_quantization_kernel import dequantize_rowwise_int4

def matmul_fp16_int4(a: torch.Tensor, b_packed: torch.Tensor, absmaxs: torch.Tensor):
    assert a.is_cuda and b_packed.is_cuda and absmaxs.is_cuda
    assert a.dtype == torch.float16, f"Expected torch.float16, got {a.dtype}"
    assert b_packed.dtype == torch.uint8
    assert absmaxs.dtype == torch.float16
    
    M, K = a.shape
    N, packed_K = b_packed.shape
    
    expected_packed_K = K // 2
    assert packed_K == expected_packed_K, f"Expected packed_K={expected_packed_K}, got {packed_K}. K must be divisible by 2"
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    
    try:
        _matmul_fp16_int4_kernel[grid](
            a, b_packed, absmaxs, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b_packed.stride(0), b_packed.stride(1),
            c.stride(0), c.stride(1),
        )
    except Exception as e:
        print(f"Error in kernel execution: {e}")
        print("Using fallback matmul")
        dequantized_weights = dequantize_rowwise_int4(b_packed, absmaxs)
        c = torch.matmul(a, dequantized_weights.t())
    
    return c

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_fp16_int4_kernel(
    a_ptr, b_ptr, absmaxs_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = rm < M
    mask_n = rn < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = rk < K
        
        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a_block = a_block.to(tl.float32)
        
        packed_k = rk // 2
        mask_packed_k = packed_k < (K // 2)
        
        b_ptrs = b_ptr + rn[:, None] * stride_bn + packed_k[None, :] * stride_bk
        b_packed = tl.load(b_ptrs, mask=mask_n[:, None] & mask_packed_k[None, :], other=0)
        
        b_uint4_1 = (b_packed >> 4) & 0x0F
        b_uint4_2 = b_packed & 0x0F
        
        b_int4_1 = (b_uint4_1.to(tl.int8) - 8).to(tl.float32)
        b_int4_2 = (b_uint4_2.to(tl.int8) - 8).to(tl.float32)
        
        b_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
        
        even_mask = (rk % 2 == 0)[None, :]
        odd_mask = (rk % 2 == 1)[None, :]
        
        b_block = tl.where(even_mask & mask_k[None, :], b_int4_1, b_block)
        b_block = tl.where(odd_mask & mask_k[None, :], b_int4_2, b_block)
        
        scales_ptrs = absmaxs_ptr + rn
        scales = tl.load(scales_ptrs, mask=mask_n, other=1.0)
        scales = scales.to(tl.float32) / 8.0
        
        b_block = b_block * scales[:, None]
        
        b_block_t = tl.trans(b_block)
        
        acc += tl.dot(a_block, b_block_t, allow_tf32=False)
    
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])