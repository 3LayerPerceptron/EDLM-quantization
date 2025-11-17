import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def _quantize_rowwise_int4(
    x_ptr, output_ptr, output_maxs, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):

    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    abs_row = tl.abs(row)
    row_max = tl.max(tl.where(mask, abs_row, 0.0))
    
    row_max_safe = tl.maximum(row_max, 1e-12)
    scale = 8.0 / row_max_safe
    
    
    tl.store(output_maxs + row_idx, row_max_safe.to(tl.float16))
    
    packed_row_stride = n_cols // 2
    packed_row_start = row_idx * packed_row_stride
    
    for packed_start in range(0, packed_row_stride, BLOCK_SIZE):
        packed_offs = packed_start + tl.arange(0, BLOCK_SIZE)
        mask_packed = packed_offs < packed_row_stride
        
        orig_offs = packed_offs * 2
        mask1 = orig_offs < n_cols
        mask2 = (orig_offs + 1) < n_cols
        
        val1 = tl.load(x_ptr + row_start + orig_offs, mask=mask1 & mask_packed, other=0.0)
        val2 = tl.load(x_ptr + row_start + orig_offs + 1, mask=mask2 & mask_packed, other=0.0)
        
        quant1 = tl.extra.cuda.libdevice.rint(val1 * scale)
        quant2 = tl.extra.cuda.libdevice.rint(val2 * scale)
        
        quant1 = tl.minimum(tl.maximum(quant1, -8.0), 7.0)
        quant2 = tl.minimum(tl.maximum(quant2, -8.0), 7.0)
        
        uint4_1 = (quant1 + 8).to(tl.uint8)
        uint4_2 = (quant2 + 8).to(tl.uint8)
        packed = (uint4_1 << 4) | uint4_2
        
        tl.store(output_ptr + packed_row_start + packed_offs, packed, mask=mask_packed)

def quantize_rowwise_int4(x: torch.Tensor):
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    assert n_cols % 2 == 0
    
    packed_n_cols = n_cols // 2
    q_packed = torch.empty((n_rows, packed_n_cols), device=x.device, dtype=torch.uint8)
    absmaxs = torch.empty(n_rows, device=x.device, dtype=torch.float16)
    
    grid = (n_rows,)
    _quantize_rowwise_int4[grid](x, q_packed, absmaxs, n_rows, n_cols)
    
    return q_packed, absmaxs

def dequantize_rowwise_int4(packed: torch.Tensor, absmaxs: torch.Tensor):
    assert packed.is_cuda and absmaxs.is_cuda
    assert packed.dtype == torch.uint8
    assert absmaxs.dtype == torch.float16
    assert packed.dim() == 2 and absmaxs.dim() == 1
    assert packed.size(0) == absmaxs.size(0)
    
    n_rows, packed_n_cols = packed.shape
    n_cols = packed_n_cols * 2
    
    uint4_1 = (packed >> 4) & 0x0F
    uint4_2 = packed & 0x0F
    
    int4_1 = uint4_1.to(torch.int8) - 8
    int4_2 = uint4_2.to(torch.int8) - 8
    
    dequantized = torch.empty((n_rows, n_cols), device=packed.device, dtype=torch.float16)
    dequantized[:, 0::2] = int4_1.to(torch.float16)
    dequantized[:, 1::2] = int4_2.to(torch.float16)
    
    scale = (absmaxs / 8.0).unsqueeze(1)
    dequantized = dequantized * scale
    
    return dequantized