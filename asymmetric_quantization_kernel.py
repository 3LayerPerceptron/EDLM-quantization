import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rowwise_int4_asymmetric(
    x_ptr, output_ptr, output_scales, output_zeros, n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    
    row_min = 1e10
    row_max = -1e10
    
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        global_offs = row_start + col_offsets
        block = tl.load(x_ptr + global_offs, mask=mask, other=0.0)
        
        block_min = tl.min(tl.where(mask, block, 1e10))
        block_max = tl.max(tl.where(mask, block, -1e10))
        
        row_min = tl.minimum(row_min, block_min)
        row_max = tl.maximum(row_max, block_max)
    
    range_val = row_max - row_min
    range_safe = tl.maximum(range_val, 1e-12)
    scale = range_safe / 15.0
    zero_point = (-row_min / scale).to(tl.uint8)
    
    zero_point = tl.minimum(tl.maximum(zero_point, 0), 15)
    
    tl.store(output_scales + row_idx, scale.to(tl.float16))
    tl.store(output_zeros + row_idx, zero_point)
    
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
        
        quant1 = tl.extra.cuda.libdevice.rint((val1 - row_min) / scale)
        quant2 = tl.extra.cuda.libdevice.rint((val2 - row_min) / scale)
        
        quant1 = tl.minimum(tl.maximum(quant1, 0.0), 15.0)
        quant2 = tl.minimum(tl.maximum(quant2, 0.0), 15.0)
        
        uint4_1 = quant1.to(tl.uint8)
        uint4_2 = quant2.to(tl.uint8)
        packed = (uint4_1 << 4) | uint4_2
        
        tl.store(output_ptr + packed_row_start + packed_offs, packed, mask=mask_packed)

def quantize_rowwise_int4_asymmetric(x: torch.Tensor):
    n_rows, n_cols = x.shape
    assert n_cols % 2 == 0
    
    packed_n_cols = n_cols // 2
    q_packed = torch.empty((n_rows, packed_n_cols), device=x.device, dtype=torch.uint8)
    scales = torch.empty(n_rows, device=x.device, dtype=torch.float16)
    zeros = torch.empty(n_rows, device=x.device, dtype=torch.uint8)
    
    grid = (n_rows,)
    _quantize_rowwise_int4_asymmetric[grid](x, q_packed, scales, zeros, n_rows, n_cols, 128)
    
    return q_packed, scales, zeros

def dequantize_rowwise_int4_asymmetric(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    n_rows, packed_n_cols = packed.shape
    n_cols = packed_n_cols * 2
    
    uint4_1 = (packed >> 4).to(torch.uint8)
    uint4_2 = (packed & 0x0F).to(torch.uint8)
    
    dequantized = torch.empty((n_rows, n_cols), device=packed.device, dtype=torch.float16)
    
    scales_expanded = scales.unsqueeze(1)
    zeros_expanded = zeros.unsqueeze(1).to(torch.float16)
    
    dequantized[:, 0::2] = (uint4_1.to(torch.float16) - zeros_expanded) * scales_expanded
    dequantized[:, 1::2] = (uint4_2.to(torch.float16) - zeros_expanded) * scales_expanded
    
    return dequantized