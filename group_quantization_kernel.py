import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_groupwise_int4(
    x_ptr, output_ptr, output_scales, n_rows, n_cols, group_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_groups_per_row = tl.cdiv(n_cols, group_size)
    row_idx = pid // num_groups_per_row
    group_idx = pid % num_groups_per_row
    
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    group_start = group_idx * group_size
    group_end = tl.minimum(group_start + group_size, n_cols)
    actual_group_size = group_end - group_start
    
    group_max = 0.0
    for start in range(0, actual_group_size, BLOCK_SIZE):
        col_offsets = group_start + start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < group_end
        
        global_offs = row_start + col_offsets
        block = tl.load(x_ptr + global_offs, mask=mask, other=0.0)
        
        abs_block = tl.abs(block)
        block_max = tl.max(tl.where(mask, abs_block, 0.0))
        group_max = tl.maximum(group_max, block_max)
    
    group_max_safe = tl.maximum(group_max, 1e-12)
    scale = 8.0 / group_max_safe
    
    scale_idx = row_idx * num_groups_per_row + group_idx
    tl.store(output_scales + scale_idx, group_max_safe.to(tl.float16))

    packed_group_size = (actual_group_size + 1) // 2
    packed_row_start = row_idx * (n_cols // 2)
    packed_group_start = packed_row_start + (group_start // 2)
    
    for packed_start in range(0, packed_group_size, BLOCK_SIZE):
        packed_offs = packed_start + tl.arange(0, BLOCK_SIZE)
        global_packed_offs = packed_group_start + packed_offs
        mask_packed = (packed_offs < packed_group_size) & (global_packed_offs < (row_idx + 1) * (n_cols // 2))

        orig_offs = group_start + packed_offs * 2
        mask1 = orig_offs < group_end
        mask2 = (orig_offs + 1) < group_end
        
        val1 = tl.load(x_ptr + row_start + orig_offs, mask=mask1 & mask_packed, other=0.0)
        val2 = tl.load(x_ptr + row_start + orig_offs + 1, mask=mask2 & mask_packed, other=0.0)
        
        quant1 = tl.extra.cuda.libdevice.rint(val1 * scale)
        quant2 = tl.extra.cuda.libdevice.rint(val2 * scale)
        
        quant1 = tl.minimum(tl.maximum(quant1, -8.0), 7.0)
        quant2 = tl.minimum(tl.maximum(quant2, -8.0), 7.0)
        
        uint4_1 = (quant1 + 8).to(tl.uint8)
        uint4_2 = (quant2 + 8).to(tl.uint8)
        packed = (uint4_1 << 4) | uint4_2
        
        tl.store(output_ptr + global_packed_offs, packed, mask=mask_packed)

def quantize_groupwise_int4(x: torch.Tensor, group_size: int = 128):
    n_rows, n_cols = x.shape
    assert n_cols % 2 == 0, "n_cols must be even for int4 packing"
    assert group_size % 2 == 0, "group_size must be even for int4 packing"
    
    packed_n_cols = n_cols // 2
    q_packed = torch.empty((n_rows, packed_n_cols), device=x.device, dtype=torch.uint8)
    
    num_groups_per_row = (n_cols + group_size - 1) // group_size
    scales = torch.empty(n_rows * num_groups_per_row, device=x.device, dtype=torch.float16)
    
    total_blocks = n_rows * num_groups_per_row
    
    BLOCK_SIZE = 128
    if group_size < 128:
        BLOCK_SIZE = 64
    
    grid = (total_blocks,)
    
    _quantize_groupwise_int4[grid](
        x, q_packed, scales, n_rows, n_cols, group_size, BLOCK_SIZE
    )
    
    scales = scales.view(n_rows, num_groups_per_row)
    return q_packed, scales

def dequantize_groupwise_int4(packed: torch.Tensor, scales: torch.Tensor, group_size: int = 128):
    n_rows, packed_n_cols = packed.shape
    n_cols = packed_n_cols * 2
    num_groups_per_row = scales.shape[1]
    
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    
    uint4_1 = (packed >> 4).to(torch.int32)
    uint4_2 = (packed & 0x0F).to(torch.int32)
    
    int4_1 = uint4_1 - 8
    int4_2 = uint4_2 - 8

    dequantized = torch.empty((n_rows, n_cols), device=packed.device, dtype=torch.float16)
    dequantized[:, 0::2] = int4_1.to(torch.float16)
    dequantized[:, 1::2] = int4_2.to(torch.float16)
    
    scale_matrix = torch.zeros((n_rows, n_cols), device=scales.device, dtype=torch.float16)
    
    for group_idx in range(num_groups_per_row):
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, n_cols)
        if group_start < n_cols:
            scale_matrix[:, group_start:group_end] = scales[:, group_idx:group_idx+1]

    dequantized = dequantized * (scale_matrix / 8.0)
    
    return dequantized