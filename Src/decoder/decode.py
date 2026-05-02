import numpy as np
from encoder.dct import dequantize, idct_2d
from encoder.rle import rle_decode

BLOCK = 8


def decode_frame(frame, q=50):
    """Décode une I-frame"""
    blocks = []
    
    for encoded_block in frame:
        # RLE decode
        flat = rle_decode(encoded_block)
        arr = np.array(flat).reshape((8, 8))
        
        # Déquantification
        deq = dequantize(arr, q=q)
        
        # IDCT
        block = idct_2d(deq)
        block = np.clip(block, 0, 255)
        
        blocks.append(block)
    
    return blocks


def decode_pframe(motions, residuals, ref_frame, q=50):
    """Décode une P-frame"""
    BLOCK_MB = 16
    h, w = ref_frame.shape
    reconstructed = np.zeros_like(ref_frame)
    
    idx = 0
    
    for x in range(0, h - BLOCK_MB + 1, BLOCK_MB):
        for y in range(0, w - BLOCK_MB + 1, BLOCK_MB):
            if idx >= len(motions):
                break
                
            dx, dy = motions[idx]
            
            # Vérifier les limites
            ref_x = x + dx
            ref_y = y + dy
            
            if ref_x < 0: ref_x = 0
            if ref_y < 0: ref_y = 0
            if ref_x + BLOCK_MB > h: ref_x = h - BLOCK_MB
            if ref_y + BLOCK_MB > w: ref_y = w - BLOCK_MB
            
            ref_block = ref_frame[ref_x:ref_x+BLOCK_MB, ref_y:ref_y+BLOCK_MB]
            
            # Décoder le résiduel (4 blocs 8x8)
            residual_sum = np.zeros((BLOCK_MB, BLOCK_MB))
            
            if idx < len(residuals):
                residual_blocks = residuals[idx]
                for i2, res_block in enumerate(residual_blocks):
                    # RLE decode
                    flat = rle_decode(res_block)
                    arr = np.array(flat).reshape((8, 8))
                    
                    # Déquantification
                    deq = dequantize(arr, q=q)
                    
                    # IDCT
                    res = idct_2d(deq)
                    
                    # Position dans le macrobloc 16x16
                    row = (i2 // 2) * 8
                    col = (i2 % 2) * 8
                    residual_sum[row:row+8, col:col+8] = res
            
            # Reconstruire
            block = ref_block + residual_sum
            block = np.clip(block, 0, 255)
            reconstructed[x:x+BLOCK_MB, y:y+BLOCK_MB] = block
            
            idx += 1
    
    return reconstructed


def rebuild_image(blocks, size=(256, 256)):
    """Reconstruit une image à partir des blocs 8x8"""
    img = np.zeros(size)
    idx = 0
    
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            if idx < len(blocks):
                img[i:i+8, j:j+8] = blocks[idx]
                idx += 1
    
    return img