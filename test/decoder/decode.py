import cv2
import numpy as np
from encoder.dct import dequantize, idct_2d
from encoder.rle import rle_decode

BLOCK = 8


def decode_frame(frame, q=50):
    """
    Décode une I-frame : liste de blocs RLE-encodés → liste de blocs 8x8 reconstruits.
    Utilisé par main_decode.py après désérialisation.
    """
    blocks = []
    for encoded_block in frame:
        flat = rle_decode(encoded_block)
        if len(flat) != 64:
            continue
        arr = np.array(flat, dtype=np.int16).reshape((8, 8))
        deq = dequantize(arr, q=q)   # ← utilise la vraie matrice JPEG, pas Q*q
        block = idct_2d(deq)
        blocks.append(block.astype(np.float32))
    return blocks


def decode_pframe(motions, residuals, ref_frame, q=50):
    """
    Décode une P-frame en combinant :
      - le bloc prédit depuis ref_frame (compensation de mouvement)
      - le résiduel décodé (RLE → déquantif → IDCT)
    """
    BLOCK_MB = 16
    h, w = ref_frame.shape
    reconstructed = np.zeros((h, w), dtype=np.float32)

    idx = 0
    for x in range(0, h - BLOCK_MB + 1, BLOCK_MB):
        for y in range(0, w - BLOCK_MB + 1, BLOCK_MB):
            if idx >= len(motions):
                break

            dx, dy = motions[idx]

            ref_x = max(0, min(x + dx, h - BLOCK_MB))
            ref_y = max(0, min(y + dy, w - BLOCK_MB))
            ref_block = ref_frame[ref_x:ref_x + BLOCK_MB, ref_y:ref_y + BLOCK_MB].astype(np.float32)

            # Décoder les 4 sous-blocs 8x8 du résiduel
            residual_sum = np.zeros((BLOCK_MB, BLOCK_MB), dtype=np.float32)
            if idx < len(residuals):
                for i2, res_block in enumerate(residuals[idx]):
                    flat = rle_decode(res_block)
                    if len(flat) != 64:
                        continue
                    arr = np.array(flat, dtype=np.int16).reshape((8, 8))
                    deq = dequantize(arr, q=q)
                    # Résiduel : pas de +128, on utilise cv2.idct directement
                    res = cv2.idct(np.float32(deq))
                    row = (i2 // 2) * 8
                    col = (i2 % 2) * 8
                    residual_sum[row:row + 8, col:col + 8] = res

            reconstructed[x:x + BLOCK_MB, y:y + BLOCK_MB] = np.clip(ref_block + residual_sum, 0, 255)
            idx += 1

    return reconstructed.astype(np.float32)


def rebuild_image(blocks, size=(256, 256)):
    """Recolle les blocs 8x8 en une image complète."""
    img = np.zeros(size, dtype=np.float32)
    idx = 0
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            if idx < len(blocks):
                img[i:i + 8, j:j + 8] = blocks[idx]
                idx += 1
    return img
