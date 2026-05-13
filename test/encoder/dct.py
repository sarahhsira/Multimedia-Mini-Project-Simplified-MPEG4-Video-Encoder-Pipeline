import cv2
import numpy as np

# Matrice de quantification JPEG standard — définie une seule fois
# pour être sûr que quantize() et dequantize() utilisent exactement la même
Q_STANDARD = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


def _build_Q(q):
    """Construit la matrice de quantification à partir du facteur q."""
    scale = q / 50.0
    if scale < 1:
        scale = 1.0 / (2.0 - scale)
    return np.maximum(Q_STANDARD * scale, 1.0)


def dct_2d(block):
    """DCT 2D sur un bloc 8x8 (centre autour de 0 avant la DCT)."""
    return cv2.dct(np.float32(block) - 128)


def idct_2d(dct_coeffs):
    """IDCT 2D — recentre et clippe entre 0 et 255."""
    return np.clip(cv2.idct(np.float32(dct_coeffs)) + 128, 0, 255).astype(np.uint8)


def quantize(block, q=50):
    """Quantification : divise par Q et arrondit."""
    return np.round(block / _build_Q(q)).astype(np.int16)


def dequantize(quantized_block, q=50):
    """Déquantification : multiplie par la même matrice Q."""
    return quantized_block.astype(np.float32) * _build_Q(q)
