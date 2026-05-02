# dct.py corrigé
import cv2
import numpy as np

def dct_2d(block):
    """DCT 2D sur un bloc 8x8"""
    block_float = np.float32(block)
    # Normaliser entre -128 et 127 avant DCT (standard JPEG)
    block_centered = block_float - 128
    return cv2.dct(block_centered)

def idct_2d(dct_coeffs):
    """IDCT 2D"""
    reconstructed = cv2.idct(dct_coeffs)
    # Re-ajouter 128 et arrondir
    return np.clip(reconstructed + 128, 0, 255).astype(np.uint8)

def quantize(block, q=50):
    """Quantification standard JPEG"""
    # Matrice de quantification JPEG standard (qualité 50)
    Q_standard = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Ajuster selon le facteur q
    scale = q / 50.0
    if scale < 1:
        scale = 1 / (2 - scale)
    
    Q = Q_standard * scale
    Q = np.maximum(Q, 1)
    
    return np.round(block / Q).astype(np.int16)

def dequantize(quantized_block, q=50):
    """Déquantification"""
    Q_standard = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    scale = q / 50.0
    if scale < 1:
        scale = 1 / (2 - scale)
    
    Q = Q_standard * scale
    Q = np.maximum(Q, 1)
    
    return quantized_block.astype(np.float32) * Q