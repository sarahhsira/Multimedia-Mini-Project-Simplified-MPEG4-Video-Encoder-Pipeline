import numpy as np


def compression_ratio(original_size, compressed_size):
    
    if compressed_size == 0:
        return 0
    return original_size / compressed_size


def compression_percentage(original_size, compressed_size):
    # Pourcentage de gain 
    if original_size == 0:
        return 0
    return (1 - compressed_size / original_size) * 100

# Calcule le PSNR
def psnr(original, reconstructed):

    orig = original.astype(np.float32)
    rec  = reconstructed.astype(np.float32)
    mse  = np.mean((orig - rec) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)

# Calcule le SSIM
def ssim(original, reconstructed):

    orig = original.astype(np.float64)
    rec  = reconstructed.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = np.mean(orig)
    mu2 = np.mean(rec)

    sigma1_sq = np.var(orig)
    sigma2_sq = np.var(rec)
    sigma12   = np.mean((orig - mu1) * (rec - mu2))

    numerator   = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return numerator / denominator