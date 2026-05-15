import numpy as np


def compression_ratio(original_size, compressed_size):
    """Ratio original / compressé (ex: 5.3 veut dire 5.3x plus petit)."""
    if compressed_size == 0:
        return 0
    return original_size / compressed_size


def compression_percentage(original_size, compressed_size):
    """Pourcentage de gain (ex: 81% veut dire qu'on a économisé 81% de l'espace)."""
    if original_size == 0:
        return 0
    return (1 - compressed_size / original_size) * 100


def psnr(original, reconstructed):
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio) entre deux images.
    - original / reconstructed : numpy arrays (H, W) ou (H, W, C), dtype uint8 ou float
    - Retourne la valeur en dB (plus c'est élevé, meilleure est la qualité).
      > 40 dB  → excellent
      30–40 dB → bon
      < 30 dB  → qualité visible dégradée
    """
    orig = original.astype(np.float32)
    rec  = reconstructed.astype(np.float32)
    mse  = np.mean((orig - rec) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(original, reconstructed):
    """
    Calcule le SSIM (Structural Similarity Index) simplifié.
    Valeur entre -1 et 1 (1 = images identiques).
    Travaille sur des images en niveaux de gris (2D).
    """
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