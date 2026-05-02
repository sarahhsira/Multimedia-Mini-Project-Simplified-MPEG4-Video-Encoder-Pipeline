import numpy as np

# =========================
# Compression ratio
# =========================
def compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return 0
    return original_size / compressed_size


# =========================
# Compression percentage (bonus utile rapport)
# =========================
def compression_percentage(original_size, compressed_size):
    if original_size == 0:
        return 0
    return (1 - compressed_size / original_size) * 100