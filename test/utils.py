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
