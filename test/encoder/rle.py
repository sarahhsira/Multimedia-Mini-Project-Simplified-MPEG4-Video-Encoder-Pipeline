def rle_encode(data):
    """Encode une liste par RLE → liste de tuples (valeur, count)."""
    if len(data) == 0:
        return []
    encoded = []
    prev = data[0]
    count = 1
    for x in data[1:]:
        if x == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = x
            count = 1
    encoded.append((prev, count))
    return encoded


def rle_decode(data):
    """Décode une liste de tuples (valeur, count) → liste."""
    out = []
    for value, count in data:
        out.extend([value] * count)
    return out
