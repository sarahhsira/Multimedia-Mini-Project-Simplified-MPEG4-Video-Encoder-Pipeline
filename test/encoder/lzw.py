def lzw_encode_bytes(data):
    
    if not data:
        return []
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    result = []
    current = bytes([])
    for byte in data:
        new_current = current + bytes([byte])
        if new_current in dictionary:
            current = new_current
        else:
            if current:
                result.append(dictionary[current])
            dictionary[new_current] = dict_size
            dict_size += 1
            current = bytes([byte])
    if current:
        result.append(dictionary[current])
    return result


def lzw_decode_bytes(encoded_data):

    if not encoded_data:
        return bytes([])
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    result = []
    current = bytes([encoded_data[0]])
    result.append(current)
    for code in encoded_data[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = current + current[:1]
        else:
            raise ValueError(f"Code LZW invalide: {code}")
        result.append(entry)
        dictionary[dict_size] = current + entry[:1]
        dict_size += 1
        current = entry
    return b''.join(result)
