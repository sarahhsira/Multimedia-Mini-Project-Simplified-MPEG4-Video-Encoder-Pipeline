# lzw.py corrigé - Version qui travaille avec des bytes
def lzw_encode_bytes(data):
    """LZW encoding sur des données bytes"""
    if not data:
        return []
    
    # Dictionnaire initial avec les 256 valeurs possibles
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    
    result = []
    current = bytes([])
    
    for byte in data:
        byte_bytes = bytes([byte])
        new_current = current + byte_bytes
        
        if new_current in dictionary:
            current = new_current
        else:
            if current:
                result.append(dictionary[current])
            dictionary[new_current] = dict_size
            dict_size += 1
            current = byte_bytes
    
    if current:
        result.append(dictionary[current])
    
    return result

def lzw_decode_bytes(encoded_data):
    """LZW decoding vers des bytes"""
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
            raise ValueError(f"Code invalide: {code}")
        
        result.append(entry)
        dictionary[dict_size] = current + entry[:1]
        dict_size += 1
        current = entry
    
    return b''.join(result)