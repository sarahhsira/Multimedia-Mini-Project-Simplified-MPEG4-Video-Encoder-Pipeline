def lzw_encode(data):
    dict_ = {chr(i): i for i in range(256)}
    current = ""
    code = 256
    result = []

    for c in data:
        temp = current + c
        if temp in dict_:
            current = temp
        else:
            result.append(dict_[current])
            dict_[temp] = code
            code += 1
            current = c

    if current:
        result.append(dict_[current])

    return result


def lzw_decode(data):
    dict_ = {i: chr(i) for i in range(256)}
    current = chr(data[0])
    result = [current]
    code = 256

    for k in data[1:]:
        if k in dict_:
            entry = dict_[k]
        else:
            entry = current + current[0]

        result.append(entry)
        dict_[code] = current + entry[0]
        code += 1
        current = entry

    return ''.join(result)