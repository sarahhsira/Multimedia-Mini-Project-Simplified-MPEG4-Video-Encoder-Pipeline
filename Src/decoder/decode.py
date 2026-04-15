import numpy as np
from encoder.dct import dequantize, idct_2d
from encoder.rle import rle_decode

BLOCK = 8

def decode_frame(frame):
    blocks = []

    for encoded_block in frame:
        flat = rle_decode(encoded_block)
        arr = np.array(flat).reshape((8, 8))

        deq = dequantize(arr, q=50)
        block = idct_2d(deq)

        blocks.append(block)

    return blocks