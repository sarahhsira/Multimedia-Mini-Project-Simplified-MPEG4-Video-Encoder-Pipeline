import os
import numpy as np
from preprocess import load_image, resize, to_ycbcr
from dct import dct_2d, quantize
from rle import rle_encode
import pickle

BLOCK = 8

def encode_frame(img):
    ycbcr = to_ycbcr(img)
    Y = ycbcr[:, :, 0]

    h, w = Y.shape
    compressed = []

    for i in range(0, h, BLOCK):
        for j in range(0, w, BLOCK):
            block = Y[i:i+BLOCK, j:j+BLOCK]

            if block.shape == (8, 8):
                dct = dct_2d(block)
                q = quantize(dct, q=50)

                flat = q.flatten().tolist()
                compressed.append(rle_encode(flat))

    return compressed


def encode_folder(folder):
    frames = sorted(os.listdir(folder))
    video = []

    for f in frames:
        img = load_image(os.path.join(folder, f))
        img = resize(img)
        video.append(encode_frame(img))

    return video