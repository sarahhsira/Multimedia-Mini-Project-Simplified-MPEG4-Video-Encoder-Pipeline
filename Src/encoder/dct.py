import cv2
import numpy as np

BLOCK = 8

def dct_2d(block):
    return cv2.dct(np.float32(block))

def idct_2d(block):
    return cv2.idct(block)

def quantize(block, q=50):
    return np.round(block / q)

def dequantize(block, q=50):
    return block * q