import cv2
import numpy as np

def load_image(path):
    return cv2.imread(path)

def resize(img, size=(256, 256)):
    return cv2.resize(img, size)

def to_ycbcr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)