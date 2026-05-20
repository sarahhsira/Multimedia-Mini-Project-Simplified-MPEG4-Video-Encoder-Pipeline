import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image non trouvée: {path}")
    return img


def resize(img, size=(256, 256)):
    return cv2.resize(img, size)


def to_ycbcr(img):
    #Convertit BGR to YCrCb 
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def from_ycbcr(img):
    #Convertit YCrCb to  BGR
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)


def subsample_chrominance(ycbcr_img, ratio=2):
    #Sous-échantillonnage 4:2:0 
    Y, Cb, Cr = cv2.split(ycbcr_img)
    Cb_sub = cv2.resize(Cb, (Cb.shape[1] // ratio, Cb.shape[0] // ratio),
                        interpolation=cv2.INTER_LINEAR)
    Cr_sub = cv2.resize(Cr, (Cr.shape[1] // ratio, Cr.shape[0] // ratio),
                        interpolation=cv2.INTER_LINEAR)
    return Y, Cb_sub, Cr_sub


def upsample_chrominance(Cb_small, Cr_small, target_shape):
    #Ré-échantillonnage
    Cb_big = cv2.resize(Cb_small, (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    Cr_big = cv2.resize(Cr_small, (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    return Cb_big, Cr_big
