import pickle
import numpy as np
from decoder.decode import decode_frame

with open("output/video.bin", "rb") as f:
    video = pickle.load(f)

reconstructed = []

for frame in video:
    reconstructed.append(decode_frame(frame))

print("Decoding finished ✔")