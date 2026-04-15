from encoder.encode import encode_folder
import pickle

video = encode_folder("frames")

with open("output/video.bin", "wb") as f:
    pickle.dump(video, f)

print("Encoding finished ✔")