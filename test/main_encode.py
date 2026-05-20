import numpy as np
import pickle
import os
import cv2

from encoder.encode import encode_folder, decode_frame
from encoder.lzw import lzw_encode_bytes
from encoder.dct import dct_2d, quantize, _build_Q
from encoder.preprocess import load_image, resize, to_ycbcr
from utils import compression_ratio, compression_percentage
from visualize import visualize_complete_pipeline

GOP = 3
Q = 50

# 1. ENCODAGE
frames_path = "test/frames"
print("=" * 50)
print("ENCODEUR VIDÉO MPEG-4")
print("=" * 50)
print(f"\nDossier : {frames_path}")

video = encode_folder(frames_path)

i_frames = sum(1 for f in video if f[0] == "I")
p_frames = len(video) - i_frames
print(f"\nEncodage terminé : {len(video)} frames")
print(f"  - I-frames : {i_frames}")
print(f"  - P-frames : {p_frames}")

# 2. SÉRIALISATION + COMPRESSION LZW --> fichier .bin

print("\nCompression LZW + écriture du fichier .bin ...")
data = pickle.dumps(video)
print(f"  Taille pickle (avant LZW) : {len(data)} bytes")

compressed_codes = lzw_encode_bytes(data)
print(f"  Codes LZW : {len(compressed_codes)}")

output_path = "test/output"
os.makedirs(output_path, exist_ok=True)
file_path = os.path.join(output_path, "video.bin")

with open(file_path, "wb") as f:
    f.write(np.array(compressed_codes, dtype=np.uint32).tobytes())

compressed_size = os.path.getsize(file_path)
print(f"  Taille finale du .bin : {compressed_size} bytes")

# 3. STATISTIQUES DE COMPRESSION
frames_list = sorted([
    f for f in os.listdir(frames_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
original_size = 0
for f in frames_list:
    img = cv2.imread(os.path.join(frames_path, f))
    if img is not None:
        original_size += img.shape[0] * img.shape[1] * img.shape[2]

print(f"\n--- STATISTIQUES ---")
print(f"Taille originale (pixels bruts) : {original_size} bytes")
print(f"Taille compressée (.bin)        : {compressed_size} bytes")
print(f"Taux de compression             : {compression_ratio(original_size, compressed_size):.2f}:1")
print(f"Gain d'espace                   : {compression_percentage(original_size, compressed_size):.1f}%")


# 4. PRÉPARATION DES DONNÉES POUR LA VISUALISATION
print("\nPréparation de la visualisation ...")

# Image de référence pour la visualisation
test_img = resize(load_image(os.path.join(frames_path, frames_list[0])))
ycbcr_original = to_ycbcr(test_img)

# Vecteurs de mouvement 
motion_vectors_real = []
for frame in video:
    if frame[0] == "P":
        motion_vectors_real = frame[1]
        break

# Pipeline DCT 
block_y = ycbcr_original[0:8, 0:8, 0].astype(np.float32)
block_centered = block_y - 128
dct_coeffs = cv2.dct(block_centered)
Q_mat = _build_Q(Q)
quantized = np.round(dct_coeffs / Q_mat)
dequantized = quantized * Q_mat
reconstructed_block = np.clip(cv2.idct(dequantized) + 128, 0, 255)

dct_pipeline_visu = {
    'original':       block_y,
    'dct_norm':       np.abs(dct_coeffs) / (np.max(np.abs(dct_coeffs)) + 1e-6) * 255,
    'quantized_norm': np.abs(quantized) / (np.max(np.abs(quantized)) + 1e-6) * 255,
    'reconstructed':  reconstructed_block,
}

# Carte de résiduel
residual_visu = np.zeros((256, 256), dtype=np.float32)
for i, (dx, dy) in enumerate(motion_vectors_real[:256]):
    x = (i // 16) * 16
    y = (i % 16) * 16
    if x < 256 and y < 256:
        residual_visu[x:x + 4, y:y + 4] = np.sqrt(dx ** 2 + dy ** 2) * 5

# Image reconstruite 
reconstructed_visu = ycbcr_original[:, :, 0]

# 5. VISUALISATION

print("Génération de la figure ...")
visualize_complete_pipeline(
    frames_path=frames_path,
    video_data=video,
    motion_vectors=motion_vectors_real,
    ycbcr_img=ycbcr_original,
    dct_pipeline_data=dct_pipeline_visu,
    residual_img=residual_visu,
    reconstructed_img=reconstructed_visu,
)

print("\n=== ENCODAGE TERMINÉ AVEC SUCCÈS ===")
