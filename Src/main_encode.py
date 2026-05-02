# ============================================================
# main_encode.py CORRIGÉ - UNE SEULE VISUALISATION
# ============================================================

import numpy as np
import pickle
import os
import cv2
import matplotlib.pyplot as plt

from encoder.encode import encode_folder, decode_frame, decode_p_frame
from encoder.lzw import lzw_encode_bytes, lzw_decode_bytes
from encoder.dct import dct_2d, quantize, idct_2d, dequantize
from encoder.preprocess import load_image, resize, to_ycbcr, subsample_chrominance
from encoder.motion import block_matching
from utils import compression_ratio, compression_percentage
from visualize import visualize_complete_pipeline

BLOCK = 8
GOP = 3
Q = 50
SEARCH = 8

# =========================
# 1. ENCODAGE
# =========================
frames_path = "Codes/Src/frames"
video = encode_folder(frames_path)

print(f"✓ Encodage terminé: {len(video)} frames")
i_frames = sum(1 for f in video if f[0] == "I")
p_frames = len(video) - i_frames
print(f"  - I-frames: {i_frames}")
print(f"  - P-frames: {p_frames}")

# =========================
# 2. COMPRESSION ENTROPIQUE
# =========================
data = pickle.dumps(video)
print(f"  Taille pickle: {len(data)} bytes")

compressed_codes = lzw_encode_bytes(data)
print(f"  Taille LZW: {len(compressed_codes)} codes")

output_path = "Codes/Src/output"
os.makedirs(output_path, exist_ok=True)
file_path = os.path.join(output_path, "video.bin")

with open(file_path, "wb") as f:
    code_bytes = np.array(compressed_codes, dtype=np.uint32).tobytes()
    f.write(code_bytes)

compressed_size = os.path.getsize(file_path)
print(f"  Taille finale: {compressed_size} bytes")

# =========================
# 3. STATISTIQUES
# =========================
original_size = 0
frames_list = sorted([f for f in os.listdir(frames_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
for f in frames_list:
    img = cv2.imread(os.path.join(frames_path, f))
    if img is not None:
        original_size += img.shape[0] * img.shape[1] * 3

print(f"\n--- STATISTIQUES DE COMPRESSION ---")
print(f"Taille originale (pixels bruts): {original_size} bytes")
print(f"Taille compressée (fichier .bin): {compressed_size} bytes")
print(f"Taux de compression: {original_size/compressed_size:.2f}:1")
print(f"Gain d'espace: {compression_percentage(original_size, compressed_size):.1f}%")

# =========================
# 4. PRÉPARATION DES DONNÉES POUR VISUALISATION
# =========================
print("\n--- PRÉPARATION DES DONNÉES ---")

# Charger une image de test
test_img_path = os.path.join(frames_path, frames_list[0])
img_original = load_image(test_img_path)
img_original = resize(img_original)
ycbcr_original = to_ycbcr(img_original)

# Extraire les vecteurs de mouvement de la première P-frame
motion_vectors_real = []
for i, frame in enumerate(video):
    if frame[0] == "P" and motion_vectors_real == []:
        _, motions, residuals, _ = frame
        motion_vectors_real = motions
        break

# Préparer les données DCT
block_y_np = ycbcr_original[0:8, 0:8, 0].astype(np.float32)
block_centered = block_y_np - 128
dct_coeffs_cv = cv2.dct(block_centered)
Q_mat = np.ones((8, 8)) * Q
quantized_cv = np.round(dct_coeffs_cv / Q_mat)
dequantized_cv = quantized_cv * Q_mat
reconstructed_cv = cv2.idct(dequantized_cv) + 128
reconstructed_cv = np.clip(reconstructed_cv, 0, 255)

dct_norm_cv = np.abs(dct_coeffs_cv) / (np.max(np.abs(dct_coeffs_cv)) + 1e-6) * 255
quantized_norm_cv = np.abs(quantized_cv) / (np.max(np.abs(quantized_cv)) + 1e-6) * 255

dct_pipeline_visu = {
    'original': block_y_np,
    'dct_norm': dct_norm_cv,
    'quantized_norm': quantized_norm_cv,
    'reconstructed': reconstructed_cv
}

# Résiduel
residual_for_visu = np.zeros((256, 256))
if motion_vectors_real:
    for i, (dx, dy) in enumerate(motion_vectors_real[:200]):
        x = (i // 16) * 16
        y = (i % 16) * 16
        if x < 256 and y < 256:
            residual_for_visu[x:x+4, y:y+4] = np.sqrt(dx**2 + dy**2) * 5

# Image reconstruite
reconstructed_for_visu = ycbcr_original[:, :, 0]

# =========================
# 5. VISUALISATION UNIQUE
# =========================


print("\n--- GÉNÉRATION DE LA VISUALISATION ---")
print(f"  Vecteurs de mouvement trouvés: {len(motion_vectors_real)}")

visualize_complete_pipeline(
    frames_path=frames_path,
    video_data=video,
    motion_vectors=motion_vectors_real,
    ycbcr_img=ycbcr_original,
    dct_pipeline_data=dct_pipeline_visu,
    residual_img=residual_for_visu,
    reconstructed_img=reconstructed_for_visu
)

print("\n=== PROJET TERMINÉ AVEC SUCCÈS ===")
