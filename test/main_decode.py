import numpy as np
import pickle
import cv2
import os

from encoder.lzw import lzw_decode_bytes
from decoder.decode import decode_frame, decode_pframe, rebuild_image

# ============================================================
# MAIN — DÉCOMPRESSEUR VIDÉO MPEG-4
# ============================================================

def main():
    print("=" * 50)
    print("DÉCOMPRESSEUR VIDÉO MPEG-4")
    print("=" * 50)

    bin_path = os.path.join("test/output", "video.bin")

    if not os.path.exists(bin_path):
        print(f"Fichier non trouvé : {bin_path}")
        print("Veuillez d'abord exécuter main_encode.py")
        return

    # 1. Lire le fichier .bin
    print(f"\nChargement : {bin_path}")
    with open(bin_path, "rb") as f:
        raw = f.read()
    print(f"  Taille : {len(raw)} bytes")

    # 2. LZW decode
    print("\nDécompression LZW ...")
    codes = np.frombuffer(raw, dtype=np.uint32).tolist()
    data_bytes = lzw_decode_bytes(codes)
    print(f"  Données décompressées : {len(data_bytes)} bytes")

    # 3. Désérialisation pickle
    print("\nDésérialisation ...")
    video = pickle.loads(data_bytes)
    print(f"  Frames : {len(video)}")

    # 4. Décodage frame par frame
    print("\nDécodage des images ...")
    reconstructed_frames = []
    prev_Y = None          # canal Y de la frame précédente (pour les P-frames)
    q = 50
    i_count = p_count = 0

    for i, frame_data in enumerate(video):
        frame_type = frame_data[0]

        # ── I-FRAME ──────────────────────────────────────────
        if frame_type == "I":
            encoded_data = frame_data[1]
            q = frame_data[2].get("q", 50)

            # Décoder les 3 canaux Y, Cb, Cr
            Y_blocks  = decode_frame(encoded_data['Y'],  q)
            Cb_blocks = decode_frame(encoded_data['Cb'], q)
            Cr_blocks = decode_frame(encoded_data['Cr'], q)

            Y  = rebuild_image(Y_blocks,  size=(256, 256))
            Cb = rebuild_image(Cb_blocks, size=(256, 256))
            Cr = rebuild_image(Cr_blocks, size=(256, 256))

            # Recombiner en image YCbCr puis convertir en BGR
            ycbcr = np.stack([
                np.clip(Y,  0, 255).astype(np.uint8),
                np.clip(Cb, 0, 255).astype(np.uint8),
                np.clip(Cr, 0, 255).astype(np.uint8),
            ], axis=2)
            bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

            reconstructed_frames.append(bgr)
            prev_Y = Y.astype(np.float32)
            i_count += 1
            print(f"  Frame {i}: I-frame OK")

        # ── P-FRAME ──────────────────────────────────────────
        elif frame_type == "P":
            motions   = frame_data[1]
            residuals = frame_data[2]
            q = frame_data[3].get("q", 50)

            if prev_Y is None:
                print(f"  Frame {i}: P-frame ignorée (pas de référence)")
                continue

            Y_rec = decode_pframe(motions, residuals, prev_Y, q)

            # Pour Cb/Cr on prend ceux de la dernière frame reconstruite
            last_ycbcr = cv2.cvtColor(reconstructed_frames[-1], cv2.COLOR_BGR2YCrCb)
            Cb = last_ycbcr[:, :, 1].astype(np.float32)
            Cr = last_ycbcr[:, :, 2].astype(np.float32)

            ycbcr = np.stack([
                np.clip(Y_rec, 0, 255).astype(np.uint8),
                np.clip(Cb,    0, 255).astype(np.uint8),
                np.clip(Cr,    0, 255).astype(np.uint8),
            ], axis=2)
            bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

            reconstructed_frames.append(bgr)
            prev_Y = Y_rec.astype(np.float32)
            p_count += 1
            print(f"  Frame {i}: P-frame OK ({len(motions)} vecteurs)")

    # 5. Sauvegarder les images décodées
    print("\nSauvegarde ...")
    out_dir = os.path.join("test/output", "decoded_frames")
    os.makedirs(out_dir, exist_ok=True)

    for i, img in enumerate(reconstructed_frames):
        path = os.path.join(out_dir, f"decoded_{i:03d}.png")
        cv2.imwrite(path, np.clip(img, 0, 255).astype(np.uint8))

    print(f"  {len(reconstructed_frames)} images sauvegardées dans {out_dir}/")

    # 6. Statistiques
    print("\n" + "=" * 50)
    print("STATISTIQUES")
    print("=" * 50)
    print(f"Frames décodées : {len(reconstructed_frames)}")
    print(f"  - I-frames    : {i_count}")
    print(f"  - P-frames    : {p_count}")
    print("\nDÉCOMPRESSION TERMINÉE AVEC SUCCÈS")
    print("=" * 50)


if __name__ == "__main__":
    main()
