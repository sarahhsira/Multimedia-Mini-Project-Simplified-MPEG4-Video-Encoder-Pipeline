# ============================================================
# main_decode.py - DÉCOMPRESSEUR VIDÉO MPEG-4
# ============================================================

import numpy as np
import pickle
import cv2
import os

# ============================================================
# 1. FONCTIONS NÉCESSAIRES POUR LE DÉCODAGE
# ============================================================

# RLE Decode
def rle_decode(data):
    if len(data) == 0:
        return []
    out = []
    for value, count in data:
        out.extend([value] * count)
    return out

# LZW Decode (version bytes)
def lzw_decode_bytes(encoded_data):
    if not encoded_data:
        return bytes([])
    
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    
    result = []
    current = bytes([encoded_data[0]])
    result.append(current)
    
    for code in encoded_data[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = current + current[:1]
        else:
            raise ValueError(f"Code invalide: {code}")
        
        result.append(entry)
        dictionary[dict_size] = current + entry[:1]
        dict_size += 1
        current = entry
    
    return b''.join(result)

# DCT et IDCT
def idct_2d(block):
    return cv2.idct(np.float32(block))

def dequantize(block, q=50):
    Q = np.ones((8, 8)) * q
    return block * Q

# Décodage I-frame
def decode_frame(frame_data, q=50):
    blocks = []
    for encoded_block in frame_data:
        flat = rle_decode(encoded_block)
        if len(flat) == 64:
            arr = np.array(flat).reshape((8, 8))
            deq = dequantize(arr, q)
            block = idct_2d(deq)
            block = np.clip(block, 0, 255)
            blocks.append(block)
    return blocks

def rebuild_image(blocks, size=(256, 256)):
    img = np.zeros(size)
    idx = 0
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            if idx < len(blocks):
                img[i:i+8, j:j+8] = blocks[idx]
                idx += 1
    return img

# Décodage P-frame
def decode_pframe(motions, residuals, ref_frame, q=50):
    BLOCK_MB = 16
    h, w = ref_frame.shape
    reconstructed = np.zeros_like(ref_frame)
    
    idx = 0
    for x in range(0, h - BLOCK_MB + 1, BLOCK_MB):
        for y in range(0, w - BLOCK_MB + 1, BLOCK_MB):
            if idx >= len(motions):
                break
                
            dx, dy = motions[idx]
            
            ref_x = x + dx
            ref_y = y + dy
            
            if ref_x < 0: ref_x = 0
            if ref_y < 0: ref_y = 0
            if ref_x + BLOCK_MB > h: ref_x = h - BLOCK_MB
            if ref_y + BLOCK_MB > w: ref_y = w - BLOCK_MB
            
            ref_block = ref_frame[ref_x:ref_x+BLOCK_MB, ref_y:ref_y+BLOCK_MB]
            
            # Décoder le résiduel
            residual_sum = np.zeros((BLOCK_MB, BLOCK_MB))
            
            if idx < len(residuals):
                res_blocks = residuals[idx]
                for i2, res_block in enumerate(res_blocks):
                    flat = rle_decode(res_block)
                    if len(flat) == 64:
                        arr = np.array(flat).reshape((8, 8))
                        deq = dequantize(arr, q)
                        res = idct_2d(deq)
                        row = (i2 // 2) * 8
                        col = (i2 % 2) * 8
                        residual_sum[row:row+8, col:col+8] = res
            
            block = ref_block + residual_sum
            block = np.clip(block, 0, 255)
            reconstructed[x:x+BLOCK_MB, y:y+BLOCK_MB] = block
            idx += 1
    
    return reconstructed

# ============================================================
# 2. MAIN - DÉCOMPRESSION
# ============================================================

def main():
    print("="*50)
    print("DÉCOMPRESSEUR VIDÉO MPEG-4")
    print("="*50)
    
    # Chemin du fichier compressé
    bin_path = "Codes/Src/output/video.bin"
    
    if not os.path.exists(bin_path):
        print(f"❌ Fichier non trouvé: {bin_path}")
        print("   Veuillez d'abord exécuter l'encodeur pour créer video.bin")
        return
    
    # 1. Charger le fichier .bin
    print(f"\n📂 Chargement: {bin_path}")
    with open(bin_path, "rb") as f:
        compressed_data = f.read()
    print(f"   Taille du fichier: {len(compressed_data)} bytes")
    
    # 2. LZW decode
    print("\n🔓 Décompression LZW...")
    codes = np.frombuffer(compressed_data, dtype=np.uint32).tolist()
    data_bytes = lzw_decode_bytes(codes)
    print(f"   Données décompressées: {len(data_bytes)} bytes")
    
    # 3. Désérialisation
    print("\n📦 Désérialisation...")
    video = pickle.loads(data_bytes)
    print(f"   Frames trouvées: {len(video)}")
    
    # 4. Décodage des frames
    print("\n🖼️ Décodage des images...")
    reconstructed_frames = []
    prev_frame = None
    q = 50
    
    i_frames = 0
    p_frames = 0
    
    for i, frame_data in enumerate(video):
        frame_type = frame_data[0]
        
        if frame_type == "I":
            encoded_data = frame_data[1]
            blocks = decode_frame(encoded_data, q)
            img = rebuild_image(blocks, size=(256, 256))
            reconstructed_frames.append(img)
            prev_frame = img
            i_frames += 1
            print(f"   Frame {i}: I-frame ✅")
            
        elif frame_type == "P":
            motions = frame_data[1]
            residuals = frame_data[2]
            if prev_frame is not None:
                img = decode_pframe(motions, residuals, prev_frame, q)
                reconstructed_frames.append(img)
                prev_frame = img
                p_frames += 1
                print(f"   Frame {i}: P-frame ✅ ({len(motions)} vecteurs)")
    
    # 5. Sauvegarde des images décodées
    print("\n💾 Sauvegarde des images décodées...")
    output_dir = "Codes/Src/output/decoded_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(reconstructed_frames):
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"decoded_{i:03d}.png"), img_uint8)
    
    print(f"   ✅ {len(reconstructed_frames)} images sauvegardées")
    print(f"   📁 Dossier: {output_dir}")
    
    # 6. Statistiques finales
    print("\n" + "="*50)
    print("STATISTIQUES FINALES")
    print("="*50)
    print(f"📊 Frames décodées: {len(reconstructed_frames)}")
    print(f"   - I-frames: {i_frames}")
    print(f"   - P-frames: {p_frames}")
    print(f"📁 Images sauvegardées: {output_dir}/decoded_000.png à decoded_{len(reconstructed_frames)-1:03d}.png")
    print("\n✅ DÉCOMPRESSION TERMINÉE AVEC SUCCÈS !")
    print("="*50)

if __name__ == "__main__":
    main()