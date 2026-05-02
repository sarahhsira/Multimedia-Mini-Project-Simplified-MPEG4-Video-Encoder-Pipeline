# encode.py corrigé
import os 
import numpy as np
from encoder.preprocess import load_image, resize, to_ycbcr
from encoder.dct import dct_2d, quantize, dequantize, idct_2d
from encoder.rle import rle_encode
from encoder.motion import block_matching

BLOCK = 8
GOP = 3
Q = 50  # Facteur de quantification fixe

def encode_frame(img, q=Q):
    """Encode une I-frame complète (Y, Cb, Cr)"""
    ycbcr = to_ycbcr(img)
    compressed = {}
    
    for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
        channel = ycbcr[:, :, channel_idx]
        h, w = channel.shape
        channel_data = []
        
        for i in range(0, h, BLOCK):
            for j in range(0, w, BLOCK):
                block = channel[i:i+BLOCK, j:j+BLOCK]
                
                if block.shape == (BLOCK, BLOCK):
                    dct = dct_2d(block.astype(np.float32))
                    q_block = quantize(dct, q)
                    flat = q_block.flatten().tolist()
                    channel_data.append(rle_encode(flat))
                else:
                    # Gérer les bords - zéro padding
                    padded = np.zeros((BLOCK, BLOCK))
                    padded[:block.shape[0], :block.shape[1]] = block
                    dct = dct_2d(padded.astype(np.float32))
                    q_block = quantize(dct, q)
                    flat = q_block.flatten().tolist()
                    channel_data.append(rle_encode(flat))
        
        compressed[channel_name] = channel_data
    
    return compressed

def encode_folder(folder):
    frames = sorted([
        f for f in os.listdir(folder)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    
    video = []
    prev_decoded = None  # Image décodée pour référence (pas l'originale!)
    
    for i, f in enumerate(frames):
        print(f"Encoding frame {i+1}/{len(frames)}: {f}")
        img = load_image(os.path.join(folder, f))
        img = resize(img)
        ycbcr = to_ycbcr(img)
        
        # I FRAME
        if i % GOP == 0:
            encoded = encode_frame(img)
            video.append(("I", encoded, {"q": Q}))
            
            # Décoder la I-frame pour référence future
            prev_decoded = decode_frame(encoded, ycbcr.shape[:2], Q)
            
        else:
            # P FRAME - doit utiliser l'image DÉCODÉE comme référence
            if prev_decoded is None:
                # Fallback: encoder comme I-frame
                encoded = encode_frame(img)
                video.append(("I", encoded, {"q": Q}))
                prev_decoded = decode_frame(encoded, ycbcr.shape[:2], Q)
                continue
                
            # Extraire Y de l'image courante et de l'image décodée précédente
            Y_current = ycbcr[:, :, 0]
            Y_ref = prev_decoded[:, :, 0]  # Canal Y de l'image décodée!
            
            h, w = Y_current.shape
            motions = []
            residuals = []
            
            # Macroblocs 16x16
            for x in range(0, h, 16):
                for y in range(0, w, 16):
                    # Vérifier les limites
                    if x+16 > h or y+16 > w:
                        continue
                    
                    # Estimation du mouvement
                    dx, dy = block_matching(Y_current, Y_ref, x, y)
                    motions.append((dx, dy))
                    
                    # Récupérer le bloc prédit
                    pred_x = x + dx
                    pred_y = y + dy
                    
                    # S'assurer que le bloc prédit est dans les limites
                    if pred_x < 0: pred_x = 0
                    if pred_y < 0: pred_y = 0
                    if pred_x+16 > h: pred_x = h-16
                    if pred_y+16 > w: pred_y = w-16
                    
                    pred_block = Y_ref[pred_x:pred_x+16, pred_y:pred_y+16]
                    curr_block = Y_current[x:x+16, y:y+16]
                    
                    # Calcul du résiduel
                    residual = curr_block.astype(np.float32) - pred_block.astype(np.float32)
                    
                    # Diviser le résiduel en blocs 8x8 pour DCT
                    residual_blocks = []
                    for i2 in range(0, 16, 8):
                        for j2 in range(0, 16, 8):
                            sub = residual[i2:i2+8, j2:j2+8]
                            dct_sub = dct_2d(sub)
                            q_sub = quantize(dct_sub, Q)
                            residual_blocks.append(rle_encode(q_sub.flatten().tolist()))
                    
                    residuals.append(residual_blocks)
            
            video.append(("P", motions, residuals, {"q": Q}))
            
            # Décoder la P-frame pour référence future
            prev_decoded = decode_p_frame(ycbcr.shape[:2], motions, residuals, Y_ref, Q, ycbcr[:, :, 1], ycbcr[:, :, 2])
    
    return video

def decode_frame(encoded_data, shape, q):
    """Décode une I-frame"""
    h, w = shape
    Y = np.zeros((h, w), dtype=np.float32)
    
    idx = 0
    for i in range(0, h, BLOCK):
        for j in range(0, w, BLOCK):
            if idx < len(encoded_data['Y']):
                rle_data = encoded_data['Y'][idx]
                # Ici il faudrait décompresser RLE puis déquantifier puis IDCT
                # Pour l'instant, simplification
            idx += 1
    
    # Similar pour Cb, Cr
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :, 0] = Y
    
    return result

def decode_p_frame(shape, motions, residuals, Y_ref, q, Cb, Cr):
    """Décode une P-frame"""
    h, w = shape
    Y_pred = np.zeros((h, w), dtype=np.float32)
    
    mb_idx = 0
    for x in range(0, h, 16):
        for y in range(0, w, 16):
            if x+16 > h or y+16 > w:
                continue
            if mb_idx >= len(motions):
                break
                
            dx, dy = motions[mb_idx]
            
            # Bloc prédit
            pred_x = x + dx
            pred_y = y + dy
            if pred_x < 0: pred_x = 0
            if pred_y < 0: pred_y = 0
            if pred_x+16 > h: pred_x = h-16
            if pred_y+16 > w: pred_y = w-16
            
            pred_block = Y_ref[pred_x:pred_x+16, pred_y:pred_y+16]
            
            # Décoder le résiduel
            residual_sum = np.zeros((16, 16), dtype=np.float32)
            # ... décodage des blocs 8x8
            
            Y_pred[x:x+16, y:y+16] = pred_block + residual_sum
            mb_idx += 1
    
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :, 0] = np.clip(Y_pred, 0, 255)
    result[:, :, 1] = Cb
    result[:, :, 2] = Cr
    
    return result