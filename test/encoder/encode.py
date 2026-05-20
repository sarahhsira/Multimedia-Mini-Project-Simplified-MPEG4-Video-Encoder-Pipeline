import os
import numpy as np
from encoder.preprocess import load_image, resize, to_ycbcr
from encoder.dct import dct_2d, quantize, dequantize, idct_2d
from encoder.rle import rle_encode, rle_decode
from encoder.motion import block_matching

BLOCK = 8
GOP = 3
Q = 50



# Encodage  I-FRAME
def encode_frame(img, q=Q):

    ycbcr = to_ycbcr(img)
    compressed = {}

    for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
        channel = ycbcr[:, :, channel_idx].astype(np.float32)
        h, w = channel.shape
        channel_data = []

        for i in range(0, h, BLOCK):
            for j in range(0, w, BLOCK):
                block = channel[i:i + BLOCK, j:j + BLOCK]

                if block.shape != (BLOCK, BLOCK):
                    padded = np.zeros((BLOCK, BLOCK), dtype=np.float32)
                    padded[:block.shape[0], :block.shape[1]] = block
                    block = padded

                dct = dct_2d(block)
                q_block = quantize(dct, q)
                channel_data.append(rle_encode(q_block.flatten().tolist()))

        compressed[channel_name] = channel_data

    return compressed

# Decodage I-FRAME  
def decode_frame(encoded_data, shape, q=Q):
    h, w = shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
        channel_blocks = encoded_data[channel_name]
        channel_img = np.zeros((h, w), dtype=np.float32)
        idx = 0

        for i in range(0, h, BLOCK):
            for j in range(0, w, BLOCK):
                if idx >= len(channel_blocks):
                    break

                #  RLE decode
                flat = rle_decode(channel_blocks[idx])
                idx += 1

                if len(flat) != BLOCK * BLOCK:
                    continue

                #  Déquantification
                arr = np.array(flat, dtype=np.int16).reshape((BLOCK, BLOCK))
                deq = dequantize(arr, q)

                # IDCT
                block = idct_2d(deq).astype(np.float32)

                i_end = min(i + BLOCK, h)
                j_end = min(j + BLOCK, w)
                channel_img[i:i_end, j:j_end] = block[:i_end - i, :j_end - j]

        result[:, :, channel_idx] = np.clip(channel_img, 0, 255).astype(np.uint8)

    return result



# Decodage P-FRAME 
def decode_p_frame(shape, motions, residuals, Y_ref, q, Cb, Cr):

    h, w = shape
    Y_pred = np.zeros((h, w), dtype=np.float32)

    mb_idx = 0
    for x in range(0, h, 16):
        for y in range(0, w, 16):
            if x + 16 > h or y + 16 > w:
                continue
            if mb_idx >= len(motions):
                break

            dx, dy = motions[mb_idx]

            pred_x = max(0, min(x + dx, h - 16))
            pred_y = max(0, min(y + dy, w - 16))
            pred_block = Y_ref[pred_x:pred_x + 16, pred_y:pred_y + 16].astype(np.float32)

            residual_sum = np.zeros((16, 16), dtype=np.float32)

            if mb_idx < len(residuals):
                for i2, res_rle in enumerate(residuals[mb_idx]):
                    flat = rle_decode(res_rle)
                    if len(flat) != 64:
                        continue
                    arr = np.array(flat, dtype=np.int16).reshape((8, 8))
                    deq = dequantize(arr, q)
        
                    import cv2
                    res = cv2.idct(np.float32(deq))
                    row = (i2 // 2) * 8
                    col = (i2 % 2) * 8
                    residual_sum[row:row + 8, col:col + 8] = res

            Y_pred[x:x + 16, y:y + 16] = np.clip(pred_block + residual_sum, 0, 255)
            mb_idx += 1

    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :, 0] = np.clip(Y_pred, 0, 255).astype(np.uint8)
    result[:, :, 1] = np.clip(Cb, 0, 255).astype(np.uint8)
    result[:, :, 2] = np.clip(Cr, 0, 255).astype(np.uint8)
    return result


# Encodage  de toutes les frames 
def encode_folder(folder):

    frames = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    video = []
    prev_decoded = None

    for i, f in enumerate(frames):
        print(f"  Encodage frame {i + 1}/{len(frames)}: {f}")
        img = load_image(os.path.join(folder, f))
        img = resize(img)
        ycbcr = to_ycbcr(img)

        # I-FRAME 
        if i % GOP == 0 or prev_decoded is None:
            encoded = encode_frame(img, q=Q)
            video.append(("I", encoded, {"q": Q}))
            prev_decoded = decode_frame(encoded, ycbcr.shape[:2], Q)

        # P-FRAME 
        else:
            Y_current = ycbcr[:, :, 0].astype(np.float32)
            Y_ref = prev_decoded[:, :, 0].astype(np.float32)
            h, w = Y_current.shape
            motions = []
            residuals = []

            for x in range(0, h, 16):
                for y in range(0, w, 16):
                    if x + 16 > h or y + 16 > w:
                        continue

                    dx, dy = block_matching(Y_current, Y_ref, x, y)
                    motions.append((dx, dy))

                    pred_x = max(0, min(x + dx, h - 16))
                    pred_y = max(0, min(y + dy, w - 16))
                    pred_block = Y_ref[pred_x:pred_x + 16, pred_y:pred_y + 16]
                    curr_block = Y_current[x:x + 16, y:y + 16]

                    residual = curr_block - pred_block

                    import cv2
                    res_blocks = []
                    for i2 in range(0, 16, 8):
                        for j2 in range(0, 16, 8):
                            sub = residual[i2:i2 + 8, j2:j2 + 8]
                            dct_sub = cv2.dct(np.float32(sub))
                            q_sub = quantize(dct_sub, Q)
                            res_blocks.append(rle_encode(q_sub.flatten().tolist()))
                    residuals.append(res_blocks)

            video.append(("P", motions, residuals, {"q": Q}))

            prev_decoded = decode_p_frame(
                ycbcr.shape[:2], motions, residuals,
                Y_ref, Q,
                ycbcr[:, :, 1], ycbcr[:, :, 2]
            )

    return video
