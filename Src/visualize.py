import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_complete_pipeline(frames_path, video_data, motion_vectors, 
                                  ycbcr_img, dct_pipeline_data, residual_img, reconstructed_img):
    
    # Figure simple et élégante
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Pipeline complet d\'encodage vidéo MPEG-4', fontsize=16, fontweight='bold', y=0.98)
    
    # =========================================================
    # 1. TOUTES LES IMAGES ORIGINALES (6 frames)
    # =========================================================
    frames = sorted([f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    axes[0, 0].clear()
    
    # Afficher les 6 images en grille 2x3
    positions_x = [25, 120, 215]
    positions_y = [20, 120]
    
    for idx, f in enumerate(frames[:6]):
        img = cv2.imread(os.path.join(frames_path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (80, 80))
        
        row = idx // 3
        col = idx % 3
        x = positions_x[col]
        y = positions_y[row]
        
        axes[0, 0].imshow(img, extent=[x, x+80, y+80, y])
        axes[0, 0].text(x+40, y+92, f'Frame {idx}', fontsize=9, ha='center')
    
    axes[0, 0].set_xlim(0, 310)
    axes[0, 0].set_ylim(220, -5)
    axes[0, 0].set_title('1. Images originales', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # GOP, Q, Search en bas à gauche - sans cadre
    axes[0, 0].text(5, 210, 'GOP = 3', fontsize=9, color='gray')
    axes[0, 0].text(5, 198, 'Q = 50', fontsize=9, color='gray')
    axes[0, 0].text(5, 186, 'Search = ±8', fontsize=9, color='gray')
    
    # =========================================================
    # 2. ESPACE YCbCr - BIEN SÉPARÉ avec traits fins
    # =========================================================
    Y, Cb, Cr = cv2.split(ycbcr_img)
    
    if Y.shape != Cb.shape:
        Cb = cv2.resize(Cb, (Y.shape[1], Y.shape[0]))
        Cr = cv2.resize(Cr, (Y.shape[1], Y.shape[0]))
    
    h, w = Y.shape
    margin = 35
    
    # Afficher les 3 canaux
    axes[0, 1].imshow(Y, cmap='gray', extent=[0, w, h, 0])
    axes[0, 1].imshow(Cb, cmap='gray', extent=[w+margin, 2*w+margin, h, 0])
    axes[0, 1].imshow(Cr, cmap='gray', extent=[2*w+2*margin, 3*w+2*margin, h, 0])
    
    axes[0, 1].set_xlim(0, 3*w+2*margin)
    axes[0, 1].set_ylim(h, -5)
    axes[0, 1].set_title('2. Espace YCbCr', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Labels simples
    axes[0, 1].text(w//2, h+30, 'Y', fontsize=11, ha='center')
    axes[0, 1].text(w+margin + w//2, h+30, 'Cb', fontsize=11, ha='center')
    axes[0, 1].text(2*w+2*margin + w//2, h+30, 'Cr', fontsize=11, ha='center')
    
    # Traits fins de séparation
    axes[0, 1].axvline(x=w, color='gray', linewidth=1, linestyle='--')
    axes[0, 1].axvline(x=w+margin+w, color='gray', linewidth=1, linestyle='--')
    
        # =========================================================
    # 3. DCT & QUANTIFICATION - AVEC ESPACES ENTRE LES BLOCS
    # =========================================================
    
    # Créer un tableau plus large pour avoir de l'espace (8x60)
    display = np.zeros((8, 60))
    
    # Bloc 1: BRUT (colonnes 2 à 9) - avec marge à gauche
    display[:, 2:10] = dct_pipeline_data['original']
    
    # Bloc 2: DCT (colonnes 18 à 25) - espace de 8 colonnes (10 à 17)
    display[:, 18:26] = dct_pipeline_data['dct_norm']
    
    # Bloc 3: QUANT (colonnes 34 à 41) - espace de 8 colonnes (26 à 33)
    display[:, 34:42] = dct_pipeline_data['quantized_norm']
    
    # Bloc 4: RECONST (colonnes 50 à 57) - espace de 8 colonnes (42 à 49)
    display[:, 50:58] = dct_pipeline_data['reconstructed']
    
    axes[0, 2].imshow(display, cmap='gray')
    axes[0, 2].set_title('3. DCT + Quantification (bloc 8x8)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # =========================================================
    # TEXTES SUR 2 LIGNES BIEN ESPACÉS
    # =========================================================
    
    # Première ligne (y=11)
    axes[0, 2].text(6, 11, 'Bloc 8x8', fontsize=8, ha='center')
    axes[0, 2].text(22, 11, 'Coefficients', fontsize=8, ha='center')
    axes[0, 2].text(38, 11, 'Coefficients', fontsize=8, ha='center')
    axes[0, 2].text(54, 11, 'Bloc', fontsize=8, ha='center')
    
    # Deuxième ligne (y=18)
    axes[0, 2].text(6, 18, 'brut', fontsize=8, ha='center')
    axes[0, 2].text(22, 18, 'DCT', fontsize=8, ha='center')
    axes[0, 2].text(38, 18, 'quantifiés', fontsize=8, ha='center')
    axes[0, 2].text(54, 18, 'reconstruit', fontsize=8, ha='center')
    
    # =========================================================
    # FLÈCHES ENTRE LES BLOCS
    # =========================================================
    axes[0, 2].annotate('→', xy=(15, 5), xytext=(12, 5), fontsize=12, color='gray')
    axes[0, 2].annotate('→', xy=(31, 5), xytext=(28, 5), fontsize=12, color='gray')
    axes[0, 2].annotate('→', xy=(47, 5), xytext=(44, 5), fontsize=12, color='gray')
    
    # =========================================================
    # TRAITS DE SÉPARATION VERTICAUX (optionnel)
    # =========================================================
    axes[0, 2].axvline(x=10, color='lightgray', linewidth=0.8, linestyle='--')
    axes[0, 2].axvline(x=26, color='lightgray', linewidth=0.8, linestyle='--')
    axes[0, 2].axvline(x=42, color='lightgray', linewidth=0.8, linestyle='--')
    
    # =========================================================
    # 4. VECTEURS DE MOUVEMENT - simple, sans cadre coloré
    # =========================================================
    first_frame = cv2.imread(os.path.join(frames_path, frames[0]))
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_frame_resized = cv2.resize(first_frame_gray, (256, 256))
    
    axes[1, 0].imshow(first_frame_resized, cmap='gray')
    axes[1, 0].set_title('4. Vecteurs de mouvement', fontsize=12, fontweight='bold')
    
    if motion_vectors:
        step = 16
        idx = 0
        for x in range(0, 256, step):
            for y in range(0, 256, step):
                if idx < len(motion_vectors):
                    dx, dy = motion_vectors[idx]
                    if abs(dx) > 0 or abs(dy) > 0:
                        axes[1, 0].arrow(y, x, dy*2, dx*2, color='red', 
                                       head_width=3, head_length=3, alpha=0.6, linewidth=0.8)
                idx += 1
    
    axes[1, 0].axis('off')
    # Texte simple en bas
    axes[1, 0].text(128, 275, f'{len(motion_vectors)} vecteurs', fontsize=9, ha='center', color='gray')
    
    # =========================================================
    # 5. CARTE DE RÉSIDUEL
    # =========================================================
    im = axes[1, 1].imshow(residual_img, cmap='RdBu', vmin=-40, vmax=40)
    axes[1, 1].set_title('5. Carte de résiduel', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.05, pad=0.03)
    cbar.set_label('', fontsize=8)
    
    # =========================================================
    # 6. IMAGE RECONSTRUITE - texte simple en bas
    # =========================================================
    axes[1, 2].imshow(reconstructed_img, cmap='gray')
    axes[1, 2].set_title('6. Image reconstruite', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    axes[1, 2].text(128, 275, 'Après décodage', fontsize=9, ha='center', color='gray')
    
    # Espacement
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, wspace=0.3, hspace=0.3)
    
    # Sauvegarde
    output_dir = os.path.join(frames_path, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pipeline_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Visualisation sauvegardée: {save_path}")


def show_final_pipeline(original, ycbcr, dct_block, dct_q, dct_rec, 
                        motion_img, residual, reconstructed):
   
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pipeline de décodage vidéo MPEG-4', fontsize=14, fontweight='bold')
    
    # 1. Image originale
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('1. Image originale')
    axes[0, 0].axis('off')
    
    # 2. Espace YCbCr (Y seulement pour l'affichage)
    axes[0, 1].imshow(ycbcr[:,:,0] if len(ycbcr.shape) > 2 else ycbcr, cmap='gray')
    axes[0, 1].set_title('2. Canal Y (Luminance)')
    axes[0, 1].axis('off')
    
    # 3. Pipeline DCT
    display = np.zeros((8, 32))
    display[:, 0:8] = dct_block
    display[:, 8:16] = dct_q
    display[:, 16:24] = dct_rec
    axes[0, 2].imshow(display, cmap='gray')
    axes[0, 2].set_title('3. DCT → Quant → IDCT')
    axes[0, 2].axis('off')
    
    # 4. Vecteurs de mouvement
    axes[1, 0].imshow(motion_img, cmap='gray')
    axes[1, 0].set_title('4. Image décodée')
    axes[1, 0].axis('off')
    
    # 5. Résiduel
    axes[1, 1].imshow(residual, cmap='RdBu')
    axes[1, 1].set_title('5. Carte de résiduel')
    axes[1, 1].axis('off')
    
    # 6. Reconstruction
    axes[1, 2].imshow(reconstructed, cmap='gray')
    axes[1, 2].set_title('6. Image reconstruite')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()