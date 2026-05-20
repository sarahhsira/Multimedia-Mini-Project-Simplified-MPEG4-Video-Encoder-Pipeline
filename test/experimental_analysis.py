# générer les graphes du taux de compression selon Q et le GOP 

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

from encoder.encode import encode_folder
from encoder.lzw import lzw_encode_bytes
from utils import compression_ratio, compression_percentage, psnr

FRAMES_PATH = "test/frames"
OUTPUT_PATH = "test/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def get_original_size(frames_path):
    total = 0
    for f in sorted(os.listdir(frames_path)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(frames_path, f))
            if img is not None:
                img = cv2.resize(img, (256, 256))
                total += img.shape[0] * img.shape[1] * img.shape[2]
    return total


def encode_and_measure(frames_path, q_val, gop_val):
    
    import encoder.encode as enc_mod
    import encoder.motion as mot_mod

    # Modifier les constantes Q et GOP 
    orig_Q   = enc_mod.Q
    orig_GOP = enc_mod.GOP

    enc_mod.Q   = q_val
    enc_mod.GOP = gop_val

    try:
        video = encode_folder(frames_path)
        data  = pickle.dumps(video)
        codes = lzw_encode_bytes(data)
        compressed_size = len(np.array(codes, dtype=np.uint32).tobytes())
    finally:
        enc_mod.Q   = orig_Q
        enc_mod.GOP = orig_GOP

    return compressed_size

# GRAPHE 1 — Compression ratio vs Q
def plot_ratio_vs_q(frames_path, output_path):
    print("\n[1/2] Compression ratio vs Q ...")

    Q_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    ratios   = []
    gains    = []

    original_size = get_original_size(frames_path)

    for q in Q_values:
        print(f"  Q = {q} ...", end=" ", flush=True)
        comp_size = encode_and_measure(frames_path, q_val=q, gop_val=3)
        r = compression_ratio(original_size, comp_size)
        g = compression_percentage(original_size, comp_size)
        ratios.append(r)
        gains.append(g)
        print(f"ratio = {r:.2f}:1  ({g:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Compression ratio vs Quantization Factor (Q)", fontsize=14, fontweight='bold')

    # Sous-graphe gauche : ratio
    ax1.plot(Q_values, ratios, 'o-', color='steelblue', linewidth=2, markersize=7)
    ax1.set_xlabel("Quantization Factor (Q)", fontsize=12)
    ax1.set_ylabel("Compression Ratio (x:1)", fontsize=12)
    ax1.set_title("Taux de compression", fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.set_xticks(Q_values)
    for x, y in zip(Q_values, ratios):
        ax1.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9)

    # Sous-graphe droit : gain %
    ax2.bar(Q_values, gains, color='steelblue', alpha=0.75, width=6)
    ax2.set_xlabel("Quantization Factor (Q)", fontsize=12)
    ax2.set_ylabel("Gain d'espace (%)", fontsize=12)
    ax2.set_title("Pourcentage d'espace économisé", fontsize=11)
    ax2.grid(True, alpha=0.4, axis='y')
    ax2.set_xticks(Q_values)
    for x, y in zip(Q_values, gains):
        ax2.text(x, y + 0.5, f"{y:.1f}%", ha='center', fontsize=9)

    # Annotation : Q=50 
    default_idx = Q_values.index(50)
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Q=50 (défaut)')
    ax1.legend(fontsize=9)
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Q=50 (défaut)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_path, "graph_ratio_vs_Q.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  → Sauvegardé : {save_path}")
    return fig


# GRAPHE 2 — Compression ratio vs GOP size
def plot_ratio_vs_gop(frames_path, output_path):
    print("\n[2/2] Compression ratio vs GOP size ...")

    GOP_values = [1, 2, 3, 4, 5, 6, 8, 10]
    ratios     = []
    gains      = []

    original_size = get_original_size(frames_path)

    for gop in GOP_values:
        print(f"  GOP = {gop} ...", end=" ", flush=True)
        comp_size = encode_and_measure(frames_path, q_val=50, gop_val=gop)
        r = compression_ratio(original_size, comp_size)
        g = compression_percentage(original_size, comp_size)
        ratios.append(r)
        gains.append(g)
        print(f"ratio = {r:.2f}:1  ({g:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Compression ratio vs GOP Size", fontsize=14, fontweight='bold')

    # Sous-graphe gauche : ratio
    ax1.plot(GOP_values, ratios, 's-', color='darkorange', linewidth=2, markersize=7)
    ax1.set_xlabel("GOP Size (G)", fontsize=12)
    ax1.set_ylabel("Compression Ratio (x:1)", fontsize=12)
    ax1.set_title("Taux de compression", fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.set_xticks(GOP_values)
    for x, y in zip(GOP_values, ratios):
        ax1.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9)

    # Annotation GOP=1 = que des I-frames
    ax1.axvline(x=1, color='gray', linestyle=':', alpha=0.7, label='GOP=1 (100% I-frames)')
    ax1.axvline(x=3, color='red',  linestyle='--', alpha=0.5, label='GOP=3 (défaut)')
    ax1.legend(fontsize=9)

    # Sous-graphe droit : gain %
    colors = ['darkorange' if g != 3 else 'red' for g in GOP_values]
    bars = ax2.bar(GOP_values, gains, color=colors, alpha=0.75, width=0.6)
    ax2.set_xlabel("GOP Size (G)", fontsize=12)
    ax2.set_ylabel("Gain d'espace (%)", fontsize=12)
    ax2.set_title("Pourcentage d'espace économisé", fontsize=11)
    ax2.grid(True, alpha=0.4, axis='y')
    ax2.set_xticks(GOP_values)
    for x, y in zip(GOP_values, gains):
        ax2.text(x, y + 0.3, f"{y:.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_path, "graph_ratio_vs_GOP.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  → Sauvegardé : {save_path}")
    return fig

# main
if __name__ == "__main__":
    print("=" * 55)
    print("ANALYSE EXPÉRIMENTALE — MPEG-4 ENCODER")
    print("=" * 55)

    if not os.path.exists(FRAMES_PATH):
        print(f"ERREUR : Dossier introuvable : {FRAMES_PATH}")
        print("Modifie la variable FRAMES_PATH en haut du fichier.")
        exit(1)

    plot_ratio_vs_q(FRAMES_PATH, OUTPUT_PATH)
    plot_ratio_vs_gop(FRAMES_PATH, OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("ANALYSE TERMINÉE")
    print(f"Graphes sauvegardés dans : {OUTPUT_PATH}/")
    print("  - graph_ratio_vs_Q.png")
    print("  - graph_ratio_vs_GOP.png")
    print("=" * 55)
