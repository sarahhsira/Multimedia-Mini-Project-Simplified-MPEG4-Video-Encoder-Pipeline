🎬 Simplified MPEG-4 Video Encoder Pipeline
📌 Overview

This project is a simplified multimedia compression system inspired by MPEG-4 concepts.
It processes a sequence of images as a video and demonstrates how raw visual data can be transformed into a compressed binary representation and reconstructed back with minimal loss.

⚙️ Features

🖼️ Frame preprocessing with color space conversion (YCbCr)
🎯 DCT-based intra-frame compression (JPEG-inspired)
📉 Quantization of frequency coefficients
🧭 Motion estimation using macroblocks (P-frame prediction)
🔁 RLE (Run-Length Encoding) for efficient data reduction
🔐 LZW lossless compression for entropy coding
🔄 Reconstruction of compressed visual data

🧠 Educational Goal

This project demonstrates how modern video compression systems work by breaking down the process into simple stages: transforming images into frequency representations, reducing redundancy, compressing data efficiently, and reconstructing visual content.

🛠️ Tech Stack

Python 🐍
OpenCV
NumPy
Matplotlib
