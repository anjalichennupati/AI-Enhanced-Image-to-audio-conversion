
# AI Enhanced Image to Audio Encryption with Data Hiding

# Overview

This project presents a novel approach to secure multimedia data sharing by integrating image-to-audio encryption with AI-enhanced data-hiding techniques. It transforms static images into meaningful audio representations while leveraging adversarial training and reinforcement learning for enhanced security. The framework ensures robust protection against unauthorized access and tampering, offering new possibilities in multimedia communication, artistic expression, and data security. Below is an overview of the analysis, along with sample outputs and results. This project was done in Nov' 2023.





## Publication

- This paper was presented in the “2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT)”
- Link to the IEEE Publication : https://ieeexplore.ieee.org/abstract/document/10725794
- This paper is currently being pushed further for a journal publication


## Block Diagram

- The below block diagram gives an overview of the overall funtionality of the implemented project:
<p align="center">
  <img src="https://i.postimg.cc/SNJ99G1b/Picture1.jpg" alt="App Screenshot" width="450">
</p>

- The below block diagram showcases the encryption-decryption process in detail: 
<p align="center">
  <img src="https://i.postimg.cc/HW2xL7pD/Picture2.jpg" alt="App Screenshot" width="600">
</p>

## Features

- **Steganography & Audio Encryption**: The system integrates image-in-image steganography (Hide-Net & Reveal-Net) with image-to-audio encryption, ensuring multi-layered security. Hidden data remains imperceptible within an image before being encrypted into an audio file for further protection.
<p align="center">
  <img src="https://i.postimg.cc/tRDYRmbN/Picture3.jpg)" alt="App Screenshot" width="350">
</p>

- **U-Net Based Steganography for High Fidelity**: The Hide-Net (U-Net Generator) and Reveal-Net work together to embed and reconstruct images with minimal loss. Skip connections in U-Net preserve high-resolution details, ensuring the container image is indistinguishable from the cover image while maintaining image quality.
<p align="center">
  <img src="https://i.postimg.cc/JnTwBXPt/Picture4.jpg" alt="App Screenshot" width="400">
</p>



- **High Security**: AI-enhanced data hiding, adversarial training, and cryptographic principles make brute-force attacks impractical. The system achieves high PSNR (>40 dB) and SSIM (>0.95), ensuring quality preservation while maintaining strong resistance to detection and tampering.




## Tech Stack

- **Python** – Core language for implementing steganography, encryption, and AI models.
- **OpenCV** – Image preprocessing, manipulation, and visualization.
- **Pydub** – Audio signal processing
- **Matplotlib & Seaborn** – Plotting loss curves, comparisons, and visualizing training results.


## Installation

1. **Load the Dataset**:
- This project does not contain a pre defined dataset. The input is any image that the user wants to hide content using encryption-decryption techniques.

2. Run the files `audio_pred.py` to get the container image, output audio and reconstructed image.

3. `Hide_Net.py` and `Reveal_net.py` contain the neueral network architectures.







## Running Tests

The project can be implemented and tested to verify funtionality

