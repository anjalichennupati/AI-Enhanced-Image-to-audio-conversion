import cv2
from scipy.io import wavfile
import random
from matplotlib import pyplot as plt
import numpy as np
from pydub import AudioSegment
import io
import os
import platform
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
from PIL import Image
from models.Hide_Net import UnetGenerator
from models.Reveal_Net import RevealNet
import utils.transform as transforms


import torch.nn.functional as F

# Function to calculate Pixel error


def calculate_pixel_error(image_path1, image_path2):
    # Load images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Convert images to numpy arrays
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    # Calculate absolute pixel-wise difference
    pixel_diff = np.abs(image1_np.astype(float) - image2_np.astype(float))

    # Calculate mean absolute error
    mae = np.mean(pixel_diff)

    # Normalize MAE to [0, 1]
    normalized_mae = mae / 255.0

    return normalized_mae


def predict(cover_path, secret_path, imageSize, Hnet, device):
    Hnet.eval().to(device)
    # Load cover and secret images
    cover_img = Image.open(cover_path)
    secret_img = Image.open(secret_path)

    # Define the transformation to be applied to the images
    transform = transforms.Compose([
        transforms.Resize([imageSize, imageSize]),
        transforms.ToTensor(),
    ])

    # Apply the transformation
    cover_img = transform(cover_img).unsqueeze(0).to(device)
    secret_img = transform(secret_img).unsqueeze(0).to(device)

    # Concatenate cover and secret images
    concat_img = torch.cat((cover_img, secret_img), dim=1).to(device)

    with torch.no_grad():
        concat_imgv = Variable(concat_img)

    with torch.no_grad():
        cover_imgv = Variable(cover_img)

    # Take concat_img as input of H-net and get the container_img
    container_img = Hnet(concat_imgv)

    # Calculate loss between input cover image and container image
    cover_container_loss = F.mse_loss(cover_imgv, container_img).item()
    print(
        f"Loss between input cover image and container image: {cover_container_loss:.6f}")

    # Visualize the differences between cover and container images
    visualize_difference(cover_imgv, container_img,
                         "Cover Image", "Container Image")

    # Save the container image
    container_img = container_img.squeeze(0).detach().cpu()
    container_img = transforms.ToPILImage()(container_img)
    container_img.save('container_image.png')

    return cover_container_loss


def reveal(container_path, Rnet, device, secret_path):
    Rnet.eval().to(device)
    # Load the container image and original secret image
    container_img = Image.open(container_path)
    secret_img = Image.open(secret_path)

    # Define the transformation to be applied to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply the transformation
    container_img = transform(container_img).unsqueeze(0).to(device)
    secret_img = transform(secret_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Duplicate the container image
        container_img = container_img

    # Take container_img as input of R-net and get the revealed_img
    revealed_img = Rnet(container_img)

    # Calculate loss between original secret image and revealed image
    secret_reveal_loss = F.mse_loss(secret_img, revealed_img).item()
    print(
        f"Loss between original secret image and revealed image: {secret_reveal_loss:.6f}")

    # Visualize the differences between secret and revealed images
    visualize_difference(secret_img, revealed_img,
                         "Secret Image", "Revealed Image")

    # Save the revealed image
    revealed_img = revealed_img.squeeze(0).detach().cpu()
    revealed_img = transforms.ToPILImage()(revealed_img)
    revealed_img.save('revealed_image.png')

    return secret_reveal_loss


def calculate_mean_pixel_error(image_paths1, image_paths2):
    # Ensure the number of image pairs match
    assert len(image_paths1) == len(
        image_paths2), "Number of image pairs must be the same."

    # Calculate pixel error for each pair
    pixel_errors = [calculate_pixel_error(
        path1, path2) for path1, path2 in zip(image_paths1, image_paths2)]

    # Calculate the mean pixel error
    mean_pixel_error = np.mean(pixel_errors)

    return mean_pixel_error


def visualize_difference(image1, image2, label1, label2):
    # Convert images to numpy arrays
    img1_np = image1.squeeze().detach().cpu().numpy()
    img2_np = image2.squeeze().detach().cpu().numpy()

    # Compute the absolute difference
    diff = np.abs(img1_np - img2_np)

    # Display the images and the difference
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(np.transpose(img1_np, (1, 2, 0)))
    axs[0].set_title(label1)
    axs[0].axis('off')

    axs[1].imshow(np.transpose(img2_np, (1, 2, 0)))
    axs[1].set_title(label2)
    axs[1].axis('off')

    axs[2].imshow(np.transpose(diff, (1, 2, 0)), cmap='gray')
    axs[2].set_title("Absolute Difference")
    axs[2].axis('off')

    plt.show()


def extract_rgb_values(image):
    # Extract RGB values of each pixel
    r_values = image[:, :, 0].flatten()
    g_values = image[:, :, 1].flatten()
    b_values = image[:, :, 2].flatten()

    return r_values, g_values, b_values


def create_grayscale_audio(grayscale_image, audio_file, duration_factor=1.0):
    # Convert grayscale values to audio signals
    audio_signal = (grayscale_image * 32767).astype(np.int16)

    # Increase the duration of the audio by a factor
    increased_duration_signal = np.tile(
        audio_signal, int(len(audio_signal) * duration_factor)//10)

    audio_segment = AudioSegment(
        increased_duration_signal.tobytes(),
        frame_rate=44100,
        sample_width=2,  # 16-bit sample width
        channels=1
    )
    # Save the audio signal to a WAV file
    audio_segment.export(audio_file, format='wav')


def reconstruct_colored_image(grayscale_audio, r_values, g_values, b_values, duration_factor=1.0):
    # Read grayscale audio signal
    sampling_rate, grayscale_audio_signal = wavfile.read(grayscale_audio)

    # Adjust the duration factor during reconstruction
    adjusted_duration_signal = grayscale_audio_signal[:int(
        len(grayscale_audio_signal) * duration_factor)]

    # Ensure that the reshaped array has the correct size
    target_shape = (256, 256)
    reshaped_size = target_shape[0] * target_shape[1]

    # Reshape the adjusted grayscale audio signal
    reshaped_grayscale_audio_signal = adjusted_duration_signal[:reshaped_size].reshape(
        target_shape)

    # Add back the original RGB values to obtain the colored image
    reconstructed_image = np.zeros((256, 256, 3), dtype=np.uint8)
    reconstructed_image[:, :, 0] = r_values.reshape(256, 256)
    reconstructed_image[:, :, 1] = g_values.reshape(256, 256)
    reconstructed_image[:, :, 2] = b_values.reshape(256, 256)

    return reconstructed_image


# Example usage:
# Load the trained models
Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7,
                     output_function=torch.nn.Sigmoid)
Rnet = RevealNet(output_function=torch.nn.Sigmoid)

# Load the pre-trained weights
Hnet.load_state_dict(torch.load(
    "E:/ASEB/3rd Year/Signal and Image Processing/training/LAPTOP-FJM61C1T_2023-12-16-11_58_23/checkPoints/netH_epoch_99,sumloss=0.001365,Hloss=0.000897.pth"))
Rnet.load_state_dict(torch.load(
    "E:/ASEB/3rd Year/Signal and Image Processing/training/LAPTOP-FJM61C1T_2023-12-16-11_58_23/checkPoints/netR_epoch_99,sumloss=0.001365,Rloss=0.000623.pth"))

# Set the models to evaluation mode
Hnet.eval()
Rnet.eval()

# Provide paths to cover and secret images
cover_image_path = 'E:/ASEB/3rd Year/Signal and Image Processing/New folder/Photos/0a284dbed0.jpg'
secret_image_path = 'E:/ASEB/3rd Year/Signal and Image Processing/New folder/Linnaeus 5 256X256/2_256.jpg'


if platform.system() == "Windows":
    os.startfile(cover_image_path)


if platform.system() == "Windows":
    os.startfile(secret_image_path)

# Make a prediction
cover_container_loss = predict(cover_image_path, secret_image_path, 256, Hnet,
                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# print("Loss between input cover image and container image:", cover_container_loss)


output_path = 'container_image.png'
output_audio_path = 'grayscale_audio.wav'
output_image_path = 'reconstructed_colored_image.png'
pixel_cover_container = calculate_pixel_error(cover_image_path, output_path)

print(f"Pixel Error (Cover-Container): {pixel_cover_container:.2f}")


rgb_image = cv2.imread(output_path)

# Extract RGB values
r_values, g_values, b_values = extract_rgb_values(rgb_image)

# Convert the RGB image to grayscale
grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Calculate the mean of the RGB image for the duration factor
mean_duration_factor = np.mean(rgb_image)//20
# Create audio signal from grayscale image with increased duration using the mean as duration factor
create_grayscale_audio(
    grayscale_image, output_audio_path, mean_duration_factor)

# Reconstruct the colored image from grayscale audio and original RGB values
reconstructed_image = reconstruct_colored_image(
    output_audio_path, r_values, g_values, b_values, mean_duration_factor)

# Save the reconstructed image
cv2.imwrite(output_image_path, reconstructed_image)

# Reveal the secret image
secret_reveal_loss = reveal(output_image_path, Rnet,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), secret_path=secret_image_path)
# print("Loss between original secret image and revealed image:", secret_reveal_loss)
pixel_secret_reveal = calculate_pixel_error(
    secret_image_path, 'revealed_image.png')
print(f"Pixel Error (Secret-Reveal): {pixel_secret_reveal:.2f}")


pixel_errors = [pixel_cover_container, pixel_secret_reveal]
print("The mean pixel error of this run: ", np.mean(pixel_errors))
