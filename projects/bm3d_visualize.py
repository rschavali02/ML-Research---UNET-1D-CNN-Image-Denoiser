# Import necessary libraries
import os

import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import matplotlib.pyplot as plt
from bm3d import bm3d, BM3DProfile  # Ensure you have the BM3D library installed
import scipy.io as sio

# Set up directories
run_directory = '/project/cigserver4/export1/c.rahul/projects'

# Load the dataset (assumes the MATLAB file contains a hyperspectral cube)
data_path = "/project/cigserver4/export1/c.rahul/Indian_pines.mat"
trees = sio.loadmat(data_path)
trees = trees["indian_pines"]  # Hyperspectral cube: [H, W, C]

# Define training and testing splits
train_data = trees[32:, 32:, :]  
test_data = trees[:32, :32, :]  

# Normalize the data by the maximum value
data_max = train_data.max()
train_data = train_data / data_max
test_data = test_data / data_max

# Add noise to the test data
noise_level = 0.5
noisy_test_data = test_data + noise_level * np.random.randn(*test_data.shape)
noisy_test_data = np.clip(noisy_test_data, 0, 1) 

# Save noisy and ground truth images for visualization
gt_image_path = os.path.join(run_directory, "ground_truth.png")
noisy_image_path = os.path.join(run_directory, "noisy_image.png")

# Plot and save a band for ground truth and noisy images (Same band as used with visualizing my model)
plt.figure()
plt.title("Ground Truth (Band 10)")
plt.imshow(test_data[:, :, 10], cmap="gray")  # Display band 10
plt.colorbar()
plt.savefig(gt_image_path)

plt.figure()
plt.title("Noisy Image (Band 10)")
plt.imshow(noisy_test_data[:, :, 10], cmap="gray")  # Display band 10
plt.colorbar()
plt.savefig(noisy_image_path)

# Compare SNR
compare_snr_2d = lambda x, xhat: 20 * np.log10(
    np.linalg.norm(x.flatten('F')) / np.linalg.norm(x.flatten('F') - xhat.flatten('F')))

total_cleanSNR = []
total_noisySNR = []
denoised_cube = []
# Define the BM3D denoising function
def denoise_hypercube_with_bm3d(noisy_hypercube, sigma_psd=0.1):
    """
    Apply BM3D denoising to each spectral band of a hyperspectral image. 
    :return: [H, W, C] Denoised hyperspectral cube.
    """
    for band_idx in range(noisy_hypercube.shape[-1]):
        ground_truth = test_data[:, :, band_idx]
        noisy_band = noisy_hypercube[:, :, band_idx]  # Extract the band
        snr_noisy = compare_snr_2d(ground_truth, noisy_band)
        total_noisySNR.append(snr_noisy)
        denoised_band = bm3d(noisy_band, sigma_psd, profile=BM3DProfile())  # Apply BM3D
        snr_clean= compare_snr_2d(ground_truth, denoised_band)
        total_cleanSNR.append(snr_clean)
        denoised_cube.append(denoised_band)
    return np.stack(denoised_cube, axis=-1)

# Apply BM3D denoising
sigma_psd = noise_level  # Set BM3D noise standard deviation to match synthetic noise
denoised_data_bm3d = denoise_hypercube_with_bm3d(noisy_test_data, sigma_psd)


# Calculate SNR values
noisy_snr = (np.sum(total_noisySNR))/220
bm3d_snr =  (np.sum(total_cleanSNR))/220

# Save results
bm3d_result_path = os.path.join(run_directory, f"bm3d_results_{noise_level}")
os.makedirs(bm3d_result_path, exist_ok=True)

bm3d_image_path = os.path.join(bm3d_result_path, "denoised_bm3d.npy")
np.save(bm3d_image_path, denoised_data_bm3d)

# Visualize results
plt.figure(figsize=(15, 5))

# Ground truth
plt.subplot(1, 3, 1)
plt.title("Ground Truth (Band 10)")
plt.imshow(test_data[:, :, 10], cmap="gray")
plt.colorbar()

# Noisy image
plt.subplot(1, 3, 2)
plt.title("Noisy Image (Band 10)")
plt.imshow(noisy_test_data[:, :, 10], cmap="gray")
plt.colorbar()

# BM3D denoised image
plt.subplot(1, 3, 3)
plt.title("BM3D Denoised (Band 10)")
plt.imshow(denoised_data_bm3d[:, :, 10], cmap="gray")
plt.colorbar()

plt.tight_layout()
bm3d_plot_path = os.path.join(bm3d_result_path, "bm3d_denoising_results.png")
plt.savefig(bm3d_plot_path)

# Visualize the entire 3D dataset
plt.figure(figsize=(20, 5))
plt.title("Entire 3D Hyperspectral Dataset")
plt.imshow(test_data[:,:,10], aspect='auto')
plt.colorbar()
plt.savefig(os.path.join(run_directory, "full_3d_dataset.png"))
plt.close()

print(f"BM3D results saved at {bm3d_result_path}")
print(f"Noisy SNR: {noisy_snr:.2f} dB")
print(f"BM3D Denoised SNR: {bm3d_snr:.2f} dB")

print(denoised_data_bm3d.shape)
print(trees.shape)
print(noisy_test_data.shape)

three_n_snr = compare_snr_2d(test_data,noisy_test_data[:, :, :])
three_dn_snr = compare_snr_2d(test_data,denoised_data_bm3d[:, :, :])

print(f"BM3C 3D Noisy SNR (Total Dataset): {three_n_snr}")
print(f"BM3C 3D Denoised SNR (Total Dataset): {three_dn_snr}")

with open(os.path.join(bm3d_result_path, "bm3d_snr.txt"), "w") as f:
    f.write(f"Noisy SNR: {noisy_snr:.2f} dB\n")
    f.write(f"BM3D Denoised SNR: {bm3d_snr:.2f} dB\n")
    f.write(f"3D Noisy SNR: {three_n_snr:.2f} dB\n")
    f.write(f"3D BM3D Denoised SNR: {three_dn_snr:.2f} dB\n")
