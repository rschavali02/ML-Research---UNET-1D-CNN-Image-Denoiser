import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import numpy as np
import scipy.io as sio
from model import UNetRes
import matplotlib.pyplot as plt

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("check1")

def compare_snr(batch, batch_hat):
    # Ensure both tensors are on the same device (GPU in this case)
    batch = batch.to(device)
    batch_hat = batch_hat.to(device)
    
    # Initialize a list to store SNR for each sample
    snr_values = []
    
    # Get the total number of elements in dimension 0
    num_samples = batch.shape[0]  # Size of dimension 0 (the dataset size)
    
    # Loop through each sample in the batch
    for i in range(num_samples):  # Loop over dataset size
        x = batch[i]      # Single sample from batch (ground truth)
        xhat = batch_hat[i]  # Corresponding predicted sample
        
        # Compute SNR for this sample directly on GPU using PyTorch operations
        snr = 20 * torch.log10(torch.norm(x) / (torch.norm(x - xhat) + 1e-10)) 
        
        snr_values.append(snr.item())
    return snr_values

# Paths
model_run_directory = '/project/cigserver4/export1/c.rahul/run_20241122_165730/'
model_path = os.path.join(model_run_directory, 'model-20241122_165730-epoch_490.pt')
run_directory = '/project/cigserver4/export1/c.rahul/projects'  
output_file = os.path.join(run_directory, "evaluation_results.txt")

# Load Model
n_channels = 2
model = UNetRes(
    in_nc=n_channels, 
    out_nc=1, 
    nc=[64, 128, 256, 512], 
    nb=4, 
    act_mode='R', 
    downsample_mode="strideconv", 
    upsample_mode="convtranspose"
).float()
model.to(device)
print("check2")

# Load Model Weights
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()  # Switch to evaluation mode

# Load Test Data
trees = sio.loadmat("/project/cigserver4/export1/c.rahul/Indian_pines.mat")
trees = trees["indian_pines"]
test_data = trees[1, 1, :]  # Select a signal to test
train_data = trees[32:, 32:, :].reshape(-1, 1, trees.shape[-1])  # Remaining data as train set
data_max = train_data.max()

# Normalize Test Data
test_data = test_data / data_max
img_gt = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, L]
print("check3")

# Noise Levels to Test
noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]

# Evaluation Loop
results = []
for sigma in noise_levels:
    print(f"Testing with noise level: {sigma}")

    # Add Gaussian Noise
    sigma_channel = sigma * torch.ones_like(img_gt)
    noisy_signal = img_gt + sigma * torch.randn_like(img_gt)
    inputs = torch.cat((noisy_signal, sigma_channel), dim=1).to(device)
    inputs = torch.clamp(inputs, 0, 1)  # Ensure values are in [0, 1]

    # Compute SNR for the noisy signal
    noisy_snr = compare_snr(img_gt, noisy_signal)[0]  # Compute SNR for noisy signal (first sample)

    # Run Model
    with torch.no_grad():
        outputs = model(inputs)

    # Compute SNR for the denoised signal
    denoised_snr = compare_snr(img_gt, outputs)[0]  # Compute SNR for denoised signal (first sample)

    # Append Results
    results.append((sigma, noisy_snr, denoised_snr))
    print(f"Noise Level: {sigma}, Noisy SNR: {noisy_snr:.2f} dB, Denoised SNR: {denoised_snr:.2f} dB")

    # Save Results to File
    with open(output_file, 'a') as f:
        f.write(f"Noise Level: {sigma}, Noisy SNR: {noisy_snr:.2f} dB, Denoised SNR: {denoised_snr:.2f} dB\n")

print("check4")       

# Summary
print("Evaluation Completed.")
for noise_level, noisy_snr, denoised_snr in results:
    print(f"Noise Level: {noise_level}, Noisy SNR: {noisy_snr:.2f} dB, Denoised SNR: {denoised_snr:.2f} dB")

import matplotlib.pyplot as plt

# Separate the data for plotting
noise_levels = [r[0] for r in results]
noisy_snrs = [r[1] for r in results]
denoised_snrs = [r[2] for r in results]

# Create a bar graph
x = np.arange(len(noise_levels))  # Positions for the groups
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, noisy_snrs, width, label='Noisy SNR', color='orange')
bars2 = ax.bar(x + width/2, denoised_snrs, width, label='Denoised SNR', color='blue')

# Add labels, title, and legend
ax.set_xlabel('Noise Level (σ)', fontsize=12)
ax.set_ylabel('SNR (dB)', fontsize=12)
ax.set_title('Noisy vs. Denoised SNR at Different Noise Levels', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"{nl:.2f}" for nl in noise_levels], fontsize=10)
ax.legend(fontsize=10)

# Annotate bars with their values
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text slightly above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)

# Save the graph to file
plot_path = os.path.join(run_directory, "noiseSNR.png")
plt.tight_layout()
plt.savefig(plot_path)
print(f"Bar graph saved at {plot_path}")
