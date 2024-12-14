import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNetRes 
from datetime import datetime
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# Specify the directory where model and output files are saved
run_directory = '/project/cigserver4/export1/c.rahul/projects'  
model_run_directory = '/project/cigserver4/export1/c.rahul/run_20241122_165730/'
# Load the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#Select gpu 1 (NVIDIA GeForce GTX 1080 Ti)
n_channels = 2
model = UNetRes(in_nc=n_channels, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', 
                downsample_mode="strideconv", upsample_mode="convtranspose").float()
model.to(device)
 
# Load model weights
model_path = os.path.join(model_run_directory, 'model-20241122_165730-epoch_490.pt')
checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Load test data or a sample signal from the dataset
trees = sio.loadmat("/project/cigserver4/export1/c.rahul/Indian_pines.mat")
trees = trees["indian_pines"]

train_data = trees[32:,32:,:].reshape(-1,1, trees.shape[-1]) #Use the rest of the square
dataMax = train_data.max()
test_data = trees[:32,:32,:].reshape(-1,1, trees.shape[-1]) #testing SNR on 100 values
print("Check1")
print(test_data.shape)
test_data = test_data/dataMax
print("Check2")
print(test_data.shape)

# Save the original dimensions of the test data
original_height = trees[:32, :32, :].shape[0]  # H
original_width = trees[:32, :32, :].shape[1]   # W
original_channels = trees[:32, :32, :].shape[2]  # C

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

compare_snr_2d = lambda x, xhat: 20 * np.log10(
    np.linalg.norm(x.flatten('F')) / np.linalg.norm(x.flatten('F') - xhat.flatten('F')))

class MatlabDataset(Dataset):
    def __init__(self, trees):
        self.data = trees

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get one sample of data
        sample = self.data[idx]
        
        # Convert to a torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        
        return sample
    
test_dataset = MatlabDataset(test_data)

batch_size = 128

test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

noise_level = 0.5

model.eval() 

all_gt = []    # To store ground truth signals
all_noisy = [] # To store noisy signals
all_denoised = []  # To store denoised signals

with torch.no_grad():  # Disable gradient computation
    total_snr = 0.0  # Initialize total SNR
    for batch_idx, cleanBatch in enumerate(test_dataloader):
        # Move the clean batch to the same device as the model
        cleanBatch = cleanBatch.to(device)

        # Generate noisy signals
        noisy_batch = cleanBatch + noise_level * torch.randn_like(cleanBatch)
        noisy_batch = torch.clamp(noisy_batch, 0, 1)  # Clamp noisy signals to [0, 1]

        # Create the sigma channel
        sigma_channel = noise_level * torch.ones_like(cleanBatch)

        # Concatenate noisy signal and sigma channel along the channel dimension
        model_input = torch.cat((noisy_batch, sigma_channel), dim=1)

        # Get the model's denoised output
        denoised_batch = model(model_input)
        
        #Calculate snr
        noisy_batch_snr_value = compare_snr(cleanBatch, noisy_batch)
        denoised_batch_snr_value = compare_snr(cleanBatch, denoised_batch)
        print("NOISY SNR VALUES INCOMING")
        print(noisy_batch_snr_value)
        print("DENOISED SNR VALUES INCOMING")
        print(denoised_batch_snr_value)
        
        image_file = os.path.join(run_directory, f'imageValues_{noise_level}.txt')
        with open(image_file, 'a') as f:
            f.write(f"Batch Idx: {batch_idx}\n Noisy SNR: {noisy_batch_snr_value}\n Denoised SNR: {denoised_batch_snr_value}\n")

        # Append results to the lists
        all_gt.append(cleanBatch.cpu().numpy())
        all_noisy.append(noisy_batch.cpu().numpy())
        all_denoised.append(denoised_batch.cpu().numpy())

        # Print batch information
        print(f"Processed batch {batch_idx + 1}/{len(test_dataloader)}")

# Convert results to numpy arrays for further analysis or visualization
all_gt = np.concatenate(all_gt, axis=0)  # Shape: [num_samples, 1, signal_length]
all_noisy = np.concatenate(all_noisy, axis=0)  # Same shape as all_gt
all_denoised = np.concatenate(all_denoised, axis=0)  # Same shape as all_gt

#Reshape back to initial shapes
all_gt_image = all_gt.squeeze(axis=1).reshape(original_height, original_width, original_channels)
all_noisy_image = all_noisy.squeeze(axis=1).reshape(original_height, original_width, original_channels)
all_denoised_image = all_denoised.squeeze(axis=1).reshape(original_height, original_width, original_channels)

print("Check3")
noisy_snr = compare_snr_2d(all_gt_image, all_noisy_image)
denoised_snr = compare_snr_2d(all_gt_image, all_denoised_image)
with open(image_file, 'a') as f:
    f.write(f"2D Noisy SNR: {noisy_snr}\n 2D Denoised SNR: {denoised_snr}\n")

print("Check4")
print(all_gt_image.shape)

print("plots starting")


plt.figure(figsize=(15, 5))

# Ground truth
plt.subplot(1, 3, 1)
plt.title("Ground Truth Image")
plt.imshow(all_gt_image[:, :, 10], cmap="gray")  
plt.colorbar()


# Noisy image
plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(all_noisy_image[:, :, 10], cmap="gray")  
plt.colorbar()


# Denoised image
plt.subplot(1, 3, 3)
plt.title("Denoised Image")
plt.imshow(all_denoised_image[:, :, 10], cmap="gray")  
plt.colorbar()

plt.tight_layout()
image_fig_path = os.path.join(run_directory, f'Image_{noise_level}.png')
plt.savefig(image_fig_path) #Set path and savefig

print("final plots done")

print(all_gt.shape, all_noisy.shape, all_denoised.shape)
print(all_gt_image.shape, all_noisy_image.shape, all_denoised_image.shape)

three_n_snr = compare_snr_2d(test_data,all_noisy_image[:, :, :])
three_dn_snr = compare_snr_2d(test_data,all_denoised_image[:, :, :])

print(f"BM3C 3D Noisy SNR (Total Dataset): {three_n_snr}")
print(f"BM3C 3D Denoised SNR (Total Dataset): {three_dn_snr}")

with open(image_file, 'a') as f:
    f.write(f"3D Noisy SNR: {three_n_snr}\n 2D Denoised SNR: {three_dn_snr}\n")
