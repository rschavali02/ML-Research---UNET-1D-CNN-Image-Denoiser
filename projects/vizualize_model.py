import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNetRes  # Ensure the model definition is imported
from datetime import datetime
import scipy.io as sio


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

test_data = trees[1,1,:] #Set a particular signal to view 

print(test_data.shape)
train_data = trees[32:,32:,:].reshape(-1,1, trees.shape[-1]) #Use the rest of the square
dataMax = train_data.max()

test_data = test_data/dataMax
# Convert to tensor
img_gt = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, L]

print(f"img_gt: {img_gt.shape}")
# Add Gaussian noise
sigma = 0.3  # Define noise level
sigma_channel = sigma * torch.ones_like(img_gt)
ipt = torch.cat((img_gt + sigma * torch.randn_like(img_gt), sigma_channel), dim=1).to(device)
print(f"Noisy signal min: {ipt[0, 0].min()}, max: {ipt[0, 0].max()}")
print(ipt.shape)
ipt = torch.clamp(ipt, 0, 1)  # Clamp noisy signal values to [0, 1]
print(f"input shape: {ipt.shape}")
print("Check")
# Run the model
model.eval()
with torch.no_grad():
    out = model(ipt)
    
print(f"output shape: {out.shape}")

# Convert tensors to numpy for visualization
img_gt_np = img_gt[0].squeeze().cpu().numpy()  # Ground truth
print(img_gt_np.shape)
ipt_np = ipt[0,0].squeeze().cpu().numpy()        # Noisy input
out_np = out[0].squeeze().cpu().numpy()        # Denoised output

print("Check")

# Plot the signals
plt.figure(figsize=(15, 5))

# Ground Truth
plt.subplot(3, 1, 1)
plt.plot(img_gt_np, color='green', linewidth=1)
plt.title("Ground Truth Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

# Noisy Signal
plt.subplot(3, 1, 2)
plt.plot(ipt_np, color='blue', linewidth=1)
plt.title("Noisy Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

# Denoised Signal
plt.subplot(3, 1, 3)
plt.plot(out_np, color='red', linewidth=1)
plt.title("Denoised Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

# Save the plots
plot_path = os.path.join(run_directory, f'1D_signal_plot_{sigma}_.png')
plt.tight_layout()
plt.savefig(plot_path)
#plt.show()

print("Check")


# SNR and Loss Plotting
# Assuming SNR and loss values were saved in 'output.txt' in the format: "Iteration: X, Average SNR: Y"
snr_values = []
loss_values = []
epochs = []
#noise_levels = []

print("Check")

output_file = os.path.join(model_run_directory, 'output.txt')
with open(output_file, 'r') as f:
    for line in f:
        # Ensure the line is not empty and contains the expected format
        if not line.strip() or ':' not in line:
            print(f"Skipping malformed or empty line: {line}")
            continue
        
        try:
            # Split the line into parts by ", "
            parts = line.strip().split(', ')
            if len(parts) != 4:
                print(f"Skipping malformed line (not exactly 4 parts): {line}")
                continue
            
            # Parse each part
            epoch_num = float(parts[0].split(':')[1].strip())
            snr = float(parts[1].split(':')[1].strip())
            loss = float(parts[2].split(':')[1].strip())
            noise_level = float(parts[3].split(':')[1].strip())
            
            # Append to lists
            epochs.append(epoch_num)
            snr_values.append(snr)
            loss_values.append(loss)

        except (IndexError, ValueError) as e:
            print(f"Error parsing line: {line}\n{e}")

# Plot SNR over epochs
plt.figure()
plt.plot(epochs, snr_values, label='SNR')
plt.xlabel('Epoch')
plt.ylabel('SNR')
plt.title('SNR Over Epochs')
plt.legend()
plt.grid(True)
snr_fig_path = os.path.join(run_directory, f'SNR_plot_{sigma}.png')
plt.savefig(snr_fig_path) #Set path and savefig


print("Check")
# Plot Average Loss over epochs if available
if loss_values:
    plt.figure()
    plt.plot(epochs, loss_values, label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    loss_fig_path = os.path.join(run_directory, f'Loss_plot_{sigma}.png')
    plt.savefig(loss_fig_path) #Set path and savefig
