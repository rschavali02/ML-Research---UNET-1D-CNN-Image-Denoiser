import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

run_directory = '/project/cigserver4/export1/c.rahul/projects'
trees = sio.loadmat("/project/cigserver4/export1/c.rahul/Indian_pines.mat")
trees = trees["indian_pines"]

class GaussianBlur1D(nn.Module):
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        """
        A module for applying Gaussian blur to 1D signals.

        Args:
            kernel_size (int): Size of the Gaussian kernel.
            sigma (float): Standard deviation of the Gaussian kernel.
        """
        super(GaussianBlur1D, self).__init__()
        
        # Create the Gaussian kernel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2  # To maintain the same size after convolution
        
        # Generate the Gaussian weights
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma)
        
        # Convert kernel to Conv1d weights (1, 1, kernel_size)
        self.kernel = torch.tensor(self.kernel, dtype=torch.float32).view(1, 1, -1)
        
        # Define Conv1d layer
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        
        # Assign Gaussian kernel to Conv1d weights and freeze it
        self.conv1d.weight.data = self.kernel
        self.conv1d.weight.requires_grad = False

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        """
        Generate a Gaussian kernel.

        Args:
            kernel_size (int): Size of the kernel.
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            np.ndarray: Normalized Gaussian kernel.
        """
        # Generate a symmetric range around 0
        x = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1, 1)
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()  # Normalize
        return kernel

    def forward(self, x):
        """
        Apply Gaussian blur.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, signal_length).

        Returns:
            torch.Tensor: Blurred signals.
        """
        return self.conv1d(x)

# Example Usage
if __name__ == "__main__":
    # Create a dummy signal (batch_size=1, channels=1, signal_length=100)
    signal = torch.tensor(trees[1, 1, :], dtype=torch.float32).unsqueeze(0).unsqueeze(0) #Numpy array to torch tensor
    
    # Initialize Gaussian blur model
    blur_model = GaussianBlur1D(kernel_size=5, sigma=1.0)
    
    # Apply blur
    blurred_signal = blur_model(signal)
    
    print("Original Signal:", signal)
    print("Blurred Signal:", blurred_signal)
    
    # Plot the signals

# Convert tensors to NumPy arrays
signal_np = signal[0, 0].detach().cpu().numpy()  # Remove batch and channel dimensions
blurred_signal_np = blurred_signal[0, 0].detach().cpu().numpy()

# Plot the signals
plt.figure(figsize=(15, 5))

# Original Signal
plt.subplot(2, 1, 1)
plt.plot(signal_np, color='green', linewidth=1)
plt.title("Original Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

# Blurred Signal
plt.subplot(2, 1, 2)
plt.plot(blurred_signal_np, color='blue', linewidth=1)
plt.title("Blurred Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

# Save the plots
plot_path = os.path.join(run_directory, '1D_Blur_Plot.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()  # Display the plots