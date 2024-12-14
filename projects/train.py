import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import skimage as ski
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from model import UNetRes

save_directory = '/project/cigserver4/export1/c.rahul' 
    
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_directory = os.path.join(save_directory, f'run_{timestamp}')
if not os.path.exists(run_directory):
    os.makedirs(run_directory)  # Ensure the directory is created
# base path directory, add new directory for timestamp for each one
output_file = os.path.join(run_directory, 'output.txt')

print("Save Directory:", run_directory)
print("Timestamp:", timestamp)
    
writer = SummaryWriter(log_dir=os.path.join(run_directory, 'logs'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Select gpu 1 (NVIDIA GeForce GTX 1080 Ti)

trees = sio.loadmat("/project/cigserver4/export1/c.rahul/Indian_pines.mat")
trees = trees["indian_pines"]
testNumber = 32*32
#Import dataset

#plt.plot(trees[10,20,:]) 1D Signal (Groud truth)
#to train: nn.MSELoss, input: noised image (below is how), create new noise for every signal
#for 1 to i <- to numepochs
    #for data in dataset
    
train_data = trees[32:,32:,:].reshape(-1,1, trees.shape[-1]) #Use the rest of the square
dataMax = train_data.max()
test_data = trees[:32,:32,:].reshape(-1,1, trees.shape[-1])
train_data = train_data/dataMax
test_data = test_data/dataMax

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
    
train_dataset = MatlabDataset(train_data)
test_dataset = MatlabDataset(test_data)

batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True) 
 #Batch Sizes set to full dataset bc dataset is not very big
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False) 

#for batch_idx, (data) in enumerate(train_dataloader):
 #   print(f"Batch {batch_idx}:")
  #  print(f"Data: {data.shape}")  # Shape should be [145, 145, 220]

#Adding noise to the image    
#sigma = 0.05
for data in train_dataset:
    data = torch.clamp(data, 0, 1) ##Ensure data is between 0 and 1 before asserting
    assert torch.all(data >= 0) and torch.all(data <= 1)
    
for data in test_dataset:
    data = torch.clamp(data, 0, 1) ##Ensure data is between 0 and 1 before asserting
    assert torch.all(data >= 0) and torch.all(data <= 1)
    
# Function to compute SNR for a batch of data (using GPU)
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
        
#feed in batch, batch of data -> compute snr on single, don't flatten, for each point, loop through for batch-size
def compare_mse(img_test, img_true, size_average=True, return_as_tensor=True):
    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        # print(f"size average shape: {img_diff.shape}")
        mse = img_diff.mean()
    else:
        mse = img_diff.mean(-1).mean(-1).mean(-1)

    if return_as_tensor:
        return mse
    else:
        return mse.item()

def compare_psnr(img_test, img_true, size_average=True, max_value=1, return_as_tensor=True):
    psnr = 10 * torch.log10((max_value ** 2) / compare_mse(img_test, img_true, size_average))
    
    if return_as_tensor:
        return psnr
    else:
        # FIXME: check if one item 
        return psnr.item()

n_channels = 2 #added a channel for sigma channel 

# PyTorch models inherit from torch.nn.Module
model = UNetRes(in_nc=n_channels, 
            out_nc=1,  # Output remains as 1 channel since we only want the denoised image (before was n_channels)
            nc=[64, 128, 256, 512], 
            nb=4, 
            act_mode='R', 
            downsample_mode="strideconv", 
            upsample_mode="convtranspose"
           ).float()


model = model.to(device)
loss_fn = nn.MSELoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    model.train()
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
   
    # Need for 1 <- i to numepochs
    for batch_idx, (cleanBatch) in enumerate(train_dataloader):
        sigma = torch.rand(1).item() * 0.5 # Generates a random sigma value for each batch from 0 to 0.5
        print(f"Noise Level Train: {sigma}")
        ipt = cleanBatch+ sigma * torch.randn_like(cleanBatch)
        #Sigma set, about ipt, we wil sample sigma from a distribution from 0 - 0.5, unifrom distribution for sampling, torch.uniform() include batch_size in this 
        #On every epoch, batch, shoudld be sampling on a different sigma, sigma should be different across batches 
        # Every data instance is an input + label pair
        sigma_channel = sigma * torch.ones_like(cleanBatch)  #new sigma channel
        inputs = torch.cat((ipt, sigma_channel), dim=1).to(device)#to gpu
        #inputs = ipt.to(device)
        # Zero your gradients for every batch!
        cleanBatch = cleanBatch.to(device)  # Move cleanBatch to GPU
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, cleanBatch)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
    # Disable gradient computation and reduce memory consumption.
    
    #Return average loss per epoch
    avg_loss = running_loss / len(train_dataloader)
    print(f"Training Loss for epoch: {avg_loss}")
    return avg_loss

EPOCHS = 500

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval() #Switch to eval mode

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        total_snr = 0.0  # Initialize total SNR
        for batch_idx, (cleanBatch) in enumerate(test_dataloader):
            sigma = 0.25  # Random sigma for each test batch
            print(f"Noise Level Test: {sigma}")
            ipt = cleanBatch + sigma * torch.randn_like(cleanBatch) 
           # Every data instance is an input + label pair
            sigma_channel = sigma * torch.ones_like(cleanBatch) #new sigma channel
            inputs = torch.cat((ipt, sigma_channel), dim=1).to(device)#to gpu
            #inputs = ipt.to(device)
            # Zero your gradients for every batch!

            # Make predictions for this batch
            outputs = model(inputs)
            #Loop for batch_size 
            snr_value = compare_snr(cleanBatch, outputs) #Might have to switch this to numpy if errors
            #psnr_value = compare_psnr(outputs), (cleanBatch)
            total_snr += np.sum(snr_value)  # Accumulate SNR values
            
        #print(compare_snr(cleanBatch, outputs))
        
        filename = 'output.txt'
        avgSNR = total_snr/testNumber
        print(avgSNR )
        with open(output_file, 'a') as f:
            f.write(f"Epoch: {epoch + 1}, Average SNR: {avgSNR}, Average Loss: {avg_loss}, Noise Level: {sigma}\n")
 #change to avgSNR
        #f.write(f"Iteration { epoch + 1}: , SNR: {snr_value}, PSNR: {psnr_value}\n")
        #change to compute avg values, SNR sum/total numbers (testNumber variable) of element in test dataset, dont do this on train data
        #compute psnr and snr here 

    avg_vloss = running_vloss / (epoch + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    #track these values for every test in a file, save in cigserver4....

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()

    # Track best performance, and save the model's state
    #every epoch save model_latest
 
    #model_path = '/project/cigserver4/export1/c.rahul/projects/train.py'
    if (epoch % 10 == 0):
        model_save = os.path.join(run_directory, 'model-{}-epoch_{}.pt'.format(timestamp, epoch))
        #Explicit path for saving: projects/cigserver4/export1....   
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_save)
    
    model_latest = os.path.join(run_directory, 'model-{}-latest.pt'.format(timestamp))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(), #Saves the weights
        'optimizer_state_dict': optimizer.state_dict()
    }, model_save)
    
    model.train()  # Set model to training mode 

print("Model Latest Path:", model_latest)    
#saving the SNR data to a text file in the new directory
if not os.path.exists(model_latest):
    print("Warning: Model file was not saved as expected.")

#snr_value = compare_snr(test_data, ipt)
#psnr_value = compare_psnr(torch.tensor(ipt), torch.tensor(test_data))
#print(f"SNR: {snr_value}, PSNR: {psnr_value}")
#track these values for every test



saved_model = model
if not os.path.isfile(model_latest):
    print("Model file does not exist at:", model_latest)
else:
    saved_model.load_state_dict(torch.load(model_latest, weights_only=True))
    print("Model loaded successfully.")
#saved_model.load_state_dict(torch.load(model_latest, weights_only=True)) #changed to run directory




# Make this a seperate script
# Example for visualization
# Load ground truth image from a predefined dataset
#img_gt = cleanBatch  # You can replace with your own sample if needed
#img_size = 128  # Adjust size as needed  # Noise level
#img_gt = img_gt / 255.0      # Normalize to [0,1]
#img_gt = img_gt[:img_size, :img_size]  # Crop the image if needed
#img_gt = img_gt.astype('float32')
#img_gt = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(device)  # Add batch dimension and permute

# Add Gaussian noise
#ipt = cleanBatch + sigma * torch.randn_like(cleanBatch) 
    # Every data instance is an input + label pair
#inputs = ipt.to(device) #to gpu

# Add sigma map as an additional channel
#sigma_map = sigma * torch.ones(1, 1, img_size, img_size).float().to(device)
#ipt = torch.cat([ipt, sigma_map], dim=1)

# Pass through the model
#with torch.no_grad():
#    out = model(ipt)

# Convert to numpy for visualization
#img_gt_np = img_gt[0].cpu().permute(1, 2, 0).numpy()  # Ground truth
#ipt_np = ipt[0, :3].cpu().permute(1, 2, 0).numpy()    # Noisy input
#out_np = out[0].cpu().permute(1, 2, 0).numpy()        # Denoised output

# Plotting
#plt.figure(figsize=(15, 5))
#plt.subplot(1, 3, 1)
#plt.imshow(img_gt_np)
#plt.title("Ground Truth Signal")
#plt.axis('off')

#plt.subplot(1, 3, 2)
#plt.imshow(ipt_np)
#plt.title("Noisy Signal")
#plt.axis('off')

#plt.subplot(1, 3, 3)
#plt.imshow(out_np)
#plt.title("Denoised Signal")
#plt.axis('off')

#plt.show() 


#/project/cigserver4/export1/c.rahul/projects/train.py
