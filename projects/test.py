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
test_data = trees[:32,:32,:]
print(test_data.shape)
test_data = trees[:32,:32,:].reshape(-1,1, trees.shape[-1])
train_data = train_data/dataMax
test_data = test_data/dataMax



print(test_data.shape)