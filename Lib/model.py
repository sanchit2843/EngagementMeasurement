import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import copy
import os
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from ConvLSTM_pytorch.convlstm import ConvLSTM

size = 3
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = models.densenet121(pretrained = True).to('cuda')
        modules= list(model.children())[0]
        model = nn.Sequential(*modules)
        self.model = model
        self.c1 = ConvLSTM(  input_size=(size,size),
                             input_dim= 1024,
                             hidden_dim=[128, 512],
                             kernel_size=(5,5),
                             num_layers=2,
                             batch_first = True,
                             bias=True,
                             return_all_layers=False)
        self.fc1 = nn.Linear(512*size*size,1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000,256)
        self.fc3 = nn.Linear(256,4)
        self.dp = nn.Dropout(0.2)
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1,150,1024,size,size).detach()
        x = self.c1(x)
        x = x[0][0][:,-1,:,:,:]
        x = x.view(-1,512*size*size)
        x = self.dp(self.relu(self.fc1(x)))
        x = self.dp(self.relu(self.fc2(x)))
        x = self.dp(self.fc3(x))
        return x