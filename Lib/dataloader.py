import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import cv2
import pandas as pd
from constants import *
train_label = pd.read_csv(trainlabel_path)
valid_label = pd.read_csv(vallabel_path)

class video_dataset(Dataset):
    def __init__(self,frame_dir,train_csv,sequencelength = 60,transform = None):
        self.folder = os.listdir(frame_dir)
        self.id = train_csv['ClipID']
        self.engagement = train_csv['Engagement']
        self.frame_dir = frame_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.skip_length = int(300/sequence_length)
    def __len__(self):
        return len(self.id)
    def __getitem__(self,idx):
        id_1 = self.id[idx][:6]
        path1 = os.path.join(self.frame_dir,id_1)
        id_2 = self.id[idx][:-4]
        path2 = os.path.join(path1,id_2)
        seq_image = list()
        i = 0
        while i<300:
            path3 = os.path.join(path2,str(i)+'.jpg')
            image = cv2.imread(path3)
            if(self.transform):
                image = self.transform(image)
            seq_image.append(image)
            i = i+self.skip_length
        seq_image = torch.stack(seq_image)
        label = self.engagement[idx]
        seq_image = seq_image.reshape(3,self.sequence_length,im_size,im_size)
        return seq_image,label

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.RandomRotation(degrees=10),
                                        transforms.ToTensor()])
test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor()])
train_data = video_dataset(train_path,train_label,transform = train_transforms)
val_data = video_dataset(val_path,valid_label,transform = test_transforms)
train_loader = DataLoader(train_data,batch_size = 4,num_workers = 4 ,shuffle = True)
valid_loader = DataLoader(val_data,batch_size = 4,num_workers = 4 ,shuffle = True)
dataloaders = {'train':train_loader,'val':valid_loader}