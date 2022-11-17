from utils import *
from clr import OneCycle
from radam import RAdam
from opts import *
import pandas as pd
train_label = pd.read_csv(train_label_path)
valid_label = pd.read_csv(valid_label_path)
test_label = pd.read_csv(test_label_path)
train_id,train_classes = create_label(train_label,'./Daisee/Train',change =False) 
test_id,test_classes = create_label(test_label,'./Daisee/Test',change = False) 
valid_id,valid_classes = create_label(valid_label,'./Daisee/Validation',change = False) 
def extract_label(id,classes):
    id_new0 = []
    id_new1 = []
    for i,j in enumerate(classes):
        if(j == 0):
            id_new0.append(id[i])
        if(j == 1):
            id_new1.append(id[i])
    return id_new0,id_new1
train_id = train_id+test_id
train_classes = train_classes+test_classes
train_id0,train_id1 = extract_label(train_id,train_classes)
import numpy as np
train_id = train_id + 75*train_id0 + 10*train_id1
train_classes = train_classes + np.zeros(len(75*train_id0),dtype = int).tolist() + np.ones(len(10*train_id1) , dtype = int).tolist()
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import numpy as np
import cv2
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

# class video_dataset(Dataset):
#     def __init__(self,id,label,sequence_length = 60,transform = None,transform1 = None,revert = True):
#         self.id =id
#         self.engagement = label
#         self.transform = transform
#         self.transform1 = transform1
#         self.count = sequence_length
#         self.revert = revert
#         self.buffer = []
#         print('started buffer filling')
#         for i,path in enumerate(self.id):
#             frames = []
#             for frame in frame_extract(path):
#                 frames.append(frame)
#             self.buffer.append(frames)
#         print('buffer filled')
#     def __len__(self):
#         return len(self.id)
#     def __getitem__(self,idx):
#         frames = self.buffer[idx]
#         length = len(frames)
#         a = int(length/self.count)

#         hp = np.random.randint(0,2)
#         id1 = np.random.randint(0,a)
        
#         label = self.engagement[idx]
#         for i,frame in enumerate(frames):
#             if(i % a == id1):
#                 if(self.transform1):
#                     frames[i] = frame
#                 else:
#                     frames[i] = self.transform(frame)
#         if(self.transform1):
#             frames = augment_and_mix(frames,transform = self.transform)
#         if(self.revert):
#           hp = np.random.randint(0,2)
#           if(hp == 1):
#             frames = frames[::-1]
#         frames = torch.stack(frames)
#         frames = frames[:self.count]
#         return frames,label

class video_dataset(Dataset):
    def __init__(self,id,label,sequence_length = 60,transform = None,transform1 = None,revert = True):
        self.id =id
        self.engagement = label
        self.transform = transform
        self.transform1 = transform1
        self.count = sequence_length
        self.revert = revert
    def __len__(self):
        return len(self.id)
    def __getitem__(self,idx):
        video_path = self.id[idx]
        frames = []
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        a = int(length/self.count)
        hp = np.random.randint(0,2)
        id1 = np.random.randint(0,a)
        label = self.engagement[idx]
        frames = []
        for i,frame in enumerate(frame_extract(video_path)):
            if(i % a == id1):
                if(label<2):
                   if(hp == 1):
                       frame = cv2.flip(frame,1)
                if(self.transform1):
                    frames.append(frame)
                else:
                    frames.append(self.transform(frame))
        if(self.transform1):
            frames = augment_and_mix(frames,transform = self.transform)
        if(self.revert):
          hp = np.random.randint(0,2)
          if(hp == 1):
            frames = frames[::-1]
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames,label

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import cv2
from albumentations import (
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,Resize,ImageCompression,MultiplicativeNoise,ChannelDropout,IAASuperpixels,GaussianBlur
)

augs = [
            RandomBrightnessContrast(brightness_limit=0.05,contrast_limit=0.05,p=1),
            Blur(blur_limit=2,p=1),
#            OpticalDistortion(p=1),
            ImageCompression(p=1),
            MultiplicativeNoise(p=1),
#              IAASharpen(alpha=(0, 0.2) , p = 1),
#             IAAEmboss(alpha=(0, 0.3) , p = 1),
            MotionBlur(blur_limit = 3,p=1),
#             MedianBlur(blur_limit=3,p=1)
        ]

def apply_op(frames, op):
    a = op(image = frames)['image']
    return a
import random
def augment_and_mix(image, augs = augs , width = 1, depth = 2, alpha = 1.,transform = None):
    frames = []
    ops = []
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    for i in range(width):
        op = []
        ag = augs.copy()
        for j in range(depth):
            a = np.random.choice(ag)
            ag.remove(a)
            op.append(a)
        ops.append(Compose(op))
    for frame in image:
        mix = torch.zeros((3,im_size,im_size))
        for i in range(width):
            image_aug = frame.copy()
            op = ops[i]
            image_aug = transform(apply_op(image_aug, op))
            mix += ws[i] * image_aug
        frames.append((1 - m) * transform(frame) + m * mix)
    return frames

im_size = 112
train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
train_data = video_dataset(train_id,train_classes,sequence_length = 100,transform = train_transforms,transform1 = True,revert = True)
val_data = video_dataset(valid_id,valid_classes,sequence_length = 100,transform = train_transforms)

import cv2
import random
image,label = train_data[50]
im_plot(image[0,:,:,:])

from torchvision import models
import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,lamb_m = 0.5):
        super(Model, self).__init__()
        model = models.resnet18(pretrained = True)
        model1 = nn.Sequential(*list(model.children())[:-2])
        self.model1 = model1
        self.lstm = nn.LSTM(512, 512, 3, batch_first=True, bidirectional = False)
        self.f1 = nn.Linear(512,4)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.dp1 = nn.Dropout(0.4)
        self.sig = nn.Sigmoid()
        self.pool1 = nn.AdaptiveAvgPool2d(output_size = 1)
        self.attention1 = nn.Linear(512,1)    
        self.lamb_m = lamb_m
    def forward(self, input):
        seq_size = input.shape[1]
        input = input.view(-1,3,im_size,im_size)
        x = self.model1(input)
        x_pool = self.relu(self.pool1(x)).reshape(-1,seq_size,512)
        x_lstm,_ = self.lstm(x_pool)
        x2 = []
        for i in range(x_pool.shape[0]):
            alpha = self.sig(self.attention1(self.dp1(x_pool[i])))
            x2.append(torch.div(torch.sum(x_pool[i]*alpha.reshape(-1,1),dim = 0),torch.sum(alpha,dim = 0)))
        lstm = torch.mean(x_lstm,dim = 1)
        attention = torch.stack(x2)
        x = self.lamb_m*attention+(1-self.lamb_m)*lstm
        x = self.dp(self.f1(x))
        return x
model = Model().to('cuda')
#!rm -r results1
arch = 'resnet18-lstm+attention-0.5-(75)-10-4-2-1-0.4'#.format(split_no)
os.makedirs('results2',exist_ok = True)
result_path = os.path.join('./results2',arch)
from tensorboardX import SummaryWriter
writer = SummaryWriter()
os.makedirs(result_path,exist_ok = True)
train_logger = Logger(os.path.join(result_path, 'train1{}.log'.format(arch)),['epoch', 'loss', 'acc', 'lr'])
train_batch_logger = Logger(
            os.path.join(result_path, 'train_batch1{}.log'.format(arch)),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])    
val_logger = Logger(
            os.path.join(result_path, 'val1{}.log'.format(arch)), ['epoch', 'loss', 'acc'])
import torch
from torch import nn
import numpy as np
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
class sanchitLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Tanh()
        self.soft = nn.Softmax()
    def forward(self, pred, true):
        clases = torch.arange(0,pred.shape[1],1).reshape(1,-1).repeat(pred.shape[0],1)
        if(torch.cuda.is_available()):
            clases = clases.cuda()
        clases = (abs(clases-true.reshape(-1,1))).type(torch.cuda.DoubleTensor)
        dis = clases
        for d in dis:
            if(d[1] == 0):
                d[2] += 1
#                d[3] += 1
                continue
            if(d[2] == 0):
                d[1] += 1
 #               d[0] += 1
                continue
#        dis = dis/torch.sum(dis)
        if(torch.cuda.is_available()):
            pred = self.soft(pred).type(torch.cuda.DoubleTensor)
        loss = torch.sum(dis * pred,axis = 1).type(torch.cuda.FloatTensor)
        #sorted, indices = torch.sort(loss)
        return torch.mean(loss,axis = 0)
#        return torch.mean(sorted[int(0.2*len(sorted)):],axis = 0)
s = sanchitLoss()
s(torch.from_numpy(np.asarray([-1,  0.8490,  0.0000,  0.0000])).unsqueeze(dim = 0).type(torch.cuda.FloatTensor),torch.from_numpy(np.asarray([1])).type(torch.cuda.LongTensor))
## MSE loss pipeline
import torch
from torch.autograd import Variable
import time
import os
import sys
import os
#sa = SoftArgmax1D()
def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer, epoch_logger, batch_logger, batch_size , onecyc , writer , result_path , lamb_l = 0.5):
    print('Training Epoch {}'.format(epoch))
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    start_time = time.time()
    end_time = time.time()
    t = []
    los = nn.CrossEntropyLoss().cuda()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if torch.cuda.is_available():
            targets = targets.type(torch.cuda.LongTensor)
            inputs = inputs.cuda()
        model.zero_grad()
        outputs = model(inputs)    
        loss1 = criterion(outputs, targets.type(torch.cuda.LongTensor))
        loss  = los(outputs,targets.type(torch.cuda.LongTensor))
        loss = loss*lamb_l +  (1-lamb_l)*loss1
        loss.backward()
        optimizer.step()
        lr,mom = onecyc.calc()
        update_lr(optimizer, lr)
        update_mom(optimizer,mom)
        
        batch_time.update(time.time() - end_time)
        acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        wandb.log({'Batch accuracy': accuracies.val,'batch Loss':losses.val,'lr': optimizer.param_groups[0]['lr']})
        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d / %d] [Time %.2f %.2f] [Data %.2f %.2f] [Loss: %f, Acc: %.2f%%]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(data_loader),
                    batch_time.val,
                    batch_time.avg,
                    data_time.val,
                    data_time.avg,
                    losses.avg,
                    accuracies.avg))
        end_time = time.time()
        
    print('\nEpoch time {} mins'.format((end_time-start_time)/60))
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
        
    save_file_path = os.path.join(result_path,'save.pth')
    states = {
        'state_dict': model.state_dict(),
        'optim_dict':optimizer.state_dict()
    }
    wandb.log({'Epoch accuracy': accuracies.avg,'Epoch Loss':losses.avg})
    torch.save(states, save_file_path)
    return t


def test(epoch,model, data_loader ,criterion, batch_size, result_path,best_acc = 0 , logger = None ):
    print('Testing')
    if(epoch == 1):
        best_acc = 0
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    los = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                targets = targets.cuda().type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
            outputs = model(inputs)
            loss = torch.mean(los(outputs, targets.type(torch.cuda.LongTensor)))
            acc = calculate_accuracy(outputs,targets.type(torch.cuda.LongTensor))
            _,p = torch.max(outputs,1)
            true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            sys.stdout.write(
                    "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%]"
                    % (
                        i,
                        len(data_loader),
                        losses.avg,
                        accuracies.avg
                        )
                    )
        if(logger):
            logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        print('\nAccuracy {}'.format(accuracies.avg))
        if(accuracies.avg>best_acc):
            best_acc = accuracies.avg
            result_path = os.path.join(result_path,'best.pth')
            state = {
            'acc':best_acc,
             'state':model.state_dict()
            }
            torch.save(state,result_path)
    wandb.log({'Validation accuracy': accuracies.avg,'Validation Loss':losses.avg})
    wandb.log({'true':np.asarray(true),'pred':np.asarray(pred)})
    return true,pred,best_acc

import wandb
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from radam import RAdam
params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 0,
        'attention_lambda': 0,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')

lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))


params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 0,
        'attention_lambda': 0.5,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))

params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 0,
        'attention_lambda': 1,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))

params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 0.5,
        'attention_lambda': 0,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))


params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 0.5,
        'attention_lambda': 0.5,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))


params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 0.5,
        'attention_lambda': 1,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))


params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 1,
        'attention_lambda': 0,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))


params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 1,
        'attention_lambda': 0.5,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))


params = {
        'batch_size': 16,
        'pretrain': True,
        'optimizer' : 'sgd',
        'model' : 'resnet18',
        'loss_lambda': 1,
        'attention_lambda': 1,
        'augmix': False,
        'augmentation_out': 'mix',
        'oversample_0': 75,
        'oversample_1': 0,
        'one_cycle': True,
        'weight_decay': 1e-3,
        'num_lstm_layer': 3,
        'revert': False    
        }

os.system('wandb login {}'.format('2d5e5aa07e2a9cd4f84004f838566b5eca9f5856'))
wandb.init(name = 'new_l{}_a{}'.format(params['loss_lambda'],params['attention_lambda']),project="daisee",config = params)

model = Model(lamb_m = params['attention_lambda']).to('cuda')


lr = 1e-3
batch_size = 8
train_loader = DataLoader(train_data,batch_size = batch_size , num_workers = 8,shuffle = True,pin_memory = True,drop_last = True)
val_loader = DataLoader(val_data,batch_size = 4, num_workers = 0,pin_memory = True,drop_last = False)
criterion = sanchitLoss().cuda()
best_acc = 0
num_epochs = 30
optimizer = torch.optim.SGD(model.parameters(), lr= lr,weight_decay = 1e-3,momentum = 0.9)
onecyc = OneCycle(len(train_loader)*num_epochs, lr)
wandb.watch(model, log="all")
for epoch in range(1,num_epochs+1):
    train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer,train_logger,train_batch_logger,batch_size,onecyc,writer,result_path,lamb_l = params['loss_lambda'])
    true,pred,best_acc= test(epoch,model,val_loader,criterion,batch_size,result_path,best_acc)
    cm1 = confusion_matrix(true,pred)
    print('modified_accuracy:',(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]+cm1[2,2]+cm1[2,3]+cm1[3,2]+cm1[3,3])/np.sum(cm1))
    print('Comb\n',cm1)
    print(classification_report_imbalanced(true, pred))