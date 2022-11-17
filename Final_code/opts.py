from torch import nn
import os
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_label_path = './Daisee/Labels/TrainLabels.csv'
valid_label_path = './Daisee/Labels/ValidationLabels.csv'
test_label_path = './Daisee/Labels/TestLabels.csv'
criterion = nn.CrossEntropyLoss().to('cuda')
num_epochs = 30
arch = 'res50-112-5-512'
im_size = 224