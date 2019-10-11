from torch import nn
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]
train_label_path = './DAiSEE/Labels/TrainLabels.csv'
valid_label_path = './DAiSEE/Labels/ValidationLabels.csv'
test_label_path = './DAiSEE/Labels/TestLabels.csv'
im_size = 112
criterion = nn.CrossEntropyLoss().to('cuda')
num_epoch = 20