#!git clone https://github.com/ndrplz/ConvLSTM_pytorch
#!git clone https://github.com/ndrplz/ConvLSTM_pytorch
from ConvLSTM_pytorch.convlstm import ConvLSTM
size = 3
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model1 = EfficientNet.from_pretrained('efficientnet-b3').cuda() 
        for param in model1.parameters():
            param.requires_grad = False
        self.model1 = model1
        
        c1 = ConvLSTM(  input_size=(size,size),
                             input_dim=1536,
                             hidden_dim=[128,512],
                             kernel_size=(5,5),
                             num_layers=2,
                             batch_first = True,
                             bias=True,
                             return_all_layers = False)
        for param in c1.parameters():
            param.requires_grad = False
        self.c1 = c1
        self.f1 = nn.Linear(512*size*size*5,2048)
        self.relu = Swish()
        #fast = resnet50().cuda()
        #for param in fast.parameters():
        #    param.requires_grad = False
        #self.fast = fast
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512,4)
        
        #self.f3 = nn.Linear(256 + 256,4)
        self.dp = nn.Dropout(0.2)
    def forward(self, input):
        x = self.model1.extract_features(input)
        #y = self.fast(input)
        x = x.view(-1,150,1536,size,size).detach()
        x = self.c1(x)
        x1 = []
        with torch.no_grad():
            for i in range(5):
                x1.append(torch.mean(x[0][0][:,i*30:i*30+30,:,:,:],dim = 1))
            x1 = torch.stack(x1).cuda()
        #x = torch.mean(x[0][0],dim = 1)
        x = x1.view(-1,512*size*size*5).detach()
        x = self.dp(self.relu(self.f1(x)))
        x = self.dp(self.relu(self.fc2(x)))
        x = self.dp(self.fc3(x))
        return x
model = Model().to('cuda')
model = nn.DataParallel(model)
pretrained_dict = torch.load('./results/effnetb4-fast/save.pth')['state_dict']
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
#model.load_state_dict(torch.load('./results/effnetb4-fast/save.pth')['state_dict'])
from ConvLSTM_pytorch.convlstm import ConvLSTM
size = 3
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = models.densenet121(pretrained = True).to('cuda')
        modules= list(model.children())[0]
        model = nn.Sequential(*modules)
        for param in model.parameters():
            param.requires_grad = False
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
