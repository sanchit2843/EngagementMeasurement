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
size = 4
from torchvision import models

from efficientnet_pytorch import EfficientNet
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return SwishImplementation.apply(x)
from ConvLSTM_pytorch.convlstm import ConvLSTM
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(2048,3)
        model.load_state_dict(torch.load('./WACV/best.pth'))
        model1 = nn.Sequential(*list(model.children())[:-2])
        for param in model1.parameters():
            param.requires_grad = False
        self.model1 = model1
        c1 = ConvLSTM(  input_size=(size,size),
                             input_dim= 2048,
                             hidden_dim=[128,256],
                             kernel_size=(3,3),
                             num_layers=2,
                             batch_first = True,
                             bias= True,
                             return_all_layers = False)
        self.c1 = c1
        self.attention = nn.Linear(256*size*size,1)
        self.f1 = nn.Linear(256*size*size+2048,2048)
        self.relu = nn.ELU()
        self.fc2 = nn.Linear(2048,512)
#        self.fc3 = nn.Linear(2048,512)
        self.fc4 = nn.Linear(512,4)
        self.dp = nn.Dropout(0.4)
        self.sig = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.attention1 = nn.Linear(2048,1)
        self.beta_attention = nn.Linear(4096,1)
    def forward(self, input):
        input = input.view(-1,3,112,112)
        x = self.model1(input)
        x_pool = self.pool(x).reshape(-1,100,2048)
        x = x.view(-1,100,2048,size,size)
        x = self.c1(x)
        x = x[0][0].view(-1,256*size*size)
        x = x.view(-1,100,256*size*size)
        #x1 = []
        x2 = []
        v1 = []
        for i in range(x_pool.shape[0]):            
            '''alpha = self.sig(self.attention1(self.dp(x_pool[i])))
            x1 = torch.sum(x_pool[i]*alpha,dim = 0)
            a1 = torch.sum(alpha , dim = 0)
            x1 = torch.div(x1,a1)
            x11 = x1.unsqueeze(dim = 0).repeat(100,1)
            x3 = torch.cat([x_pool[i],x11],dim = 1)
            betas = self.sig(self.beta_attention(self.dp(x3)))
            v1.append(x3*alpha*betas)'''
            alpha = self.sig(self.attention1(self.dp(x_pool[i])))
            #x1.append(torch.div(torch.sum(x[i]*att[i].reshape(-1,1),dim = 0),torch.sum(att[i])))
            x2.append(torch.div(torch.sum(x_pool[i]*alpha.reshape(-1,1),dim = 0),torch.sum(alpha,dim = 0)))
        x2 = torch.stack(x2)
#        v1 = torch.stack(v1)
#        v1 = torch.sum(v1,dim = 1)
#        v1 = torch.div(v1,torch.sum(alpha*betas,dim = 0))
        x1 = torch.mean(x,dim = 1)
        x = torch.cat([x1,x2],dim = 1)
#        x = self.dp(self.relu(self.fc3(x)))
        #x = torch.div(x,torch.sum(att,dim = 1))
        x = self.dp(self.relu(self.f1(x)))
        x = self.dp(self.relu(self.fc2(x)))
        x = self.dp(self.fc4(x))
        return x
model = Model().to('cuda')
model = nn.DataParallel(model)
