import torch
from torch.autograd import Variable
import time
import os
import sys
from  Modular_code.utils import *
from Modular_code.opts import *

def test(epoch,model, data_loader ,criterion, batch_size, result_path,best_acc = 0 , logger = None ):
    print('Testing')
    if(epoch == 1):
        best_acc = 0
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    for i, (inputs, targets) in enumerate(data_loader):
        if(inputs.size(0)<batch_size):
            continue
        if torch.cuda.is_available():
            targets = targets.cuda()
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        with torch.no_grad():
            inputs = inputs.reshape(-1,3,im_size,im_size)
        targets = Variable(targets)
        outputs = model(inputs)
        _,p = torch.max(outputs,1)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        true += targets.detach().cpu().numpy().reshape(len(targets)).tolist()
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
    return true,pred,best_acc