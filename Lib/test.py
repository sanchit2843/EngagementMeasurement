import torch
from torch.autograd import Variable
import time
import os
import sys
from  Modular_code.utils import *
from Modular_code.opts import *

def test(model, data_loader ,criterion, batch_size, logger = None):
    print('Testing')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
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
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
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
    return losses.avg,accuracies.avg