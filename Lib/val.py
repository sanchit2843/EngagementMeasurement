import torch
from torch.autograd import Variable
import time
import os
import sys
from  Modular_code.utils import *
from Modular_code.opts import *

def val_epoch(epoch, num_epochs,data_loader, model, criterion,  logger , batch_size , writer):
    print('\nValidation Epoch {}'.format(epoch))
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        if(inputs.size(0)<batch_size):
            continue
        data_time.update(time.time() - end_time)
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
        batch_time.update(time.time() - end_time)
        end_time = time.time()
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
                    accuracies.avg
                    )
                )
    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
    writer.add_scalar('data/valacc',losses.avg , epoch)
    writer.add_scalar('data/valloss',accuracies.avg ,epoch)
    return losses.avg