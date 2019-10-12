import torch
from torch.autograd import Variable
import time
import os
import sys
from Modular_code.opts import *
from  Modular_code.utils import *
import os
def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer, epoch_logger, batch_logger, batch_size , onecyc , writer , result_path):
    print('Training Epoch {}'.format(epoch))
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    start_time = time.time()
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if(inputs.size(0)<batch_size):
            continue
        if torch.cuda.is_available():
            targets = targets.cuda()
            inputs = inputs.cuda()
        with torch.no_grad():
            inputs = inputs.reshape(-1,3,im_size,im_size)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        lr,_ = onecyc.calc()
        update_lr(optimizer, lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del inputs,targets,outputs
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        writer.add_scalar('data/acc', accuracies.val, (epoch - 1) * len(data_loader) + (i + 1))
        writer.add_scalar('data/loss', losses.val, (epoch - 1) * len(data_loader) + (i + 1))        
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
    print('\nEpoch time {} mins'.format((end_time-start_time)/60))
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    writer.add_scalar('data/acc_epoch', accuracies.avg, epoch)
    writer.add_scalar('data/loss_epoch', losses.avg, epoch)        
        
    save_file_path = os.path.join(result_path,'save.pth')
    states = {
        'epoch': epoch + 1,
        'arch': arch,
        'state_dict': model.state_dict()
    }
    torch.save(states, save_file_path)