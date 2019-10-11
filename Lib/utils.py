import csv
import os
import cv2
import matplotlib.pyplot as plt
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')
        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()
    
def frame_extract(path):
    vidObj = cv2.VideoCapture(path) 
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image
            
def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom
def create_label(data_file,path):
    id_1 = data_file['ClipID']
    label = data_file['Engagement']
    id_val = []
    label_val = []
    for id in range(len(id_1)):
        path1 = os.path.join(path,id_1[id][:6])
        id_2 = id_1[id][:-4]
        video_path = os.path.join(path1,id_2)
        if(os.path.exists(video_path)):
            try:
                p = os.path.join(video_path,os.listdir(video_path)[0])
                if(os.path.exists(p)):
                    cap = cv2.VideoCapture(p)
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if(length>=300):
                        id_val.append(os.path.join(video_path,os.listdir(video_path)[0]))
                        label_val.append(label[id])
            except:
                pass
    return id_val,label_val