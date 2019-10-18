# Author Lingge Li from XJTU(446049454@qq.com)
# functions

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from shufflenet_v2 import *
#from shufflenet_v2_to_gcn import *
#from ShuffleNetV2 import *
#from VGG16 import *

import argparse
import time
import os

gpu_id = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
#device_ids = range(torch.cuda.device_count())
#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training by shufflenet_v2')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',dest='weight_decay')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

EPOCH = 200
LR_DECAY_FACTOR = 0.1
min_loss = float('inf')



args = parser.parse_args()

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

trainData = torchvision.datasets.ImageFolder('/data2/ILSVRC2012/train', transform)
testData = torchvision.datasets.ImageFolder('/data2/ILSVRC2012/val', transform)
#print(trainData.class_to_idx)
#print(trainData.classes)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
print('traindata length: ', len(trainData))
print('testdata length: ', len(testData))


print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("hhh, let's use", torch.cuda.device_count(), "GPUs!")
    print('start creating shufflenet_v2 : ')
    shufflenet_v2 = ShuffleNet_v2()
    
    shufflenet_v2 = nn.DataParallel(shufflenet_v2).cuda()
    cudnn.benchmark = True
else:
    print('we only use one GPU', torch.cuda.get_device_name(0))
    print('start creating shufflenet_v2 : ')
    shufflenet_v2 = ShuffleNet_v2()
    shufflenet_v2 = shufflenet_v2.cuda()
    

#print(shufflenet_v2)

#shufflenet_v2 = ShuffleNet_v2()
#shufflenet_v2 = shufflenet_v2.cuda()

#for param in shufflenet_v2.named_parameters():
#print(param[0])

#init parameters: kaiming init
'''
for m in shufflenet_v2.modules():
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_()
'''
'''
print("Model's state_dict:")
for param_tensor in shufflenet_v2.state_dict():
    print(param_tensor, "\t", shufflenet_v2.state_dict()[param_tensor].size())
'''
# Loss and Optimizer
cost = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(shufflenet_v2.parameters(), lr=args.lr, momentum = args.momentum,  weight_decay = args.weight_decay)
'''
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
'''

print('start training....')
    

# Train the model
def train(epoch):
    #print('\nEpoch [{:03d}/{:03d}]'.format(epoch + 1, EPOCH))

    shufflenet_v2.train()
    train_loss = 0
    for i, (images, labels) in enumerate(trainLoader):
        #print(images.size())
        start_time = time.time()
        #for images, labels in trainLoader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        #print('labels: ', labels)

        #Forward + Backward + Optimize
        optimizer.zero_grad()
        #images = images.unsqueeze(0)

        outputs = shufflenet_v2(images)
        
        #print('Outside: input size', images.size(), 'output size', outputs.size())
        
        #_, predicted = torch.max(outputs.data, 1)
        #print('predicted: ', predicted)
        #print(outputs.size(), labels.size())

        loss = cost(outputs, labels)
        #print(loss)
        loss.backward()
        #print(loss.data, loss.data[0])

        optimizer.step()
        train_loss = train_loss + loss.item()
        end_time = time.time()
        #print ('Elapsed time: {}'.format(end_time - start_time))

        if (i+1) % 100 == 0 :
            print ('Epoch [{:03d}/{:03d}] Iter [{}/{}],  Loss [{:.4f}] / [{:.4f}]'.format(epoch + 1, EPOCH, i+1, len(trainData)//args.batch_size, loss.item(), train_loss / (i + 1)))

def validation():
    print('\nVal: ')
    shufflenet_v2.eval()
    val_loss = 0
    correct = 0.0
    total = 0
    for i, (images, labels) in enumerate(testLoader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = shufflenet_v2(images)
        loss = cost(outputs, labels)
        val_loss = val_loss + loss.item()

        _, predicted = torch.max(outputs.data, 1)

        #labels.size(0): bacth_size
        total = total + labels.size(0)
        #print(type(predicted), type(labels))
        correct = correct + (predicted == labels).float().sum().item()

    print('Accuracy of the network on the val images: {:.4f}'.format(100 * correct / total))
    print('Loss [{:.4f}] val_loss [{:.4f}]'.format(loss.item(), val_loss / (i + 1)))

    #save checkpoint
    global min_loss
    val_loss = val_loss / len(testLoader)
    if val_loss < min_loss:
        print('saving .....')
        state = {
            'epoch': epoch,
            'model_state_dict': shufflenet_v2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        if not os.path.isdir('/data1/llg/Documents/Shufflenet-v2-Pytorch/'):
            os.mkdir('/data1/llg/Documents/Shufflenet-v2-Pytorch/')
        torch.save(state, '/data1/llg/Documents/Shufflenet-v2-Pytorch/2_checkpoint.pth.tar')
        min_loss = val_loss


def learning_rate_adjust(optimizer, epoch):

    if epoch == 15 or epoch == 33 or epoch == 69 or epoch == 139:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * LR_DECAY_FACTOR

checkpoint = torch.load('/data1/llg/Documents/Shufflenet-v2-Pytorch/2_checkpoint.pth.tar')
for name in checkpoint['model_state_dict']:
    print(name)
    
shufflenet_v2.load_state_dict(checkpoint['model_state_dict'])
 
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
val_loss = checkpoint['val_loss']
for epoch in range(epoch, EPOCH):

    train(epoch)
    learning_rate_adjust(optimizer, epoch)
    if (epoch + 1) % 4 == 0:
        validation()


'''
shufflenet_v2 = ShuffleNetV2().cuda()
# Loss and Optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(shufflenet_v2.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(EPOCH):
 shufflenet_v2.train()
 for i, (images, labels) in enumerate(trainLoader):
  #for images, labels in trainLoader:
    images = Variable(images).cuda()
    labels = Variable(labels).cuda()

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    #images = images.unsqueeze(0)
    outputs = shufflenet_v2(images)
    #print(outputs.size(), labels.size())
    loss = cost(outputs, labels)
    loss.backward()
    optimizer.step()

    if (i+1) % 10 == 0 :
        print ('Epoch [%d/%d], Iter[%d/%d] Loss. %.4f' %
          (epoch+1, EPOCH, i+1, len(trainData)//BATCH_SIZE, loss.data[0]))
'''
