import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torch.utils.data as data
import torchvision.transforms as transforms

from dataset import Filtered, none_collate, targetToVariable
from tinyYOLO import TinyYOLO, TinyDarkNet
from loss import Loss

import shutil

def epoch_train(model, creterion, optimizer, dataloader, args):
    model.train()
    loss_per_epoch = 0
    for i, (x, y) in enumerate(dataloader, 1):
        # print('0', x.size(), len(y))
        optimizer.zero_grad()
        x = Variable(x).cuda()
        y= targetToVariable(y, use_cuda=True)
        y_pred = model(x)
        loss_e = creterion(y_pred, y)

        loss_e.backward()
        optimizer.step()
        loss_per_epoch += loss_e

    return loss_per_epoch/i

def epoch_validate(model, creterion, dataloader, args):
    model.eval()
    loss_per_epoch = 0
    for i, (x, y) in enumerate(dataloader, 1):
        x = Variable(x).cuda()
        y = targetToVariable(y, use_cuda=True)
        y_pred = model(x)
        loss = creterion(y_pred, y)
        loss_per_epoch += loss
        break;
    return loss_per_epoch/i

def train(args):
    epochs = args['epoch']
    newTrain = args['newTrain']

    print('Loading TinyDarknet...')
    pretrained = TinyDarkNet()

    if not newTrain:
        checkpoint = torch.load('best_checkpoint.pth.tar')
        pretrained.load_state_dict(checkpoint['state_dict'])
    model = TinyYOLO()
    model.darknet = pretrained.net

    model = nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    creterion = Loss(args)
    optimizer = optim.Adam(model.parameters(), lr=args['lr']/args['batch_size'])

    transformer = transforms.Compose([transforms.ToTensor()])
    kwargs = {'num_workers':1, 'pin_memory':True}

    print('Loading train data...')
    trainloader = data.DataLoader(
        Filtered(args['train'],
                      train=True,
                      transform=transformer,
                      use_cuda=True),
        batch_size=args['batch_size'],
        collate_fn=none_collate,
        shuffle=True,
        **kwargs
    )
    print('train data: ', len(trainloader))
    print('Loading VALID data...')
    testloader = data.DataLoader(
        Filtered(args['test'],
                  train=False,
                  transform=transformer,
                  use_cuda=True),
        batch_size=args['batch_size'],
        collate_fn=none_collate,
        shuffle=False,
        **kwargs
    )
    print('train data: ', len(testloader))
    print('Start learning...')

    best_loss = 1000000
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, args['lr'], args['batch_size'],
                            args['scales'], args['steps'])

        loss = epoch_train(model, creterion, optimizer, trainloader, args)
        val_loss = epoch_validate(model, creterion, testloader, args)
        print('Epoch {0}: Loss={1:.3f}, Validation={2:.3f}'.format(float(epoch), float(loss), float(val_loss)))

        if float(best_loss) > float(val_loss):
            print('Update model weight file')
            best_loss = val_loss
            save_checkpoint({'epoch': epoch+1,
                             'state_dict': model.module.state_dict(),
                             'optimizer': optimizer.state_dict()}, True)
        save_checkpoint({'epoch': epoch+1,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()}, False)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_checkpoint.pth.tar')

def adjust_learning_rate(optimizer, epoch, learning_rate, batch_size, scales, steps):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if epoch >= steps[i]:
            lr = lr * scale
            if epoch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

if __name__ == '__main__':
    args = {'epoch': 300,
            'lr':0.001,
            'steps': [-1, 1, 200, 250],
            'scales': [.1, 10, .1, .1],
            'use_cuda': True,
            'class_scale': 1.,
            'object_scale':5.,
            'noobject_scale':1.,
            'coord_scale':1.,
            'batch_size': 64,
            'train': 'Dataset/small_train.txt',
            'test': 'Dataset/test_data.txt',
            'newTrain': True,
            }
    print('train data is ', args['train'])
    train(args)
