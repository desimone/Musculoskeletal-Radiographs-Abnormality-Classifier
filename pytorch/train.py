from __future__ import absolute_import, division, print_function

import argparse
import random
import shutil
from os import getcwd
from os.path import exists, isdir, isfile, join

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from dataloader import MuraDataset

print("torch : {}".format(torch.__version__))
print("torch vision : {}".format(torchvision.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data_dir', default='MURA-v1.0', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', default='densenet121', choices=model_names, help='nn architecture')
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=512, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=.1, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--fullretrain', dest='fullretrain', action='store_true', help='retrain all layers of the model')
parser.add_argument('--seed', default=1337, type=int, help='random seed')

best_val_loss = 0

tb_writer = SummaryWriter()


def main():
    global args, best_val_loss
    args = parser.parse_args()
    print("=> setting random seed to '{}'".format(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        if 'resnet' in args.arch:
            # for param in model.layer4.parameters():
            model.fc = nn.Linear(2048, args.classes)

        if 'dense' in args.arch:
            if '121' in args.arch:
                # (classifier): Linear(in_features=1024)
                model.classifier = nn.Linear(1024, args.classes)
            elif '169' in args.arch:
                # (classifier): Linear(in_features=1664)
                model.classifier = nn.Linear(1664, args.classes)
            else:
                return

    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model).cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> found checkpoint")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])

            args.epochs = args.epochs + args.start_epoch
            print("=> loading checkpoint '{}' with acc of '{}'".format(
                args.resume,
                checkpoint['best_val_loss'], ))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_dir = join(getcwd(), args.data_dir)
    train_dir = join(data_dir, 'train')
    train_csv = join(data_dir, 'train.csv')
    val_dir = join(data_dir, 'valid')
    val_csv = join(data_dir, 'valid.csv')
    test_dir = join(data_dir, 'test')
    assert isdir(data_dir) and isdir(train_dir) and isdir(val_dir) and isdir(test_dir)
    assert exists(train_csv) and isfile(train_csv) and exists(val_csv) and isfile(val_csv)

    # Before feeding images into the network, we normalize each image to have
    # the same mean and standard deviation of images in the ImageNet training set.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # We then scale the variable-sized images to 224 × 224.
    # We augment by applying random lateral inversions and rotations.
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = MuraDataset(train_csv, transform=train_transforms)
    weights = train_data.balanced_weights
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # num_of_sample = 37110
    # weights = 1 / torch.DoubleTensor([24121, 1300])
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_of_sample)
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True)
    val_loader = data.DataLoader(
        MuraDataset(val_csv,
                    transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    # We use an initial learning rate of 0.0001 that is decayed by a factor of
    # 10 each time the validation loss plateaus after an epoch, and pick the
    # model with the lowest validation loss
    if args.fullretrain:
        print("=> optimizing all layers")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        print("=> optimizing fc/classifier layers")
        optimizer = optim.Adam(model.module.fc.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, verbose=True)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_loss = validate(val_loader, model, criterion, epoch)
        scheduler.step(val_loss)
        # remember best Accuracy and save checkpoint
        is_best = val_loss > best_val_loss
        best_val_loss = max(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc = AverageMeter()

    # ensure model is in train mode
    model.train()
    pbar = tqdm(train_loader)
    for i, (images, target, meta) in enumerate(pbar):
        target = target.cuda(async=True)
        image_var = Variable(images)
        label_var = Variable(target)

        # pass this batch through our model and get y_pred
        y_pred = model(image_var)

        # update loss metric
        loss = criterion(y_pred, label_var)
        losses.update(loss.data[0], images.size(0))

        # update accuracy metric
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        acc.update(prec1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("EPOCH[{0}][{1}/{2}]".format(epoch, i, len(train_loader)))
        pbar.set_postfix(
            acc="{acc.val:.4f} ({acc.avg:.4f})".format(acc=acc),
            loss="{loss.val:.4f} ({loss.avg:.4f})".format(loss=losses))

    tb_writer.add_scalar('train/loss', losses.avg, epoch)
    tb_writer.add_scalar('train/acc', acc.avg, epoch)
    return


def validate(val_loader, model, criterion, epoch):
    model.eval()
    acc = AverageMeter()
    losses = AverageMeter()
    meta_data = []
    pbar = tqdm(val_loader)
    for i, (images, target, meta) in enumerate(pbar):
        target = target.cuda(async=True)
        image_var = Variable(images, volatile=True)
        label_var = Variable(target, volatile=True)

        y_pred = model(image_var)
        # udpate loss metric
        loss = criterion(y_pred, label_var)
        losses.update(loss.data[0], images.size(0))

        # update accuracy metric on the GPU
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        acc.update(prec1[0], images.size(0))

        sm = nn.Softmax()
        sm_pred = sm(y_pred).data.cpu().numpy()
        # y_norm_probs = sm_pred[:, 0] # p(normal)
        y_pred_probs = sm_pred[:, 1]  # p(abnormal)

        meta_data.append(
            pd.DataFrame({
                'img_filename': meta['img_filename'],
                'y_true': meta['y_true'].numpy(),
                'y_pred_probs': y_pred_probs,
                'patient': meta['patient'].numpy(),
                'study': meta['study'].numpy(),
                'image_num': meta['image_num'].numpy(),
                'encounter': meta['encounter'],
            }))

        pbar.set_description("VALIDATION[{}/{}]".format(i, len(val_loader)))
        pbar.set_postfix(
            acc="{acc.val:.4f} ({acc.avg:.4f})".format(acc=acc),
            loss="{loss.val:.4f} ({loss.avg:.4f})".format(loss=losses))
    df = pd.concat(meta_data)
    ab = df.groupby(['encounter'])['y_pred_probs', 'y_true'].mean()
    ab['y_pred_round'] = ab.y_pred_probs.round()
    ab['y_pred_round'] = pd.to_numeric(ab.y_pred_round, downcast='integer')

    f1_s = f1_score(ab.y_true, ab.y_pred_round)
    prec_s = precision_score(ab.y_true, ab.y_pred_round)
    rec_s = recall_score(ab.y_true, ab.y_pred_round)
    acc_s = accuracy_score(ab.y_true, ab.y_pred_round)
    tb_writer.add_scalar('val/f1_score', f1_s, epoch)
    tb_writer.add_scalar('val/precision', prec_s, epoch)
    tb_writer.add_scalar('val/recall', rec_s, epoch)
    tb_writer.add_scalar('val/accuracy', acc_s, epoch)
    # return the metric we want to evaluate this model's performance by
    return f1_s


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
