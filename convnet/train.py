## TODO:
## - data logging with tensorboard
# https://github.com/foolwood/deepmask-pytorch/blob/master/tools/train.py
# http://www.erogol.com/use-tensorboard-pytorch/
## - optimize data logging:
# https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
# https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740


from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from apex import fp16_utils

from dataset import SuctionDatasetNew
from model import SuctionModel18, SuctionModel50
from utils import label_accuracy_score

import pytz
import math
import os
import os.path as osp
import shutil
import datetime
import tqdm
import argparse

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
options = Struct(\
        data_path = '/home/tri/skripsi/dataset/',
        sample_path = '/home/tri/skripsi/dataset/train-split.txt',
        img_height =  480,
        img_width = 640,
        batch_size = 2,
        n_class = 3,
        output_scale = 8,
        shuffle = True,
        learning_rate = 0.001,
        momentum = 0.99,
        arch = 'resnet18'
    )

def BNtoFixed(m):
    # From https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/torchtools.py
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    if type(m) == nn.BatchNorm2d:
        m.eval()


## based on:
## https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
class Trainer(object):

    def __init__(self, model, optimizer, criterion, train_loader, val_loader, 
                 output_path, log_path, max_iter, cuda=True, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        if self.cuda:
            self.criterion = self.criterion.cuda()

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Jakarta'))
        self.bn2fixed = True
        self.training = False

        
        self.interval_validate = len(self.train_loader) if interval_validate is None \
            else interval_validate

        self.output_path = output_path  # output path
        if not osp.exists(self.output_path):
            os.makedirs(self.output_path)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.output_path, 'log.csv')):
            with open(osp.join(self.output_path, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        self.writer = SummaryWriter(log_dir=os.path.join(log_path, 'tb'))
        torch.manual_seed(1234)

    def validate(self):
        # self.model.eval()
        n_class = self.train_loader.dataset.n_class

        os.system('play -nq -t alsa synth {} sine {}'.format(0.3, 440)) # sound an alarm

        val_loss = 0
        label_trues, label_preds = [], []
        for batch_idx, (input_img, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='  validation %d' % self.iteration, ncols=80,
                leave=False):
                
            ## validate
            with torch.no_grad():
                if self.cuda:
                    input_img[0], input_img[1], target = input_img[0].cuda(), input_img[1].cuda(), target.cuda()

                out = self.model(input_img)
                loss = self.criterion(out, target)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data / len(input_img[0])

            ## some stats
            lbl_pred = out.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for lt, lp in zip(lbl_true, lbl_pred):
                label_trues.append(lt.numpy())
                label_preds.append(lp)

        metrics = label_accuracy_score(label_trues, label_preds, n_class)
        val_loss /= len(self.val_loader)

        with open(osp.join(self.output_path, 'log.csv'), 'a') as f:
            val_loss_str = '%.10f' %(val_loss)
            metrics_str = ['%.10f' %(a) for a in list(metrics)]
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Jakarta')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss_str] + metrics_str + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.output_path, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.output_path, 'checkpoint.pth.tar'),
                        osp.join(self.output_path, 'model_best.pth.tar'))
        
        self.writer.add_scalar('val/loss', val_loss, self.iteration//1000)
        self.writer.add_scalar('val/accuracy', metrics[0], self.iteration//1000)
        self.writer.add_scalar('val/acc_class', metrics[1], self.iteration//1000)
        self.writer.add_scalar('val/mean_iu', metrics[2], self.iteration//1000)
        self.writer.add_scalar('val/fwacc', metrics[3], self.iteration//1000)

        if not self.bn2fixed and self.training:
            self.model.train()

    def train_epoch(self):
        if not self.bn2fixed and self.training:
            self.model.train()

        n_class = self.train_loader.dataset.n_class

        m = []
        for batch_idx, (input_img, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc=' epoch %d' % self.epoch, ncols=80, leave=False):
            
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration != 0:
                self.validate()

            if self.cuda:
                input_img[0], input_img[1] = input_img[0].cuda(), input_img[1].cuda()
                target = target.cuda()
            
            ## main training function
            self.optim.zero_grad()
            out = self.model(input_img)

            loss = self.criterion(out, target)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            ## the stats
            lbl_pred = out.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            metrics = label_accuracy_score(lbl_true, lbl_pred, n_class=n_class)

            with open(osp.join(self.output_path, 'log.csv'), 'a') as f:
                loss_data_str = '%.10f' %(loss_data)
                metrics_str = ['%.10f' %(a) for a in list(metrics)]
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Jakarta')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data_str] + \
                    metrics_str + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            
            m.append(metrics)
            if self.iteration % 100 == 0 and self.iteration != 0:
                m = np.mean(np.array(m), axis=0)
                self.writer.add_scalar('train/loss', loss_data, self.iteration//100)
                self.writer.add_scalar('train/accuracy', m[0], self.iteration//100)
                self.writer.add_scalar('train/acc_class', m[1], self.iteration//100)
                self.writer.add_scalar('train/mean_iu', m[2], self.iteration//100)
                self.writer.add_scalar('train/fwacc', m[3], self.iteration//100)
                m = []

            if self.iteration >= self.max_iter:
                break

    def train(self):
        self.training = True
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
                
if __name__ == "__main__":
    model_choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--resume', default='', type=str, help='checkpoint path'
    )
    parser.add_argument(
        '-a', '--arch', metavar='arch', default='resnet18', choices=model_choices,
        help='model architecture: ' + ' | '.join(model_choices) + ' (default: resnet18)'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-3, help='learning rate',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--datapath', dest='data_path', default='', help='suction grasp dataset path',
    )
    parser.add_argument(
        '--output-path', dest='result_path', default='', help='training result path',
    )
    parser.add_argument(
        '--use-cpu', dest='use_cpu', action='store_true', help='use cpu on training',
    )
    args = parser.parse_args()

    file_path = osp.dirname(osp.abspath(__file__))
    result_path = '/home/tri/skripsi/suction_grasp_estimate/result'
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    options.arch = args.arch
    device = torch.device("cpu" if args.use_cpu else "cuda:0")

    if args.data_path != '':
        options.data_path = args.data_path
        options.sample_path = os.path.join(args.data_path, 'train-split.txt')
    if args.result_path != '':
        result_path = args.result_path

    ## model
    if args.arch == 'resnet18' or args.arch == 'resnet34':
        model = SuctionModel18(options)
    elif args.arch == 'resnet50' or args.arch == 'resnet101' or args.arch == 'resnet152':
        model = SuctionModel50(options)
    model.apply(BNtoFixed)
    
    start_epoch = 0
    start_iteration = 0
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]))
    
    model.to(device)
    criterion.to(device)


    ## dataset
    train_dataset = SuctionDatasetNew(options, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
        shuffle=options.shuffle, num_workers=3)

    val_dataset = SuctionDatasetNew(options, 
        sample_list=os.path.join(options.data_path, 'test-split.txt'),
        mode='val')
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size,\
        shuffle=False, num_workers=3)


    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.resume != '':
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    
    ## the main deal
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion,
        train_loader=train_loader, val_loader=val_loader,
        output_path=os.path.join(result_path, now), log_path=result_path,
        max_iter=500000, cuda=(not args.use_cpu))
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()