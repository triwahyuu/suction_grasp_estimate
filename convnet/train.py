## TODO:
## - implement mixed precision training with apex
# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
# https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
# https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
# https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c
## - optimize data loading: (later)
# https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
# https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740


from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import SuctionDatasetNew
from models.model import build_model
from utils import label_accuracy_score

import pytz
import math
import os
import os.path as osp
import shutil
import datetime
import tqdm
import argparse

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


cudnn.benchmark = True
cudnn.enabled = True


class Options:
    def __init__(self):
        p = osp.dirname(osp.abspath(__file__)).split('/')[:-1]
        self.proj_path = '/'.join(p)
        self.data_path = osp.join('/'.join(p[:-1]), 'dataset/')
        self.sample_path = osp.join(self.data_path, 'train-split.txt')
        self.img_height =  480
        self.img_width = 640
        self.batch_size = 2
        self.n_class = 3
        self.output_scale = 8
        self.shuffle = True
        self.learning_rate = 0.001
        self.momentum = 0.99
        # available architecture: 
        # [resnet18, resnet34, resnet50, resnet101, rfnet50, rfnet101]
        self.arch = 'resnet18'


def BNtoFixed(m):
    # From https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/torchtools.py
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    if m.__class__.__name__.find('BatchNorm2d') != -1:
        m.eval()


## based on:
## https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
class Trainer(object):

    def __init__(self, model, optimizers, loss, train_loader, val_loader, 
                 output_path, log_path, max_epoch=200, max_iter=None, arch='resnet18',
                 cuda=True, interval_validate=None, freeze_bn=True, use_amp=False, 
                 **kwargs):
        self.cuda = cuda

        self.model = model
        self.arch = arch
        self.freeze_bn = freeze_bn

        ## if using rfnet, optimizers is an array of 2 optimizer
        ## optimizer for encoder and decoder
        self.optim = optimizers
        self.optim_dec = None
        if 'rfnet' in self.arch:
            self.optim = optimizers[0]
            self.optim_dec = optimizers[1]

        self.train_loader = train_loader
        self.val_loader = val_loader

        ## criterion
        self.alphas = None
        self.lambdas = None
        if 'bisenet' in self.arch:
            self.criterion = loss[0].cuda() if self.cuda else loss[0]
            self.crit_aux1 = loss[1].cuda() if self.cuda else loss[1]
            self.crit_aux2 = loss[2].cuda() if self.cuda else loss[2]
            self.alphas = kwargs['loss_params']
        elif 'icnet' in self.arch:
            self.criterion = loss[2].cuda() if self.cuda else loss[2]
            self.crit_sub24 = loss[1].cuda() if self.cuda else loss[1]
            self.crit_sub4 = loss[0].cuda() if self.cuda else loss[0]
            self.lambdas = kwargs['loss_params']
        else:
            self.criterion = loss.cuda() if self.cuda else loss

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Jakarta'))
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

        self.use_amp = APEX_AVAILABLE if use_amp else False
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.best_mean_iu = 0
        self.best_loss = 999.999
        self.writer = SummaryWriter(log_dir=os.path.join(log_path, 'tb'))
        torch.manual_seed(1234)

    def validate(self):
        n_class = self.train_loader.dataset.n_class

        # os.system('play -nq -t alsa synth {} sine {}'.format(0.3, 440)) # sound an alarm

        val_loss = 0
        label_trues, label_preds = [], []
        for batch_idx, (input_img, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='  val %d' % self.epoch, ncols=80,
                leave=False):
                
            ## validate
            with torch.no_grad():
                self.model.eval()
                if self.cuda:
                    input_img[0], input_img[1] = input_img[0].cuda(), input_img[1].cuda()
                    target = target.cuda()

                output = self.model(input_img)
                output = nn.functional.interpolate(output, size=target.size()[1:],
                    mode='bilinear', align_corners=False)
                
                loss = self.criterion(output, target)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    print('\n\n', loss)
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data / len(input_img[0])

            ## some stats
            lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
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
        is_loss_best = val_loss < self.best_loss
        if is_best:
            self.best_mean_iu = mean_iu
        if is_loss_best:
            self.best_loss = val_loss
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
            'best_loss': self.best_loss,
        }, osp.join(self.output_path, 'checkpoint.pth.tar'))
        if self.arch == 'rfnet':
            torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'optim_dec_state_dict': self.optim_dec.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
            'best_loss': self.best_loss,
        }, osp.join(self.output_path, 'checkpoint.pth.tar'))

        if is_best:
            shutil.copy(osp.join(self.output_path, 'checkpoint.pth.tar'),
                        osp.join(self.output_path, 'model_best.pth.tar'))
        if is_loss_best:
            shutil.copy(osp.join(self.output_path, 'checkpoint.pth.tar'),
                        osp.join(self.output_path, 'model_loss_best.pth.tar'))
        
        self.writer.add_scalar('val/loss', val_loss, self.epoch)
        self.writer.add_scalar('val/accuracy', metrics[0], self.epoch)
        self.writer.add_scalar('val/acc_class', metrics[1], self.epoch)
        self.writer.add_scalar('val/mean_iu', metrics[2], self.epoch)
        self.writer.add_scalar('val/fwacc', metrics[3], self.epoch)

        if self.training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        if self.freeze_bn:
            self.model.apply(BNtoFixed)

        n_class = self.train_loader.dataset.n_class

        m = []
        for batch_idx, (input_img, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc=' epoch %d' % self.epoch, ncols=80, leave=False):
            
            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            ## prepare input and label
            if self.cuda:
                input_img[0], input_img[1] = input_img[0].cuda(), input_img[1].cuda()
                target = target.cuda()

            ## main training function
            ## compute output of feed forward
            output = self.model(input_img)
            if 'bisenet' in self.arch:
                out_sup1 = nn.functional.interpolate( # supervise 1 output
                    output[1], size=target.size()[1:], mode='bilinear', align_corners=False
                )
                out_sup2 = nn.functional.interpolate( # supervise 2 output
                    output[2], size=target.size()[1:], mode='bilinear', align_corners=False
                )
                output = nn.functional.interpolate( # main output
                    output[0], size=target.size()[1:], mode='bilinear', align_corners=False
                )
            elif 'icnet' in self.arch:
                out_sub24 = nn.functional.interpolate(
                    output[1], size=target.size()[1:], mode='bilinear', align_corners=False
                )
                out_sub4 = nn.functional.interpolate(
                    output[2], size=target.size()[1:], mode='bilinear', align_corners=False
                )
                output = nn.functional.interpolate(
                    output[0], size=target.size()[1:], mode='bilinear', align_corners=False
                )
            else:
                output = nn.functional.interpolate(
                    output, size=target.size()[1:], mode='bilinear', align_corners=False
                )

            ## compute loss and backpropagate
            loss = None
            if 'bisenet' in self.arch:
                loss_p = self.criterion(output, target)
                loss_a1 = self.crit_aux1(out_sup1, target)
                loss_a2 = self.crit_aux2(out_sup2, target)
                # loss_p = self.criterion(output, target)
                # loss_a1 = self.crit_aux1(out_sup1, target)
                # loss_a2 = self.crit_aux2(out_sup2, target)
                loss = loss_p + self.alphas[0] * loss_a1 + self.alphas[1] * loss_a2
            elif 'icnet' in self.arch:
                loss_sub124 = self.criterion(output, target)
                loss_sub24 = self.crit_sub24(out_sub24, target)
                loss_sub4 = self.crit_sub4(out_sub4, target)
                loss = self.lambdas[0] * loss_sub4 + \
                       self.lambdas[1] * loss_sub24 + \
                       self.lambdas[2] * loss_sub124
            else:
                loss = self.criterion(output, target)

            loss_data = loss.data.item()
            self.optim.zero_grad()
            if 'rfnet' in self.arch:
                self.optim_dec.zero_grad()

            if self.use_amp:
                with amp.scale_loss(loss, self.optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            self.optim.step()
            if 'rfnet' in self.arch:
                self.optim_dec.step()

            ## the stats
            lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
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

            if self.max_iter != None and self.iteration >= self.max_iter:
                break
        
        m = np.mean(np.array(m), axis=0)
        self.writer.add_scalar('train/loss', loss_data, self.epoch)
        self.writer.add_scalar('train/accuracy', m[0], self.epoch)
        self.writer.add_scalar('train/acc_class', m[1], self.epoch)
        self.writer.add_scalar('train/mean_iu', m[2], self.epoch)
        self.writer.add_scalar('train/fwacc', m[3], self.epoch)

    def train(self):
        self.training = True
        self.max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader))) \
            if self.max_iter != None else self.max_epoch
        if self.cuda:
            torch.cuda.empty_cache()
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            ## do training
            self.train_epoch()

            ## validate
            self.validate()

            if self.max_iter != None and self.iteration >= self.max_iter:
                break
                
if __name__ == "__main__":
    model_choices = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'rfnet50', 'rfnet101', 'rfnet152', 
        'pspnet18', 'pspnet34', 'pspnet50', 'pspnet101',
        'bisenet18', 'bisenet34', 'bisenet50', 'bisenet101',
        'icnet18', 'icnet34', 'icnet50', 'icnet101',
    ]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--resume', default='', type=str, help='checkpoint path'
    )
    parser.add_argument(
        '-a', '--arch', metavar='arch', default='bisenet18', choices=model_choices,
        help='model architecture: ' + ' | '.join(model_choices) + ' (default: resnet101)'
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
        '--max-epoch', default=60, type=int, help='maximum epoch for training',
    )
    parser.add_argument(
        '--use-cpu', dest='use_cpu', action='store_true', help='use cpu on training',
    )
    parser.add_argument(
        '--use-amp', dest='use_amp', action='store_true', help='use amp on training',
    )
    parser.add_argument('--opt-level', default='O2', type=str)
    args = parser.parse_args()
    options = Options()

    result_path = osp.join(options.proj_path, 'result')
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    options.arch = args.arch
    device = torch.device("cpu" if args.use_cpu else "cuda:0")

    if args.data_path != '':
        options.data_path = args.data_path
        options.sample_path = os.path.join(args.data_path, 'train-split.txt')
    if args.result_path != '':
        result_path = args.result_path

    ## prepare model
    model = build_model(args.arch, options)
    model.apply(BNtoFixed)
    model.to(device)


    ## Loss
    loss_params = None
    if 'bisenet' in args.arch:
        # crit_principal = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]), reduction='mean')
        # crit_aux1 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]), reduction='mean')
        # crit_aux2 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]), reduction='mean')
        crit_principal = nn.BCEWithLogitsLoss(weight=torch.Tensor([1, 1, 0]))
        crit_aux1 = nn.BCEWithLogitsLoss(weight=torch.Tensor([1, 1, 0]))
        crit_aux2 = nn.BCEWithLogitsLoss(weight=torch.Tensor([1, 1, 0]))
        criterion = [crit_principal, crit_aux1, crit_aux2]
        loss_params = [1, 1]    # aux1, aux2
    elif 'icnet' in args.arch:
        crit_sub4 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]), reduction='mean')
        crit_sub24 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]), reduction='mean')
        crit_sub124 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]), reduction='mean')
        criterion = [crit_sub4, crit_sub24, crit_sub124]
        loss_params = [0.16, 0.4, 1.0] # sub4, sub24, sub124
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0])).to(device)

    ## Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if 'rfnet' in args.arch:
        import re
        enc_params = []
        dec_params = []
        for k,v in model.named_parameters():
            if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
                enc_params.append(v)
            else:
                dec_params.append(v)
        optim_enc = optim.SGD(enc_params, lr=args.lr, momentum=args.momentum)
        optim_dec = optim.SGD(dec_params, lr=args.lr, momentum=args.momentum)
        optimizer = [optim_enc, optim_dec]


    ## initialize amp
    if args.use_amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level, 
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )
    

    ## resume training
    start_epoch = 0
    start_iteration = 0
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        if 'rfnet' in args.arch:
            optim_enc.load_state_dict(checkpoint['optim_state_dict'])
            optim_dec.load_state_dict(checkpoint['optim_dec_state_dict'])
        else:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
    

    ## dataset
    # if args.arch == 'resnet18' or args.arch == 'pspnet18':
    #     options.batch_size = 4
    train_dataset = SuctionDatasetNew(options, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
        shuffle=options.shuffle, num_workers=3)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=options.shuffle)

    val_dataset = SuctionDatasetNew(options, 
        sample_list=os.path.join(options.data_path, 'test-split.txt'),
        mode='val')
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
        shuffle=False, num_workers=3)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    
    ## the main deal
    trainer = Trainer(model=model, optimizers=optimizer, loss=criterion,
        train_loader=train_loader, val_loader=val_loader, arch=args.arch,
        output_path=os.path.join(result_path, now), log_path=result_path,
        max_epoch=args.max_epoch, cuda=(not args.use_cpu), use_amp=args.use_amp,
        loss_params=loss_params)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    if args.resume != '':
        trainer.best_mean_iu = checkpoint['best_mean_iu']
        trainer.best_loss = checkpoint['best_loss']
    trainer.train()
    # trainer.validate()