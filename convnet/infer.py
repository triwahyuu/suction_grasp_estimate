## inference script 
from vis_util import visualize
from utils import post_process, prepare_input
from model import SuctionModel18, SuctionModel50

import os
import argparse
import time

import torch
import numpy as np
from PIL import Image
from skimage.transform import resize


class Options(object):
    def __init__(self):
        self.data_path = '/home/tri/skripsi/dataset/'
        self.model_path = '/home/tri/skripsi/result/04_resnet18_dropout/20190429_222828/model_best.pth.tar'
        self.img_height =  480
        self.img_width = 640
        self.output_scale = 8
        self.arch = 'resnet18'
        self.device = 'cuda:0'

if __name__ == "__main__":
    model_choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--arch', metavar='arch', default='resnet18', choices=model_choices,
        help='model architecture: ' + ' | '.join(model_choices) + ' (default: resnet18)'
    )
    parser.add_argument(
        '--datapath', dest='data_path', default='', help='suction grasp dataset path',
    )
    parser.add_argument(
        '--checkpoint', default='', help='model path',
    )
    parser.add_argument(
        '--img_input', default='', help='input image index, eg: 00001-1',
    )
    args = parser.parse_args()
    np.random.seed(int(time.time()))

    options = Options()
    options.arch = args.arch if args.arch != '' else options.arch
    options.data_path = args.data_path if args.data_path != '' else options.data_path
    options.model_path = args.checkpoint if args.checkpoint != '' else options.model_path

    ## get model
    if args.arch == 'resnet18' or args.arch == 'resnet34':
        model = SuctionModel18(options)
    elif args.arch == 'resnet50' or args.arch == 'resnet101' or args.arch == 'resnet152':
        model = SuctionModel50(options)
    checkpoint = torch.load(options.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    model.to(options.device)

    ## get random or user selected image input 
    sample_list = open(os.path.join(options.data_path, 'test-split.txt')).read().splitlines()
    input_file = np.random.choice(sample_list, 1)[0] if args.img_input == '' else args.img_input
    color = Image.open(os.path.join(options.data_path, 'color-input', input_file + '.png'))
    depth = Image.open(os.path.join(options.data_path, 'depth-input', input_file + '.png'))

    img_input = prepare_input(color, depth, options.device)

    ## inference
    print('computing forward pass: ', input_file)
    output = model(img_input)

    cls_pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0).astype(np.float64)
    cls_pred = resize(cls_pred, (options.img_height, options.img_width), 
        anti_aliasing=True, mode='reflect')

    pred = np.squeeze(output.data.cpu().numpy(), axis=0)[1,:,:]
    pred = resize(pred, (options.img_height, options.img_width), 
        anti_aliasing=True, mode='reflect')
    affordance = (pred - pred.min()) / (pred.max() - pred.min())

    ## visualize
    print('post process...')
    rgb_in = Image.open(os.path.join(options.data_path, 'color-input', input_file + '.png'))
    rgb_bg = Image.open(os.path.join(options.data_path, 'color-background', input_file + '.png'))
    depth_in = Image.open(os.path.join(options.data_path, 'depth-input', input_file + '.png'))
    depth_bg = Image.open(os.path.join(options.data_path, 'depth-background', input_file + '.png'))
    cam_intrinsic = np.loadtxt(os.path.join(options.data_path, 'camera-intrinsics', input_file + '.txt'))

    surface_norm, affordance_map, cls_pred = post_process(affordance, cls_pred, rgb_in, rgb_bg,
        depth_in, depth_bg, cam_intrinsic)
    
    print('visualize...')
    visualize(affordance_map, surface_norm, cls_pred, np.array(rgb_in))
