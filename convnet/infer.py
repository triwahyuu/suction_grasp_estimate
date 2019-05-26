## inference script 
from vis_util import visualize
from utils import prepare_input
from vis_util import post_process
from models.model import SuctionModel18, SuctionModel50
from models.model import SuctionRefineNet, SuctionRefineNetLW
from models.model import SuctionPSPNet

import os
import argparse
import time

import torch
import numpy as np
from PIL import Image
from skimage.transform import resize
import scipy.special


class Options(object):
    def __init__(self):
        p = os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]
        self.proj_path = '/'.join(p)
        self.data_path = os.path.join('/'.join(p[:-1]), 'dataset/')
        self.model_path = ''
        self.img_height =  480
        self.img_width = 640
        self.output_scale = 8
        self.n_class = 3
        self.arch = 'resnet18'
        self.device = 'cuda:0'

if __name__ == "__main__":
    model_choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'rfnet50', 'rfnet101', 'rfnet152', 'pspnet50', 'pspnet101', 'pspnet18', 'pspnet34']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--checkpoint', required=True, help='model path',
    )
    parser.add_argument(
        '-a', '--arch', metavar='arch', default='resnet18', choices=model_choices,
        help='model architecture: ' + ' | '.join(model_choices) + ' (default: resnet18)'
    )
    parser.add_argument(
        '--datapath', dest='data_path', default='', help='suction grasp dataset path',
    )
    parser.add_argument(
        '--img-input', default='', help='input image index, eg: 00001-1',
    )
    args = parser.parse_args()
    np.random.seed(int(time.time()))

    options = Options()
    options.arch = args.arch if args.arch != '' else options.arch
    options.data_path = args.data_path if args.data_path != '' else options.data_path
    options.model_path = args.checkpoint

    ## get model
    if args.arch == 'resnet18' or args.arch == 'resnet34':
        model = SuctionModel18(options)
    elif args.arch == 'resnet50' or args.arch == 'resnet101' or args.arch == 'resnet152':
        model = SuctionModel50(options)
    elif args.arch == 'rfnet50' or args.arch == 'rfnet101' or args.arch == 'rfnet152':
        backbone = 'rfnet'
        model = SuctionRefineNetLW(options)
    elif args.arch == 'pspnet50' or args.arch == 'pspnet101' \
            or args.arch == 'pspnet18' or args.arch == 'pspnet34':
        model = SuctionPSPNet(options)
    model.eval()
    model.to(options.device)

    ## get model weight
    checkpoint = torch.load(options.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

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
    # aff_sigmoid = scipy.special.expit(pred)
    # affordance = pred


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
