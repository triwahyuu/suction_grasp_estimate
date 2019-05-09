## inference script 
from vis_util import post_process, visualize
from model import SuctionModel18, SuctionModel50
import os
import argparse
import time

import torch
from torchvision.transforms import ToTensor, Normalize, Resize
import numpy as np
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
options = Struct(\
    data_path = '/home/tri/skripsi/dataset/',
    model_path = '/home/tri/skripsi/result/04_resnet18_dropout/20190429_222828/model_best.pth.tar',
    img_height =  480,
    img_width = 640,
    output_scale = 8,
    arch = 'resnet18',
    device = 'cuda:0'
)

def prepare_input(color, depth, device):
    to_tensor = ToTensor()
    normalize = Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    color_img = normalize(to_tensor(color))
    color_img = color_img.view(1, color_img.size(0), color_img.size(1), color_img.size(2))

    depth = (to_tensor(np.asarray(depth, dtype=np.float32)) * 65536/10000).clamp(0.0, 1.2)
    depth_img = torch.cat([depth, depth, depth], 0)
    depth_img = normalize(depth_img)
    depth_img = depth_img.view(1, depth_img.size(0), depth_img.size(1), depth_img.size(2))

    img_input = [color_img.to(device), depth_img.to(device)]
    return img_input

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
        '--checkpoint', default='', help='suction grasp dataset path',
    )
    args = parser.parse_args()
    # np.random.seed(time.time())

    options.arch = args.arch if args.arch != '' else options.arch
    options.data_path = args.data_path if args.data_path != '' else options.data_path
    options.model_path = args.checkpoint if args.checkpoint != '' else options.model_path

    ## transforms
    to_tensor = ToTensor()
    normalize = Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    ## get model
    if args.arch == 'resnet18' or args.arch == 'resnet34':
        model = SuctionModel18(options)
    elif args.arch == 'resnet50' or args.arch == 'resnet101' or args.arch == 'resnet152':
        model = SuctionModel50(options)
    checkpoint = torch.load(options.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    model.to(options.device)

    ## get random image input
    sample_list = open(os.path.join(options.data_path, 'test-split.txt')).read().splitlines()
    input_file = np.random.choice(sample_list, 1)[0]
    # color_img = to_tensor(Image.open(os.path.join(options.data_path, 'test', 'test-image.color.png')))
    color = Image.open(os.path.join(options.data_path, 'color-input', input_file + '.png'))
    # depth = Image.open(os.path.join(options.data_path, 'test', 'test-image.depth.png'))
    depth = Image.open(os.path.join(options.data_path, 'depth-input', input_file + '.png'))

    img_input = prepare_input(color, depth, options.device)

    ## inference
    print('computing forward pass: ', input_file)
    output = model(img_input)

    cls_pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
    cls_pred = cls_pred.astype(np.float64)
    cls_pred = resize(cls_pred, (options.img_height, options.img_width), 
        anti_aliasing=True, mode='reflect')

    pred = np.squeeze(output.data.cpu().numpy(), axis=0)[1,:,:]
    pred = resize(pred, (options.img_height, options.img_width), 
        anti_aliasing=True, mode='reflect')
    affordance = (pred - pred.min()) / (pred.max() - pred.min())
    # affordance[affordance >= 1] = 0.9999 # normalize
    # affordance[affordance < 0] = 0

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
