## inference script 
from vis_util import visualize
from utils import prepare_input, post_process_output
from vis_util import post_process
import models

import os
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image

cudnn.benchmark = True
cudnn.enabled = True

class Options(object):
    def __init__(self):
        p = os.path.dirname(os.path.abspath(__file__)).split('/')
        self.proj_path = '/'.join(p)
        self.data_path = os.path.join('/'.join(p[:-1]), 'dataset/')
        self.model_path = ''
        self.img_height =  480
        self.img_width = 640
        self.output_scale = 8
        self.n_class = 3
        self.arch = ''
        self.device = 'cuda:0'

if __name__ == "__main__":
    model_choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'rfnet50', 'rfnet101', 'rfnet152', 'pspnet50', 'pspnet101', 'pspnet18', 'pspnet34']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--checkpoint', required=True, help='model path',
    )
    parser.add_argument(
        '-a', '--arch', metavar='arch', default='', choices=model_choices,
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
    options.data_path = args.data_path if args.data_path != '' else options.data_path
    options.model_path = args.checkpoint

    ## get model
    checkpoint = torch.load(options.model_path)
    options.arch = args.arch if args.arch != '' else checkpoint['arch']
    model = models.build_model(options.arch)
    model.eval()
    model.to(options.device)

    ## get model weight
    model.load_state_dict(checkpoint['model_state_dict'])

    ## get random or user selected image input 
    sample_list = open(os.path.join(options.data_path, 'test-split.txt')).read().splitlines()
    input_file = np.random.choice(sample_list, 1)[0] if args.img_input == '' else args.img_input

    rgb_in = Image.open(os.path.join(options.data_path, 'color-input', input_file + '.png'))
    depth_in = Image.open(os.path.join(options.data_path, 'depth-input', input_file + '.png'))
    label = Image.open(os.path.join(options.data_path, 'label', input_file + '.png'))
    # rgb_bg = Image.open(os.path.join(options.data_path, 'color-background', input_file + '.png'))
    # depth_bg = Image.open(os.path.join(options.data_path, 'depth-background', input_file + '.png'))
    # cam_intrinsic = np.loadtxt(os.path.join(options.data_path, 'camera-intrinsics', input_file + '.txt'))

    rgb_input, ddd_input = prepare_input(rgb_in, depth_in, options.device)

    ## inference
    print('computing inference: ', input_file)
    with torch.no_grad():
        t = time.time()
        output = model(rgb_input, ddd_input)

        ## post-processing
        cls_pred, affordance_map = post_process_output(output, options)
        tm = time.time() - t

    affordance_img = (affordance_map * 255).astype(np.uint8)
    label_np = np.asarray(label, dtype=np.uint8)
    threshold = np.percentile(affordance_img, 99) ## top 1% prediction
    tp = np.sum(np.logical_and((affordance_img > threshold), (label_np == 128)).astype(np.int))
    fp = np.sum(np.logical_and((affordance_img > threshold), (label_np == 0)).astype(np.int))
    tn = np.sum(np.logical_and((affordance_img <= threshold), (label_np == 0)).astype(np.int))
    fn = np.sum(np.logical_and((affordance_img <= threshold), (label_np == 128)).astype(np.int))

    precision = tp/(tp + fp) if (tp + fp) != 0 else 0
    recall = tp/(tp + fn) if (tp + fn) != 0 else 0
    iou = tp/(tp + fp + fn) if (tp + fp + fn) != 0 else 0
    print("inference time: ", tm)
    print("memory allocated: ", torch.cuda.max_memory_allocated()/2**30, "GB")
    print("   prec       recall       iou   ")
    print("%.8f  %.8f  %.8f" % (precision, recall, iou))
    
    visualize(affordance_map, cls_pred, np.array(rgb_in))
