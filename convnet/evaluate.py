from vis_util import visualize
from utils import prepare_input
from vis_util import post_process
from models.model import SuctionModel18, SuctionModel50
from models.model import SuctionRefineNet, SuctionRefineNetLW
from models.model import SuctionPSPNet

import os
import argparse
import time
import tqdm

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches


cudnn.benchmark = True
cudnn.enabled = True

class Options(object):
    def __init__(self):
        p = os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]
        self.proj_path = '/'.join(p)
        self.data_path = os.path.join('/'.join(p[:-1]), 'dataset/')
        self.test_img_path = os.path.join(self.data_path, 'test-split.txt')
        self.model_path = ''
        self.img_height =  480
        self.img_width = 640
        self.output_scale = 8
        self.n_class = 3
        self.arch = 'resnet18'
        self.device = 'cuda:0'
        self.visualize = False # don't visualize it, there is a memory bug on matplotlib

if __name__ == "__main__":
    model_choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'rfnet50', 'rfnet101', 'rfnet152', 'pspnet50', 'pspnet101', 'pspnet18', 'pspnet34']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--arch', metavar='arch', default='resnet101', choices=model_choices,
        help='model architecture: ' + ' | '.join(model_choices) + ' (default: resnet18)'
    )
    parser.add_argument(
        '--datapath', dest='data_path', default='', help='suction grasp dataset path',
    )
    parser.add_argument(
        '--checkpoint', required=True, help='model path',
    )
    parser.add_argument(
        '--visualize', action='store_true', help='use amp on training',
    )
    args = parser.parse_args()

    options = Options()
    options.arch = args.arch if args.arch != '' else options.arch
    options.data_path = args.data_path if args.data_path != '' else options.data_path
    options.model_path = args.checkpoint

    ## prepare plotting canvas
    if options.visualize:
        plt.ion()
        plt.show()
        fig = plt.gcf()
        fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)
        fig.canvas.set_window_title('Evaluation Result')
        # fig, ax = plt.subplots(2,3)
        # fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)
        # fig.canvas.set_window_title('Evaluation Result')

    ## prepare model
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

    test_img_list = open(os.path.join(options.data_path, 'test-split.txt')).read().splitlines()
    test_len = len(test_img_list)

    metrics_data = np.zeros((test_len, 4), dtype=np.int64)  # [tp, tn, fp, fn]
    inf_time = np.zeros((test_len,1), dtype=np.float64)     # inference + post process time
    for n, input_path in enumerate(test_img_list):
        print(input_path, "%d/%d: " % (n, test_len), end='  ')
        
        color_in = Image.open(os.path.join(options.data_path, 'color-input', input_path + '.png'))
        color_bg = Image.open(os.path.join(options.data_path, 'color-background', input_path + '.png'))
        depth_in = Image.open(os.path.join(options.data_path, 'depth-input', input_path + '.png'))
        depth_bg = Image.open(os.path.join(options.data_path, 'depth-background', input_path + '.png'))
        label = Image.open(os.path.join(options.data_path, 'label', input_path + '.png'))
        cam_intrinsic = np.loadtxt(os.path.join(options.data_path, 'camera-intrinsics', input_path + '.txt'))
        img_input = prepare_input(color_in, depth_in, options.device)

        ## forward pass
        t = time.time()
        output = model(img_input)

        ## get segmentation class prediction
        cls_pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0).astype(np.float64)
        cls_pred = resize(cls_pred, (options.img_height, options.img_width),
            anti_aliasing=True, mode='reflect')
        
        ## get the probability of suction area (index 1)
        pred = np.squeeze(output.data.cpu().numpy(), axis=0)[1,:,:]
        pred = resize(pred, (options.img_height, options.img_width), 
            anti_aliasing=True, mode='reflect')
        affordance = (pred - pred.min()) / (pred.max() - pred.min())

        ## post-processing
        surface_norm, affordance_map, cls_pred = post_process(affordance, cls_pred,
            color_in, color_bg, depth_in, depth_bg, cam_intrinsic)
        inf_time[n,0] = time.time() - t

        affordance_map = gaussian_filter(affordance_map, 7)
        surface_norm = np.interp(surface_norm,
            (surface_norm.min(), surface_norm.max()), (0.0, 1.0))
        color_in = np.asarray(color_in, dtype=np.float64) / 255

        ## calculate metrics
        affordance_img = (affordance_map * 255).astype(np.uint8)
        label_np = np.asarray(label, dtype=np.uint8)
        threshold = np.percentile(affordance_img, 99) ## top 1% prediction
        # threshold = ((affordance_map.max() - 0.0001)*255).astype(np.uint8) ## top 1 prediction
        tp = np.sum(np.logical_and((affordance_img > threshold), (label_np == 128)).astype(np.int))
        fp = np.sum(np.logical_and((affordance_img > threshold), (label_np == 0)).astype(np.int))
        tn = np.sum(np.logical_and((affordance_img <= threshold), (label_np == 0)).astype(np.int))
        fn = np.sum(np.logical_and((affordance_img <= threshold), (label_np == 128)).astype(np.int))

        precision = tp/(tp + fp) if (tp + fp) != 0 else 0
        recall = tp/(tp + fn) if (tp + fn) != 0 else 0
        iou = tp/(tp + fp + fn) if (tp + fp + fn) != 0 else 0
        metrics_data[n,:] = np.array([tp, tn, fp, fn])
        print("%.8f  %.8f  %.8f  %.8f" % (precision, recall, iou, inf_time[n,0]))

        ## visualize
        if options.visualize:
            cmap = cm.get_cmap('jet')
            affordance_color = cmap(affordance_map)[:,:,:-1] # ommit last channel (get rgb)
            affordance_viz = affordance_color*0.5 + color_in*0.5

            cmap_cls = cm.get_cmap('Paired')
            cls_img = cmap_cls(cls_pred)[:,:,:-1]
            cls_img = cls_img*0.5 + color_in*0.5

            ## best picking point
            max_point = np.argmax(affordance_map)
            max_point = (max_point//affordance_color.shape[1], max_point%affordance_color.shape[1])
            max_circ = patches.Circle(np.flip(max_point), radius=8, fill=False, linewidth=4.0, color='k')

            depth_np = np.array(depth_in, dtype=np.float64) / 65536
            plt.subplot(2,3,1)
            plt.imshow(color_in)
            plt.yticks([]); plt.xticks([])
            plt.subplot(2,3,2)
            plt.imshow(affordance_viz)
            plt.yticks([]); plt.xticks([])
            plt.subplot(2,3,3)
            plt.imshow(surface_norm)
            plt.yticks([]); plt.xticks([])
            plt.subplot(2,3,4)
            plt.imshow(label_np)
            plt.yticks([]); plt.xticks([])
            plt.subplot(2,3,5)
            plt.imshow(cls_img)
            plt.yticks([]); plt.xticks([])
            plt.subplot(2,3,6)
            plt.imshow(depth_np)
            plt.yticks([]); plt.xticks([])
            plt.draw()
            plt.pause(0.01)
            
            # ax[0, 0].imshow(color_in)
            # ax[1, 0].imshow(label_np)
            # ax[0, 1].imshow(affordance_viz)
            # ax[1, 1].imshow(cls_img)
            # ax[0, 2].imshow(surface_norm)
            # ax[1, 2].imshow(depth_np, cmap='gray')
            # ax[0, 0].set_axis_off()
            # ax[0, 1].set_axis_off()
            # ax[0, 2].set_axis_off()
            # ax[1, 0].set_axis_off()
            # ax[1, 1].set_axis_off()
            # ax[1, 2].set_axis_off()
            # fig.canvas.draw()
            # fig.canvas.flush_events()
    
    metrics_data[np.isnan(metrics_data)] = 0
    s = np.sum(metrics_data, axis=0)
    precision = s[0]/(s[0]+s[2])
    recall = s[0]/(s[0]+s[3])
    mean_iou = s[0]/(s[0]+s[2]+s[3])
    print(precision, recall, mean_iou, np.sum(inf_time), np.mean(inf_time))

    data = np.append(metrics_data, inf_time, axis=1)
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.txt')
    np.savetxt(result_path, metrics_data, fmt='%.10f')